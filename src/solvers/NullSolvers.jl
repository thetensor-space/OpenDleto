"""
    NullSolvers

    An interface and options for solving the null spaces that arrise in Dleto.

    [TBD: Surely in Julia there is a package or standard assembly of null space solvers?
    Until I find this, here are some basic options implemented directly.]
"""


using LinearMaps
using LinearAlgebra



export NullSolver, LUSolver, SVDSolver, AutoSolver, solve, solve_nullspace
export register_solver!, available_solvers

abstract type NullSolver end

"""
    solve(method::NullSolver, L::LinearMap; nv::Integer=10)

    - method: An instance of a subtype of `NullSolver` defining the solving method.
    - L: A `LinearMap`.  Rectangular for SVD/LU; the eigen- and Krylov-based
      solvers want the square symmetric A'A.
    - nv: Number of approximate null vectors to compute (default: 10).

    Returns a named tuple `(;vals, vecs)`: the singular-type values and the
    right approximate null vectors as the columns of `vecs`.

    The keyword is `nv` for every implementation.  It used to be `nd` here and
    in the KrylovKit extension while the others used `nv`, so dispatching
    through the symbol factory raised an unsupported-keyword error.
"""
function solve(method::NullSolver, L::LinearMap; nv::Integer=10) end

# ---------------------------------------------------------------- dense gate

"""
    DENSE_LIMIT

Below this dimension a map is always safe to densify, whatever its shape.
"""
const DENSE_LIMIT = 1000

"""
    DENSE_BUDGET_BYTES

How large a dense copy to tolerate above `DENSE_LIMIT`.  A square map of side
a few thousand is a few tens of megabytes and worth densifying for the accuracy;
a rectangular densor map of the same column count is gigabytes and is not.
"""
const DENSE_BUDGET_BYTES = 1.0 * 2^30

"""
    SQUARE_DENSE_LIMIT

Side below which a *square symmetric* map is densified even when a Krylov
eigensolver is loaded.

Above it the eigensolvers win outright.  On the sphere stratification benchmark
(bench/reports/exp3-scaling.csv, bench/stratify-solver-profile.csv) the dense
SVD costs `N` map applications to form the matrix plus `O(N^3)` for the
factorisation, and its time grows as `d^5.2` in the tensor dimension; Arpack
needs a flat ~560 applications whatever `N` and grows as `d^3.3`.  Measured:
N = 165, SVD 0.02s vs Arpack 0.05s; N = 360, 0.12 vs 0.06; N = 630, 0.30 vs
0.17; N = 900, 0.71 vs 0.27; N = 2460, 7.1 vs 1.6; N = 3528, 26 vs 2.8.  The
crossover is near N = 250; 400 leaves the dense path its accuracy advantage
where the difference is hundredths of a second.
"""
const SQUARE_DENSE_LIMIT = 400

"""
    dense_is_cheap(L; dense_limit, dense_budget_bytes, square_dense_limit) -> Bool

Whether `Matrix(L)` is worth forming.

The test is on *bytes*, with the dimension as a shortcut, because dimension
alone is the wrong criterion for a rectangular map.  The derivation--densor
operator at n = 19 is 1083x1083: just over `DENSE_LIMIT`, but 9MB, so
densifying it is free and buys the accuracy of a real SVD.  The densor map at
the same n is 260642x6859: 14GB.  A pure dimension gate would either refuse the
first or accept the second.

A **square** map is the exception, and the byte budget is the wrong gate for
it.  A square symmetric map is what the eigensolvers are built for, and once
one of them is registered they beat the dense SVD from a few hundred rows on
(see `SQUARE_DENSE_LIMIT`), so a square map is densified only below that side
-- unless no matrix-free solver is loaded, in which case the byte budget
decides as before.  A rectangular map keeps the byte gate: there the
alternative is LSMR projection on the rectangular map, which is a hundred
times slower than a dense SVD that fits.
"""
function dense_is_cheap(L; dense_limit::Integer = DENSE_LIMIT,
                        dense_budget_bytes::Real = DENSE_BUDGET_BYTES,
                        square_dense_limit::Integer = SQUARE_DENSE_LIMIT)
    m, n = size(L)
    if m == n && !isempty(matrix_free_solvers(L))
        return n <= square_dense_limit
    end
    # Bytes decide; the dimension shortcut only applies when the bytes agree.
    # It used to be an `||`, and that densified a 160000x840 derivation map
    # (d = 20, valence 4) because 840 <= DENSE_LIMIT -- 1 GB for the matrix
    # and several more for the SVD workspace, past the machine's budget.  A
    # small side does not make a matrix small; the byte test already admits
    # every case the shortcut was meant for (1083x1083 is 9 MB).
    # `dense_limit` is kept in the signature for callers that pass it.
    return sizeof(eltype(L)) * m * n <= dense_budget_bytes
end

"""
    matrix_free_solvers([L]) :: Vector{Symbol}

Registered solvers that never call `Matrix`, in preference order for the map
`L` -- the order depends on the map's SHAPE.  Without `L` the rectangular
order is returned.

For a **rectangular** map (the densor map of `den`), `LSMRSolver` leads because
it is the only one that never squares the operator: it projects random vectors
off the row space with LSMR on the *rectangular* map.  Squaring is a precision
wall, not a slowdown -- a null space separated from the rest of the spectrum by
1e-8 in `σ` is separated by 1e-16 in `σ²`, which is the double-precision noise
floor.  `CGSolver` (LOBPCG on `AᵗA`) is next: it targets the right end and is
cheaper per step, but it inherits `κ(A)²`.

For a **square symmetric** map (the derivation operator of `sylvesterLM`) the
order is inverted.  Measured on the sphere stratification benchmark at
d = 24..56 (bench/reports/exp3-scaling.csv): `ArpackSolver` needs a flat
~560 map applications and finds every null vector on every seed;
`KrylovSolver` (block Lanczos) is as reliable at 4x the applications at its
default block; `CGSolver` is 10-20x slower than Arpack and its recovery error
is 1e-10 rather than 1e-13; `LSMRSolver` on this map is 100x slower than
Arpack, because its projection solves are designed for the rectangular map
and it asks for `nv + 8` of them.  `LanczosSolver` is deliberately absent:
`svdl` converges to the *largest* singular values, so it approaches the null
space from the wrong end -- see `ShiftInvertSolver` for the transform that
fixes that.
"""
matrix_free_solvers() =
    filter(s -> haskey(SOLVER_REGISTRY, s),
           [:LSMRSolver, :CGSolver, :ArpackSolver, :KrylovSolver])
matrix_free_solvers(L) =
    size(L, 1) == size(L, 2) ?
        filter(s -> haskey(SOLVER_REGISTRY, s),
               # LOBPCG (:CGSolver) is excluded below Float64: its Float32
               # Cholesky of the block Gram matrix fails and the block collapses
               # to 1-3 vectors, so it cannot hold a null space of dimension 3
               # whatever its tolerance (precision-study.md, Exp. 2b, 5).
               real(eltype(L)) === Float64 ?
                   [:ArpackSolver, :KrylovSolver, :CGSolver, :LSMRSolver] :
                   [:ArpackSolver, :KrylovSolver, :LSMRSolver]) :
        matrix_free_solvers()

"""
    wants_square(::NullSolver) -> Bool

Whether this solver needs a square symmetric operator rather than the
rectangular map.

Every eigen-based method does, since only a square operator has eigenvalues;
the SVD- and LU-based ones take the rectangular map directly, and prefer it,
because squaring squares the condition number.  Default `true`, which is right
for every eigensolver an extension might add.

This is a *trait on the solver*, so `solve_nullspace` can form `LᵗL` -- as a
composition of linear maps, never as a matrix -- on behalf of whichever solver
was asked for.  It replaces `Densors.jl`'s `__needsSquare(sym)`, which
hard-coded the same knowledge as a list of symbol names inside the caller: a
solver added by an extension was not in that list, so passing its name gave
either a wrong-shape assertion or a silently rectangular solve.
"""
wants_square(::NullSolver) = true

"""
    densifies(::NullSolver, L) -> Bool

Whether this solver will call `Matrix(L)`.  Used only to label and estimate the
progress report: a densifying solver does exactly `size(L,2)` map applications,
so its progress has an exact denominator, while an iterative one does an
unknown number.
"""
densifies(::NullSolver, L) = false

"""
    initial_request(::NullSolver[, L]) -> Int

How many eigenpairs `solve_nullspace` asks this solver for on its first try
(it doubles from there while everything returned is null).  Default 16.

The right number depends on the solver's cost model.  ARPACK's cost is flat in
the request -- 470..700 map applications for 4, 8, 16 or 32 on the sphere
benchmark (bench/reports/exp3-nv0.csv) -- and requesting 16 is what makes it
find every copy of the zero eigenvalue, so 16 is right for it.  Block Lanczos
(`KrylovSolver`) puts a block as wide as the request through the map on every
step, so its cost is nearly linear in the request: ~500 applications for 4,
~850 for 8, ~1800 for 16, ~2600 for 32.  It starts at 4 and lets the doubling
find larger null spaces; the doubling costs at most one extra solve's worth.
"""
initial_request(::NullSolver) = 16
initial_request(solver::NullSolver, L) = initial_request(solver)

struct SVDSolver <: NullSolver end
struct LUSolver <: NullSolver end

wants_square(::SVDSolver) = false
wants_square(::LUSolver) = false
densifies(::SVDSolver, L) = true
densifies(::LUSolver, L) = true

"""
    AutoSolver

Densify only when that is cheap; otherwise stay matrix-free.

`Matrix(L)` is the one operation in this file that scales with the *product* of
the map's dimensions rather than with the work of applying it, and the maps here
are not square.  The densor map has `prod(dims)` columns and
`length(Δ)*rows(P)*prod(dims)` rows, i.e. `O(n^7)` entries on a valence-3
family -- 14GB at n = 19 and 120GB at n = 26 -- while applying it is a handful
of tensor contractions.  Densifying it was never necessary: `denLM` and
`sylvesterLM` both return genuine `LinearMap`s with genuine adjoints precisely
so that Krylov, LOBPCG, Arpack and any other black-box method can be pointed
straight at them.
"""
struct AutoSolver <: NullSolver
    dense::NullSolver
    dense_limit::Int
    dense_budget_bytes::Float64
end
AutoSolver(; dense = SVDSolver(), dense_limit = DENSE_LIMIT,
           dense_budget_bytes = DENSE_BUDGET_BYTES) =
    AutoSolver(dense, dense_limit, dense_budget_bytes)

# Both of these take the rectangular map and reshape it themselves.
wants_square(::AutoSolver) = false
densifies(m::AutoSolver, L) =
    dense_is_cheap(L; dense_limit = m.dense_limit,
                   dense_budget_bytes = m.dense_budget_bytes) &&
    densifies(m.dense, L)
function initial_request(m::AutoSolver, L)
    densifies(m, L) && return initial_request(m.dense, L)
    free = matrix_free_solvers(L)
    return isempty(free) ? initial_request(m.dense, L) :
           initial_request(SOLVER_REGISTRY[first(free)], L)
end

"""
    SOLVER_REGISTRY :: Dict{Symbol, NullSolver}

Symbol -> solver instance.  Extensions register themselves here when their
trigger package loads.

This replaces a hard-coded `if/elseif` chain over seven symbol names.  That
chain could not work: the extension solver *types* are defined inside the
extension modules, so `KrylovSolver()` and friends were `UndefVarError:
not defined in Dleto`.  Five of the seven advertised solvers were unreachable
-- including Krylov, Lanczos and CG, whose packages are in [deps].
"""
const SOLVER_REGISTRY = Dict{Symbol, NullSolver}()

"""
    register_solver!(name::Symbol, solver::NullSolver)

Make `solver` reachable as `solve(L, name)`.  Called by extensions at load.
"""
register_solver!(name::Symbol, solver::NullSolver) = (SOLVER_REGISTRY[name] = solver)

"""Symbols currently reachable, i.e. whose packages are loaded."""
available_solvers() = sort(collect(keys(SOLVER_REGISTRY)))

function solve(L, sym::Symbol=:SVDSolver; kwargs...)
    haskey(SOLVER_REGISTRY, sym) || error(
        "Unknown or unavailable solver :$sym. Available now: " *
        string(available_solvers()) *
        ". Extension solvers need their package loaded first " *
        "(KrylovKit for :KrylovSolver, IterativeSolvers for :LanczosSolver " *
        "and :CGSolver, Arpack for :ArpackSolver and :ArpackDenseSolver).")
    return solve(SOLVER_REGISTRY[sym], L; kwargs...)
end

register_solver!(:SVDSolver, SVDSolver())
register_solver!(:LUSolver, LUSolver())
register_solver!(:AutoSolver, AutoSolver())

"""
    solve(::AutoSolver, L; nv, kwargs...)

Dense when `dense_is_cheap(L)`, matrix-free otherwise.

The matrix-free branch needs a *square symmetric* operator, since the methods
it delegates to are eigensolvers; when handed a rectangular `L` it forms
`LᵗL` **as a composition of linear maps**, not as a matrix.  That squares the
condition number, which is the price of never densifying, and is why the dense
branch is preferred whenever it fits.
"""
function solve(m::AutoSolver, L::LinearMap; nv::Integer = 10, kwargs...)
    if dense_is_cheap(L; dense_limit = m.dense_limit,
                      dense_budget_bytes = m.dense_budget_bytes)
        return solve(m.dense, L; nv = nv, kwargs...)
    end

    free = matrix_free_solvers(L)
    isempty(free) && error(
        "AutoSolver: Matrix(L) would need " *
        string(round(sizeof(eltype(L)) * size(L, 1) * size(L, 2) / 2^30; digits = 1)) *
        "GB for a $(size(L,1))x$(size(L,2)) map, and no matrix-free solver is " *
        "registered. Load IterativeSolvers (for :CGSolver), Arpack or KrylovKit, " *
        "or pass solver=:SVDSolver to densify anyway.")

    # Square it as a map, never as a matrix -- but only for a solver that wants
    # the square: `LSMRSolver` exists to take the rectangular map unsquared.
    chosen = SOLVER_REGISTRY[first(free)]
    S = (wants_square(chosen) && size(L, 1) != size(L, 2)) ? L' * L : L
    return solve(chosen, S; nv = nv, kwargs...)
end

# ---------------------------------------------------------------- the verdict

"""
    NullVerdict

The certificate `solve_nullspace` returns alongside the basis: which nullity
it chose, by which rule, and the spectrum around the cut so the caller can
see for itself.  Every value is RELATIVE to the operator norm `scale`.

- `nullity`     the number of vectors returned
- `rule`        `:gap` (the cut sits at a dominant multiplicative jump),
                `:threshold` (no jump cleared `gap_ratio`; the old count
                below `threshold` was used) or `:fixed` (the caller asked
                for `nd > 0` vectors; the threshold count, capped at `nd`)
- `certified`   `true` only for `:gap`, or for a nullity of 0 whose smallest
                value is `gap_ratio` above the threshold
- `gap`         the ratio at the cut, `spectrum[nullity+1] / max(spectrum[nullity], floor)`
                (for nullity 0: `spectrum[1] / threshold`); `NaN` if nothing
                was returned above the cut
- `gap_ratio`   the minimum ratio that was required
- `floor`       the precision floor `floor_eps * eps(T)`, relative
- `floor_binding`  whether the value just below the cut was *under* the
                floor, i.e. the gap was measured from the floor and not from
                the value itself.  Always true for a clean Float64 null space
                (its values sit at 1e-15); the informative case is Float32,
                where a genuine near-derivation can hide under the floor.
- `threshold`   the old-style relative threshold `max(tol, 100 eps) `; with an
                explicit `atol` this is `atol / scale`
- `near_null`   how many values ABOVE the cut are still below `threshold`:
                these are the near-derivations the fixed threshold would have
                counted as null, and are the signal on real data
- `below`, `above`  up to three relative values on either side of the cut
- `spectrum`    every relative value the solver returned, sorted
- `scale`       the operator-norm estimate the relative values divide by
- `requested`   how many values were finally requested from the solver
"""
struct NullVerdict
    nullity::Int
    rule::Symbol
    certified::Bool
    gap::Float64
    gap_ratio::Float64
    floor::Float64
    floor_binding::Bool
    threshold::Float64
    near_null::Int
    below::Vector{Float64}
    above::Vector{Float64}
    spectrum::Vector{Float64}
    scale::Float64
    requested::Int
end

function Base.show(io::IO, v::NullVerdict)
    fmt(x) = isfinite(x) ? string(round(x; sigdigits = 3)) : string(x)
    print(io, "NullVerdict(nullity = ", v.nullity, ", rule = :", v.rule,
          v.certified ? ", certified" : ", UNCERTIFIED",
          ", gap = ", fmt(v.gap), " (need ", fmt(v.gap_ratio), ")",
          v.floor_binding ? ", floor-bound" : "",
          v.near_null > 0 ? ", near-null above cut: $(v.near_null)" : "",
          "; below = ", map(fmt, v.below), ", above = ", map(fmt, v.above), ")")
end

"""
    GAP_RATIO

Default minimum multiplicative jump that certifies a nullity; see
`gap_verdict`.  Chosen from the sphere stratification data
(bench/reports/gap-verdict.md).  With the precision floor at `100 eps(T)`:

- Float64: the null cluster sits at 1e-16..1e-13 relative, so it is floored
  to 2.2e-14, and the first nonzero eigenvalue of the derivation operator is
  1.2e-8 (the near-degenerate d = 50 seed 50) to 5e-3 relative.  Gaps are
  5e5..2e11.  Consecutive ratios *among nonzero* eigenvalues are 1.4..20,
  except at that same near-degenerate seed, where `lambda_5/lambda_4 = 4.5e3`.
- Float32: the null cluster is 6e-8..6e-7 relative -- under the 1.2e-5 floor
  -- and the first nonzero eigenvalue is 2.5e-3..7e-3 at d <= 40, giving
  floored gaps of 200..600; at d >= 45, where that eigenvalue drifts to
  ~1e-4, the floored gap is 10..50 and Float32 has genuinely run out of
  margin (its recovery error is 1e-3 there).

100 sits above every within-cluster ratio (the floored null cluster spans at
most ~5x), below every Float64 gap by more than three decades, and splits
the Float32 cases exactly where the precision study found Float32 to be
trustworthy (d <= 40) or not (d >= 45).  1e3 would leave every Float32 run
uncertified, because the 100 eps floor caps the Float32 gap at
`lambda_4 / 1.2e-5 < 1e3` for any first eigenvalue below 1.2e-2.
"""
const GAP_RATIO = 100.0

"""
    FLOOR_EPS

The precision floor, in units of `eps(T)`, below which a returned value is
indistinguishable from zero.  The same constant floors the relative
threshold, so "null" has one meaning in this file.  Measured null values sit
at 1..14 eps (Float64 Arpack/SVD), up to ~450 eps (Float64 block Lanczos at
its loose tolerance) and 1..5 eps (Float32); 100 eps keeps Float32's
margin without swallowing a Float64 gap.
"""
const FLOOR_EPS = 100

"""
    gap_verdict(vals, scale; threshold, floor, gap_ratio, nd, requested) -> (perm, NullVerdict)

The `sigma_(e+1)` verdict of Algorithm 2 (null_patterns.pdf), as a GAP test
on the values a null solver returned.

`vals` are the solver's singular-type values (eigenvalues of the square map,
singular values, or residual norms -- whatever `solve` returns for the
operator it was handed) and `scale` the operator norm those are measured
against.  `threshold` and `floor` are RELATIVE.  Returns the permutation that
sorts `vals` by magnitude and the verdict on the sorted spectrum.

WHY A GAP.  The fixed relative threshold has no right value.  On the sphere
benchmark (true nullity 3) the fourth eigenvalue relative to the largest
wanders from 2.5e-3 to 8e-5 across seeds at d = 32, trends to 1e-4 at
d = 45..50, and at d = 50 seed 50 a genuine near-derivation sits at 1.2e-8:
under `tol = 1e-6` every solver reports nullity 4 there, under `1e-8` Float64
reports 3 and Float32 -- whose null values are at 6e-7 -- still 4.  The null
cluster itself is unmistakable, though: its values are at rounding, three to
ten decades below whatever comes next.  So the rule is: sort, floor at the
precision noise, find the largest consecutive ratio, and cut there when the
ratio clears `gap_ratio`.

THE CEILING.  Only cuts whose values are all below `threshold` are eligible,
so the old `tol` keeps its meaning as a ceiling and the gap test can only
*refine* the fixed count, never exceed it.  This is not optional: a gap
cannot decide whether the smallest value is "zero", because zero has no
lower neighbour but the floor, and the floor is wrong whenever the
operator's own noise exceeds the arithmetic's.  A Float32-built map
eigen-decomposed in Float64 has null values at 5e-9 relative -- 2e5 above
the Float64 floor -- and a floor-to-first-value "gap" would certify nullity
0 on a rank-3 null space.  Pass `tol = Inf` to remove the ceiling and take
the gap alone.

Nullity 0 is certified when the smallest value is `gap_ratio` above the
threshold, i.e. there is nothing anywhere near zero.

The cut is only trusted when at least one value came back above it (the
caller asks for more otherwise); values above the cut but under the
threshold are counted in `near_null` -- the near-derivations that the fixed
threshold would have swallowed.
"""
function gap_verdict(vals::AbstractVector, scale::Real;
                     threshold::Real, floor::Real, gap_ratio::Real = GAP_RATIO,
                     nd::Integer = -1, requested::Integer = length(vals))
    perm = sortperm(abs.(vals))
    rel = Float64[abs(v) / scale for v in vals[perm]]
    n = length(rel)
    threshold = Float64(threshold)
    floor = Float64(floor)
    below_thr = count(<(threshold), rel)

    # Candidate cuts k = 1..kmax: at least one value above, all below the
    # ceiling.  The ratio is against the floored lower value.
    kmax = min(below_thr, n - 1)
    best_k, best_gap = 0, 0.0
    for k in 1:kmax
        g = rel[k + 1] / max(rel[k], floor)
        if g > best_gap
            best_k, best_gap = k, g
        end
    end

    ratio_at(k) = k == 0 ? (n >= 1 ? rel[1] / threshold : NaN) :
                  k < n  ? rel[k + 1] / max(rel[k], floor) : NaN

    if nd > 0
        cut = min(below_thr, Int(nd))
        rule, certified = :fixed, false
    elseif best_k >= 1 && best_gap >= gap_ratio
        cut = best_k
        rule, certified = :gap, true
    else
        cut = below_thr
        rule = :threshold
        certified = cut == 0 && n >= 1 && rel[1] / threshold >= gap_ratio
    end
    gap = ratio_at(cut)
    take(r) = rel[r]
    return perm, NullVerdict(cut, rule, certified, gap, Float64(gap_ratio), floor,
                             cut >= 1 && rel[cut] < floor,
                             threshold, max(below_thr - cut, 0),
                             take(max(1, cut - 2):cut), take(cut + 1:min(n, cut + 3)),
                             rel, Float64(scale), Int(requested))
end

export NullVerdict, gap_verdict

"""
    solve_nullspace(L, solver; tol, atol, nd, nv0, gap_ratio, min_above)
        -> (; vals, vecs, verdict)

A null-space basis, asking an iterative solver only for as many vectors as the
null space turns out to need, and a `NullVerdict` saying how the nullity was
decided.  The result destructures as `(vals, vecs) = solve_nullspace(...)`
unchanged; the verdict is the third field.

`nd <= 0` means "the whole basis".  Every caller used to turn that into
"compute the entire spectrum" -- `nv = globalDim(Ω)` in `SylverLining`,
`nv = prod(dims)` in `den`.  For a dense solver that is merely wasteful, but it
is what made the iterative solvers useless: at n = 19 Arnoldi was asked for
1083 of 1083 eigenvalues, retried five times, and returned 9; LOBPCG was asked
for a block of 1083 and could not factorize it.  Asking for the whole spectrum
is not how you find a null space.

Instead: ask for a modest `nv`, decide the nullity from the returned values,
and double `nv` only while the returned spectrum does not reach past the null
space -- fewer than `min_above` values came back above the cut.  Cost is then
proportional to the true nullity, not to the dimension of the space.

HOW THE NULLITY IS DECIDED -- see `gap_verdict`.  The values are sorted,
floored at `FLOOR_EPS * eps(T) * ‖L‖`, and cut at the largest consecutive
ratio when that ratio clears `gap_ratio` (default `GAP_RATIO`); the result
is then `certified`.  `tol` is a CEILING: only values below
`max(tol, 100 eps(T)) * ‖L‖` can be counted null, so the gap refines the old
fixed count and never exceeds it.  When no jump clears `gap_ratio` the old
count is used, the verdict is uncertified, and a warning names the values
around the cut.  When the verdict IS certified but the cut landed at the
precision floor (`floor_binding`), an `@info` names the same values -- not a
warning, because that is the routine case in Float64, but a note that a
near-derivation hiding below the floor would look identical.  With `nd > 0`
the threshold count capped at `nd` is returned as before (`rule = :fixed`).

`squared = true` says the map handed in is ALREADY a Gram operator `AᵗA`, so
its values are `σ²` and the ceiling is squared to match; solvers that square a
rectangular map here are detected without the flag.

`nv0` is the first request; it defaults to `initial_request(solver, L)`, a
per-solver trait, because the right first request depends on the solver's
cost model (flat in the request for ARPACK, linear for block Lanczos).
`min_above = 2` asks that at least two values lie above the cut before it is
trusted, so a spectrum `0, 0, 0, 1e-10 | 1e-2` cannot be cut at 3 when only
four values were seen and at 4 when five were.  For ARPACK (16 requested)
this costs nothing; block Lanczos, which starts at 4, doubles once on a
nullity-3 problem.

The doubling rule is only sound if the solver returns *every* copy of the zero
eigenvalue it could have found -- a null space is a multiple eigenvalue, and
a single-vector Krylov method finds one copy per start vector and the rest by
luck.  Both Krylov extensions now guarantee this (ARPACK by requesting at
least 16 pairs, `KrylovSolver` by a block as wide as the request); see their
docstrings and bench/reports/krylov-calibration.md for the measurements.
"""
function solve_nullspace(L, solver::Union{Symbol,NullSolver};
                         tol::Real = 1e-6, atol::Union{Nothing,Real} = nothing,
                         nd = -1, nv0::Union{Nothing,Integer} = nothing,
                         gap_ratio::Real = GAP_RATIO, min_above::Integer = 2,
                         squared::Bool = false,
                         progress = false, label::AbstractString = "null solve",
                         kwargs...)
    N = size(L, 2)
    want_all = nd <= 0
    first_request = nv0 === nothing ? initial_request(solver, L) : Int(nv0)
    k = want_all ? min(N, max(1, first_request)) : min(N, max(1, floor(Int, nd)))

    # Square the map for the solvers that need it -- as a composition, so no
    # matrix is formed.  `AutoSolver` and `ShiftInvertSolver` decline the trait
    # because they handle the shape themselves.
    want_squared = squared || (wants_square(solver) && size(L, 1) != size(L, 2))
    L = (wants_square(solver) && size(L, 1) != size(L, 2)) ? L' * L : L

    # `tol` is RELATIVE to the operator norm, as in `LinearAlgebra.nullspace`.
    #
    # It used to be absolute, and that quietly broke every iterative solve on
    # an unnormalised operator.  The densor operator on the scrambled
    # circulant family has `‖AᵗA‖ ≈ 1e25`, so an absolute `tol = 1e-6` asks
    # for eigenvalues 31 orders of magnitude below the operator's own scale --
    # far below what any iterative method converges to, since their stopping
    # criteria are relative.  The dense SVD path got away with it because on
    # well-scaled test tensors the norm is O(1); the first genuinely
    # matrix-free case, `den` at n = 15, returned dimension 0 after 42s of
    # perfectly good LOBPCG work whose eigenvalues were simply above 1e-6.
    #
    # `‖Lop‖` bounds what any of these solvers report -- singular values of
    # `Lop`, or its eigenvalues when it is symmetric -- so one scale serves
    # every solver. Pass `atol` to override with an absolute threshold.
    #
    # SQUARING SILENTLY SQUARE-ROOTS THE TOLERANCE, and that is the other half
    # of getting a relative tolerance right.  When the operator being filtered
    # is a Gram operator `AᵗA` -- which this function forms itself for the
    # eigensolvers, and which a caller may hand in ready-made (`squared=true`)
    # -- the reported values are `λ = σ²`.  A ceiling of `tol·‖AᵗA‖` then
    # admits every direction with `σ/σ_max ≤ √tol`: ask for 1e-6 and you get
    # 1e-3.
    #
    # That is not hypothetical.  On real video boxes from Boards.mp4 (8x8x8,
    # true derivation space = the 2 scalars) it made `SVDSolver` report
    # 113--156 derivations per box at a residual of 2e-3.  Squaring the
    # tolerance restores the invariant the caller asked for: accepted
    # directions satisfy `σ/σ_max ≤ tol`.  The `FLOOR_EPS` floor below still
    # applies, so this can only tighten the ceiling as far as the element type
    # can actually resolve -- in Float32 the floor binds and `tol²` never
    # takes effect.
    #
    # The null eigenvalues an iterative solver returns sit at a few `eps(T)`
    # relative to `‖L‖` -- up to 6.2e-7 in Float32 at d = 40 on the sphere
    # benchmark, within 1.6x of the default 1e-6 -- while the first nonzero
    # eigenvalue is >= 1e-4 relative there.  Floor the relative threshold at
    # 100*eps(T) so a Float32 run does not lose a null vector to rounding;
    # in Float64 the floor (2.2e-14) is far below any sensible `tol`.
    RT = real(eltype(L))
    scale = max(sqrt(max(opnorm_estimate(L' * L; iters = 10), 0.0)), eps(RT))
    rel = want_squared ? tol^2 : tol
    threshold = atol === nothing ? max(rel, FLOOR_EPS * eps(RT)) * scale : atol
    # The gap test works in relative terms; the floor is the same constant
    # that floors the threshold, so a value under it is "zero" in both rules.
    rel_threshold = threshold / scale
    rel_floor = FLOOR_EPS * eps(RT)

    # Progress: one wrapper serves both stages, because `Matrix(L)` applies the
    # map once per column.  A densifying solver therefore has an exact
    # denominator; an iterative one has none, and gets count and rate instead
    # of a fabricated ETA.
    spec = progress_spec(progress)
    dense = densifies(solver, L)
    tr = ProgressTracker(dense ? "$label (densify $(size(L,1))x$(size(L,2)))" :
                                 "$label (iterate)",
                         dense ? :densify : :solve,
                         dense ? size(L, 2) : 0,
                         spec)
    Lp = progress_wrap(L, tr)

    try
        while true
            # NOTE the argument order: the instance form is `solve(method, L)`
            # while the symbol form is `solve(L, sym)`.  Reversed relative to
            # each other, which is a wart in the existing interface.
            result = solve(solver, Lp; nv = k, kwargs...)
            vals = result.vals
            vecs = result.vecs
            perm, verdict = gap_verdict(vals, scale; threshold = rel_threshold,
                                        floor = rel_floor, gap_ratio = gap_ratio,
                                        nd = want_all ? -1 : floor(Int, nd),
                                        requested = k)

            # Bracketed: enough came back above the cut to trust it, so we
            # have seen the whole null space.  Or we asked for everything
            # there is, or for a fixed number.
            #
            # Note what this rule does NOT distinguish: "found the whole null
            # space and then some" and "converged to nothing at all" both show
            # up as values above the cut.  That is deliberate -- an empty null
            # space is a legitimate answer (a Tucker chisel on a generic
            # tensor) and escalating on it would be pure waste -- but it means
            # a non-converging iterative solver reports "no solutions" rather
            # than "I failed".  The defence is to make the solver converge: a
            # relative `threshold` above, and block headroom inside the block
            # methods.  The verdict at least says so: such a result is
            # uncertified unless the smallest value is far above threshold.
            above = length(vals) - verdict.nullity
            if !want_all || above >= min_above || k >= N
                if want_all && !verdict.certified
                    @warn "$label: nullity $(verdict.nullity) is UNCERTIFIED -- no gap " *
                          "of $(gap_ratio)x in the spectrum below the threshold " *
                          "$(verdict.threshold) (relative). Values around the cut: " *
                          "$(verdict.below) | $(verdict.above)" *
                          (verdict.floor_binding ? "; the value below the cut is under " *
                           "the $(eltype(L)) precision floor $(verdict.floor), so a " *
                           "near-derivation there cannot be told from zero" : "") *
                          ". Consider Float64, an explicit `tol`, or `nd`." maxlog = 1
                elseif want_all && verdict.floor_binding
                    # Certified, but the cut landed AT the precision floor: the
                    # last value counted null is indistinguishable from a
                    # near-derivation that happens to sit below `eltype(L)`'s
                    # noise.  Harmless in the ordinary Float64 case, where the
                    # true null cluster always sits here -- but the only way to
                    # tell the two apart is more precision, so this stays an
                    # `@info`, not a `@warn`: it names the cut without alarming
                    # every routine Float64 call.
                    @info "$label: nullity $(verdict.nullity) certified (rule :" *
                          "$(verdict.rule)), cut at the $(eltype(L)) precision floor " *
                          "$(verdict.floor) (relative) -- a near-derivation hidden " *
                          "below that floor would look identical. Values around the " *
                          "cut: $(verdict.below) | $(verdict.above)" maxlog = 1
                end
                keep = perm[1:verdict.nullity]
                return (; vals = vals[keep], vecs = vecs[:, keep], verdict)
            end
            k = min(N, 2 * k)
        end
    finally
        finish!(tr)
    end
end

function solve_nullspace(L, solver::Symbol = :AutoSolver; kwargs...)
    haskey(SOLVER_REGISTRY, solver) || error(
        "solve_nullspace: no solver :$solver is registered. Available: " *
        join(string.(available_solvers()), ", ") * ". Extension solvers " *
        "(:ArpackSolver, :KrylovSolver, :LanczosSolver, :CGSolver, :LSMRSolver) " *
        "register when their package is loaded (`using Arpack`, ...).")
    return solve_nullspace(L, SOLVER_REGISTRY[solver]; kwargs...)
end

# ---------------------------------------------------- spectral transform
#
# Every black-box iterative eigensolver converges to the *extremes* of the
# spectrum, and most of them to the largest: `svdl` bidiagonalizes toward the
# largest singular values, Arnoldi `:LR` toward the largest eigenvalues.  A
# null space sits at the other end, which is why those solvers looked useless
# here -- `LanczosSolver` returned dimension 0 not because it failed but
# because it was pointed at the wrong end of the spectrum.
#
# The fix is the standard spectral transform: replace `M = AᵗA` by
#
#     shift_invert(M) = (M + eps*I)^-1
#
# whose eigenvalues are `1/(lambda + eps)`.  `eps` is taken RELATIVE to `‖M‖`,
# because it sets the condition number CG has to cope with -- see
# `shift_invert_map`.  The small eigenvalues of `M`
# become the large eigenvalues of the transform, so a largest-end method now
# converges straight to the null space, and it converges *fast*, because the
# transform stretches the gap between 0 and the smallest nonzero eigenvalue
# into the gap between `1/eps` and `1/(lambda_min + eps)`.
#
# `eps` is not optional.  A genuine null vector has `lambda` exactly 0 up to
# rounding, so the untransformed `1/lambda` overflows precisely on the vectors
# we are looking for -- the better the solver, the worse the blow-up.  The
# shift caps the transform at `1/eps` and keeps it finite.
#
# The inverse is never formed.  `M + eps*I` is symmetric positive definite --
# `M` is a Gram operator and `eps > 0` -- so it is applied by conjugate
# gradients, which needs only matrix-vector products.  So the whole chain from
# tensor contractions to null space stays matrix-free.

"""
    cg_solve(M, b; tol, maxiter) -> x

Conjugate gradients on a symmetric positive definite `LinearMap`.  Written out
here rather than taken from IterativeSolvers so that the shift-invert transform
works with no optional dependency loaded.
"""
function cg_solve(M, b::AbstractVector; tol::Real = 1e-10, maxiter::Integer = 500)
    x = zeros(eltype(b), length(b))
    r = copy(b)
    p = copy(r)
    rs = dot(r, r)
    rs0 = rs
    rs0 == 0 && return x
    for _ in 1:maxiter
        Mp = M * p
        α = rs / dot(p, Mp)
        @. x += α * p
        @. r -= α * Mp
        rs_new = dot(r, r)
        sqrt(rs_new / rs0) < tol && break
        @. p = r + (rs_new / rs) * p
        rs = rs_new
    end
    return x
end

"""
    opnorm_estimate(M; iters=20) -> Real

Rough largest eigenvalue of a symmetric `LinearMap` by power iteration.  A
handful of matvecs; only the order of magnitude is needed.
"""
function opnorm_estimate(M; iters::Integer = 20)
    n = size(M, 2)
    T = eltype(M)
    v = randn(T, n)
    v ./= norm(v)
    λ = zero(real(T))
    for _ in 1:iters
        w = M * v
        nw = norm(w)
        nw == 0 && return 0.0
        λ = dot(v, w)
        v = w ./ nw
    end
    return abs(λ)
end

"""
    shift_invert_map(L; shift_rel, shift, cgtol, cgmaxiter) -> (M, S)

The pair `(M, S)` where `M = LᵗL` and `S = (M + shift*I)^-1`, both as
`LinearMap`s and neither ever formed as a matrix.

Hand `S` to any largest-eigenvalue black box -- Arpack `eigs(:LM)`, KrylovKit
`eigsolve(:LR)`, `svdl` -- and its top eigenpairs are the null space of `L`.
Recover the original eigenvalue from an eigenvalue `mu` of `S` by
`lambda = 1/mu - shift`.

**The shift must be relative to `‖M‖`, not absolute.** `shift` sets the
condition number of the operator CG has to invert: `kappa(M + shift*I)` is
about `‖M‖/shift`. On these problems `‖M‖` is enormous -- the densor operator
on the circulant family has `‖AᵗA‖ ≈ 1e25` -- so an absolute `shift = 1e-8`
asks CG to invert something with condition `1e33`, and it simply never
converges. Scaling the shift by the estimated norm (`shift_rel`, default
1e-10) fixes the inner condition number at `1/shift_rel` regardless of how the
tensor happens to be scaled, while still separating an exact null vector
(`1/shift`) from the smallest nonzero eigenvalue. Pass `shift` explicitly to
override.
"""
function shift_invert_map(L; shift_rel::Real = 1e-10, shift::Union{Nothing,Real} = nothing,
                          cgtol::Real = 1e-4, cgmaxiter::Integer = 100)
    n = size(L, 2)
    M = size(L, 1) == size(L, 2) && LinearMaps.issymmetric(L) ? L : L' * L
    T = eltype(M)
    RT = typeof(real(zero(T)))
    σ = convert(RT, shift === nothing ? max(shift_rel * opnorm_estimate(M), eps(RT)) : shift)
    Mshift = LinearMaps.LinearMap{T}(v -> M * v + σ * v, n, n;
                                  issymmetric = true, isposdef = true)
    apply(v) = cg_solve(Mshift, collect(v); tol = cgtol, maxiter = cgmaxiter)
    S = LinearMaps.LinearMap{T}(apply, apply, n, n;
                             issymmetric = true, isposdef = true)
    return (M, S)
end

"""
    ShiftInvertSolver(outer::Symbol; shift, cgtol, cgmaxiter)

Null vectors via the spectral transform above, using `outer` -- any registered
largest-end solver -- to do the eigen work.

This is what turns the largest-end black boxes into null-space solvers.  It
returns eigenvalues of `LᵗL`, matched to what the other eigen-based solvers
report, so the caller's `tol` filter is unchanged.
"""
struct ShiftInvertSolver <: NullSolver
    outer::Symbol
    shift_rel::Float64
    cgtol::Float64
    cgmaxiter::Int
end
ShiftInvertSolver(outer::Symbol = :KrylovSolver; shift_rel = 1e-10, cgtol = 1e-4,
                  cgmaxiter = 100) =
    ShiftInvertSolver(outer, shift_rel, cgtol, cgmaxiter)

wants_square(::ShiftInvertSolver) = false

function solve(m::ShiftInvertSolver, L::LinearMap; nv::Integer = 10, kwargs...)
    haskey(SOLVER_REGISTRY, m.outer) || error(
        "ShiftInvertSolver needs :$(m.outer) registered; available: " *
        string(available_solvers()))
    (M, S) = shift_invert_map(L; shift_rel = m.shift_rel, cgtol = m.cgtol,
                              cgmaxiter = m.cgmaxiter)

    # The outer solver's own `nv` semantics differ (smallest vs largest), so
    # rather than trust its ordering we take what it returns and score every
    # vector directly against M.  That is honest regardless of which end the
    # outer method aimed at, and costs one extra matvec per vector.
    result = solve(SOLVER_REGISTRY[m.outer], S; nv = nv, kwargs...)
    V = result.vecs
    T = eltype(L)
    size(V, 2) == 0 && return (; vals = typeof(real(zero(T)))[], vecs = zeros(T, size(L, 2), 0))

    λ = [ let v = V[:, j], nrm = norm(v)
              nrm == 0 ? Inf : dot(v, M * v) / (nrm^2)
          end for j in 1:size(V, 2) ]
    ord = sortperm(λ)
    keep = ord[1:min(nv, length(ord))]
    return (; vals = λ[keep], vecs = V[:, keep])
end

export ShiftInvertSolver, shift_invert_map, cg_solve, dense_is_cheap

# Two shift-invert variants, so the wrong-ended black boxes become usable:
# Arnoldi and, once IterativeSolvers is loaded, `svdl`.
register_solver!(:ShiftInvertSolver, ShiftInvertSolver(:KrylovSolver))

    
function solve(::SVDSolver, L::LinearMap; nv::Integer = 10)
    println("Using SVDSolver...")
    # Use LinearAlgebra to compute the null space of L.
    println("Converting LinearMap to Matrix for SVD...")
    M = Matrix(L)
    m, n = size(M)
    # A WIDE map (m < n) has at least n - m null directions, and the thin SVD
    # cannot express them: `V` then has only m columns.  Ask for the full `V`
    # and extend `S` with the n - m exact zeros those columns belong to.  Found
    # by QuickDer on a valence-2 tensor (7x5, 35 equations, 74 unknowns):
    # SylverLining reported nullity 0 where the true nullity is 39.
    svds = LinearAlgebra.svd(M; full = m < n)
    S = m < n ? vcat(svds.S, zeros(eltype(svds.S), n - m)) : svds.S
    nvals = min(nv, length(S))
    return (;vals=S[end:-1:(end-nvals+1)], vecs=svds.V[:, end:-1:(end-nvals+1)])
end

"""
    solve(::LUSolver, L; nv, tol)

Null vectors by a rank-revealing triangular factorization -- column-pivoted
QR -- plus back-substitution.  The name is historical: this was LU, and why
it is not any more is the point of this docstring.

`lu` pivots ROWS.  Within the current column it divides by the largest entry
it can find; it never reorders columns.  So when column `j` is a linear
combination of columns `1:j-1`, every entry of that column has already been
eliminated to roundoff by the time it is reached, the pivot is `~eps`
whichever row is chosen, and elimination carries on dividing by it.  A tiny
pivot therefore lands wherever the FIRST dependent column happens to be, not
at the trailing end.  On the derivation-densor operator that is structural,
not bad luck: the operator space is three axis blocks, the two scalar
derivations make the first column of the third block depend on the first two,
and the tiny pivot sits at exactly `2n/3` at every size (110 of 165 at
d = 10, 650 of 975 at d = 25, 930 of 1395 at d = 30) with the other two
trailing.

The old code counted `abs.(diag(U)) .> tol` to get `r = n - 3`, then treated
the LAST `n - r` columns as free and solved `U[1:r,1:r] x = -U[1:r,j]` for
the pivot variables.  But `U[1:r,1:r]` contains the `1e-15` pivot at `2n/3`,
so the back-substitutions divided by it and returned rubbish.  Worse, the
rubbish is not reliably detectable: the solve amplifies the `e_{2n/3}`
component by `1/pivot ≈ 1e15`, and THAT direction is itself a null vector
(the dependent column minus its combination), so what comes back is
`1e15 · (one true null vector) + (garbage of size 1)`.  Its residual
`‖Mv‖/‖v‖` is `~1e-14` -- it passes any threshold -- while being the same
null vector three times over.  Hence the sphere sweep: nullity 1 at most
sizes (the garbage dominated; residuals 0.03 and 2.4 at d = 10), and at
d = 25 a "nullity 3" whose three vectors spanned a two-dimensional space --
two of them the same amplified vector to thirteen digits (`‖v‖ = 2.8e14` and
`6.4e13`, singular values of the normalized basis `1.43, 0.98, 6e-13`) --
so the random derivation `stratify` drew was missing a direction, and the
reconstruction error was 4.6e-2 after 4.3 s and 13 GB of eigensolver
flailing.  A per-vector residual is no defence against a basis that repeats
itself.

Column-pivoted QR (`qr(M, ColumnNorm())`, LAPACK `geqp3`) fixes the first
problem at the root.  At each step it moves the column of largest remaining
norm to the front, so `|R[1,1]| ≥ |R[2,2]| ≥ ...` and every dependent column
is pushed to the trailing block, where `|R[j,j]| ≈ eps·|R[1,1]|`.  The rank
is then a count from the end, `R[1:r,1:r]` is well conditioned by
construction, and with `M P = Q R` the null vectors are

    P * [ -R[1:r,1:r] \\ R[1:r,j] ; e_j ]        for j = n, n-1, ..., r+1,

each satisfying `M v = Q R Pᵗ v = Q [0; R[j,j]; 0] ≈ 0`, exactly as the
factorization says.  CPQR is not rank revealing in the worst case (Kahan's
matrix), but it is on everything Dleto forms, and its failure mode is loud:
a mis-ranked column shows up as a large residual, not as a duplicated basis
vector.  Cost is `4/3 n³` -- 0.14 s at n = 1395 against 0.64 s for `svd` --
and it asks nothing of `M`: not square, not symmetric, not positive.  A
pivoted Cholesky of the (symmetric positive semidefinite) derivation-densor
operator would be another 7x cheaper, but LAPACK's `pstrf` reported full rank
on these very matrices under its default tolerance, it would need `LᵗL` for
the rectangular densor maps this solver also serves (the squaring the LSMR
path was rewritten to avoid), and the saving is tenths of a second against a
densification that costs seconds.

The second problem -- a basis whose vectors all point the same way -- is
closed by orthonormalizing what is returned (a thin `qr` of the `n × k`
candidate set) BEFORE computing `vals = ‖M q‖`.  Duplicates would cancel to
noise and fail the residual test; independent null vectors stay null.  The
same step makes the padding honest.  `solve_nullspace` doubles `nv` while
every returned value is below its threshold, so when `nv` exceeds the
nullity this must return something visibly non-null: it continues the same
formula into the independent columns, `j = r, r-1, ...`, with
`R[1:j-1,1:j-1]` in place of `R[1:r,1:r]` -- the vector expressing how far
column `j` is from the span of the columns before it, `‖M v‖ = |R[j,j]|`
before orthogonalization and at least the smallest non-zero singular value
after it, since it is then orthogonal to the null space.  `k = min(nv, n)`
vectors come back, null ones first, most-null first.

`tol` is the rank cut, RELATIVE to `|R[1,1]|` (the largest column norm):
column `j` is dependent when `|R[j,j]| ≤ tol·|R[1,1]|`.  It used to be an
absolute `1e-8`, which an operator of norm 1e25 or 1e-25 defeats.  The
default `max(m, n)·eps(T)` is `LinearAlgebra.rank`'s convention and holds in
Float32 too, where on the sphere operator at n = 1395 the null diagonals sit
at roundoff (`1.3e-7, 6.7e-8, 4.8e-8` relative, against a cut of `1.7e-4`) and
the first independent one at `5.4e-2`, the same 3-against-1392 count as
Float64.  This is separate from `solve_nullspace`'s
`tol`, which filters the honest residuals afterwards; pass this one as a
keyword to `solve` directly.

Previously (two bugs ago) this returned a bare `Vector`, violating the
`(;vals, vecs)` contract above -- callers written against it died with
`FieldError: type Array has no field vals` -- and chose its free variables as
the last `nv` columns regardless of the computed rank.
"""
function solve(::LUSolver, L::LinearMap; nv::Integer = 10, tol = nothing)
    M = Matrix(L)
    T = eltype(M)
    m, n = size(M)
    k = min(nv, n)
    k <= 0 && return (; vals = real(T)[], vecs = zeros(T, n, 0))

    F = qr(M, ColumnNorm())
    R = F.R
    p = F.p
    dR = abs.(diag(R))
    rtol = tol === nothing ? max(m, n) * eps(real(T)) : real(T)(tol)
    r = isempty(dR) ? 0 : count(>(rtol * dR[1]), dR)

    # Candidates j = n, n-1, ...: the first n - r are null vectors, the rest
    # padding.  In pivoted coordinates each is a unit at j, zeros after it,
    # and the back-substituted combination of the (independent) columns before
    # it -- never more than r of them, so no division by a dropped pivot.
    V = zeros(T, n, k)
    for (c, j) in enumerate(n:-1:(n - k + 1))
        mm = min(r, j - 1)
        w = zeros(T, n)
        if mm > 0
            w[1:mm] = -(UpperTriangular(view(R, 1:mm, 1:mm)) \ R[1:mm, j])
        end
        w[j] = one(T)
        V[p, c] = w
    end

    # Orthonormalize in the given order, so the leading n - r columns still
    # span the null space; then the residuals are honest per direction.
    Q = Matrix(qr(V).Q)[:, 1:k]
    vals = [norm(M * view(Q, :, c)) for c in 1:k]
    return (; vals, vecs = Q)
end
