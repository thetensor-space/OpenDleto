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
    dense_is_cheap(L; dense_limit, dense_budget_bytes) -> Bool

Whether `Matrix(L)` is worth forming.

The test is on *bytes*, with the dimension as a shortcut, because dimension
alone is the wrong criterion for a rectangular map.  The derivation--densor
operator at n = 19 is 1083x1083: just over `DENSE_LIMIT`, but 9MB, so
densifying it is free and buys the accuracy of a real SVD.  The densor map at
the same n is 260642x6859: 14GB.  A pure dimension gate would either refuse the
first or accept the second.
"""
dense_is_cheap(L; dense_limit::Integer = DENSE_LIMIT,
               dense_budget_bytes::Real = DENSE_BUDGET_BYTES) =
    minimum(size(L)) <= dense_limit ||
    8.0 * size(L, 1) * size(L, 2) <= dense_budget_bytes

"""
    matrix_free_solvers() :: Vector{Symbol}

Registered solvers that never call `Matrix`, in preference order.

`LSMRSolver` leads because it is the only one that never squares the operator:
it runs shift-invert subspace iteration on the *rectangular* map, with LSMR as
the inner solve.  Squaring is a precision wall, not a slowdown -- a null space
separated from the rest of the spectrum by 1e-8 in `σ` is separated by 1e-16 in
`σ²`, which is the double-precision noise floor.  `CGSolver` (LOBPCG on `AᵗA`)
is next: it targets the right end and is cheaper per step, but it inherits
`κ(A)²`.  `KrylovSolver` (Arnoldi, `:SR`) stops on invariant subspaces and
typically returns about half the basis.  `LanczosSolver` is deliberately
absent: `svdl` converges to the *largest* singular values, so it approaches the
null space from the wrong end -- see `ShiftInvertSolver` for the transform that
fixes that.
"""
matrix_free_solvers() =
    filter(s -> haskey(SOLVER_REGISTRY, s),
           [:LSMRSolver, :CGSolver, :ArpackSolver, :KrylovSolver])

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

    free = matrix_free_solvers()
    isempty(free) && error(
        "AutoSolver: Matrix(L) would need " *
        string(round(8.0 * size(L, 1) * size(L, 2) / 2^30; digits = 1)) *
        "GB for a $(size(L,1))x$(size(L,2)) map, and no matrix-free solver is " *
        "registered. Load IterativeSolvers (for :CGSolver), Arpack or KrylovKit, " *
        "or pass solver=:SVDSolver to densify anyway.")

    # Square it as a map, never as a matrix.
    S = size(L, 1) == size(L, 2) ? L : L' * L
    return solve(SOLVER_REGISTRY[first(free)], S; nv = nv, kwargs...)
end

"""
    solve_nullspace(L, solver; tol, nd, nv0) -> (vals, vecs)

A null-space basis, asking an iterative solver only for as many vectors as the
null space turns out to need.

`nd <= 0` means "the whole basis".  Every caller used to turn that into
"compute the entire spectrum" -- `nv = globalDim(Ω)` in `SylverLining`,
`nv = prod(dims)` in `den`.  For a dense solver that is merely wasteful, but it
is what made the iterative solvers useless: at n = 19 Arnoldi was asked for
1083 of 1083 eigenvalues, retried five times, and returned 9; LOBPCG was asked
for a block of 1083 and could not factorize it.  Asking for the whole spectrum
is not how you find a null space.

Instead: ask for a modest `nv`, count how many returned values fall below
`tol`, and double `nv` only while *every* returned value is below it -- the
signal that the null space has not yet been bracketed.  Cost is then
proportional to the true nullity, not to the dimension of the space.
"""
function solve_nullspace(L, solver::Union{Symbol,NullSolver};
                         tol::Real = 1e-6, atol::Union{Nothing,Real} = nothing,
                         nd = -1, nv0::Integer = 16,
                         progress = false, label::AbstractString = "null solve",
                         kwargs...)
    N = size(L, 2)
    want_all = nd <= 0
    k = want_all ? min(N, max(1, nv0)) : min(N, max(1, floor(Int, nd)))

    # Square the map for the solvers that need it -- as a composition, so no
    # matrix is formed.  `AutoSolver` and `ShiftInvertSolver` decline the trait
    # because they handle the shape themselves.
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
    scale = atol === nothing ? sqrt(max(opnorm_estimate(L' * L; iters = 10), 0.0)) : 1.0
    threshold = atol === nothing ? tol * max(scale, eps()) : atol

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
            keep = findall(v -> abs(v) < threshold, vals)

            # Bracketed: something came back above tolerance, so we have seen
            # the whole null space.  Or we asked for everything there is.
            #
            # Note what this rule does NOT distinguish: "found the whole null
            # space and then some" and "converged to nothing at all" both show
            # up as `length(keep) < length(vals)`.  That is deliberate -- an
            # empty null space is a legitimate answer (a Tucker chisel on a
            # generic tensor) and escalating on it would be pure waste -- but
            # it means a non-converging iterative solver reports "no
            # solutions" rather than "I failed".  The defence is to make the
            # solver converge: a relative `threshold` above, and block
            # headroom inside the block methods.
            if !want_all || length(keep) < length(vals) || k >= N
                return (vals[keep], vecs[:, keep])
            end
            k = min(N, 2 * k)
        end
    finally
        finish!(tr)
    end
end

solve_nullspace(L, solver::Symbol = :AutoSolver; kwargs...) =
    solve_nullspace(L, SOLVER_REGISTRY[solver]; kwargs...)

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
    v = randn(n)
    v ./= norm(v)
    λ = 0.0
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
    σ = shift === nothing ? max(shift_rel * opnorm_estimate(M), eps()) : shift
    Mshift = LinearMaps.LinearMap(v -> M * v + σ * v, n, n;
                                  issymmetric = true, isposdef = true)
    apply(v) = cg_solve(Mshift, collect(v); tol = cgtol, maxiter = cgmaxiter)
    S = LinearMaps.LinearMap(apply, apply, n, n;
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
    size(V, 2) == 0 && return (; vals = Float64[], vecs = zeros(Float64, size(L, 2), 0))

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
    svds = LinearAlgebra.svd(M)
    nvals = min(nv, length(svds.S))
    return (;vals=svds.S[end:-1:(end-nvals+1)], vecs=svds.V[:, end:-1:(end-nvals+1)])
end

"""
    solve(::LUSolver, L; nv, tol)

    Null vectors by LU plus back-substitution.

    CAVEAT: `lu` pivots rows only, so it is **not rank revealing** -- this is
    only valid when the leading `rank` columns are independent.  It therefore
    reports an honest residual `‖Lv‖/‖v‖` for each vector it returns, so a
    caller filtering on `vals` discards a bad basis rather than trusting it.
    Prefer `SVDSolver` unless you know the column order is benign.

    Previously this returned a bare `Vector`, violating the `(;vals, vecs)`
    contract above -- callers written against it died with
    `FieldError: type Array has no field vals` -- and chose its free variables
    as the last `nv` columns regardless of the computed rank.
"""
function solve(::LUSolver, L::LinearMap; nv::Integer = 10, tol = 1e-8)
    M = Matrix(L)
    n = size(M, 2)
    F = lu(M; check = false)
    U = F.U
    r = min(sum(abs.(diag(U)) .> tol), size(U, 1), n)

    cols = Vector{Vector{eltype(M)}}()
    for j in (r + 1):n
        v = zeros(eltype(M), n)
        v[j] = 1
        if r > 0
            # U[1:r,1:r] x = -U[1:r,j] makes the pivot variables consistent.
            v[1:r] = U[1:r, 1:r] \ (-U[1:r, j])
        end
        push!(cols, v)
    end

    if isempty(cols)
        return (; vals = eltype(M)[], vecs = zeros(eltype(M), n, 0))
    end

    V = hcat(cols...)
    vals = [ norm(M * V[:, k]) / max(norm(V[:, k]), eps()) for k in 1:size(V, 2) ]
    keep = 1:min(nv, size(V, 2))
    return (; vals = vals[keep], vecs = V[:, keep])
end

