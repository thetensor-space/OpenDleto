#
# Strata Dleto: Precision.jl
#   One place where the floating-point type decides what "zero" means.
#
# Every rank, nullity and feasibility decision in this package is a decision
# that some number is zero, and none of them can be made without knowing the
# precision of the arithmetic that produced the number.  Before this file the
# constants were spread over five files and mostly absolute: `tol = 1e-6` in
# `derTrOpsReduced`, `1e-8` in `_qd_nullspace`, `1e-10` in three extension
# solvers, `1e-12` in LSMR.  Float64 got away with it.  Float32 did not
# (KrylovKit's `1e-12 * ‖L‖` sits 1e5 below the Float32 residual floor, so
# block Lanczos ran to `maxiter` for nothing -- 40x the time and allocation for
# the same answer), and Float16 was silently, confidently wrong: on the sphere
# benchmark at d = 10 with a true nullity of 3, every SylverLining route
# reported 32-41 derivations at a reconstruction error of 0.9.
#
# THREE TYPES, NOT ONE.  The type a tensor is STORED in, the type the
# arithmetic RUNS in, and the resolution a VERDICT may claim are three
# different things, and conflating them is what made Float16 dishonest:
#
#   * `compute_eltype(T)` -- what the arithmetic runs in.  Float16 promotes to
#     Float32, because there is no half-precision BLAS or LAPACK on any CPU
#     this package targets: ARPACK throws `MethodError` on `saupd`, KrylovKit
#     on `hschur!`, LOBPCG on `eigen!`, and the routes that do run (dense SVD,
#     Gram) have a noise floor of `eps16 * ‖L‖ ~ 0.09` relative, which is
#     ABOVE the derivation operator's first nonzero eigenvalue.  There is
#     nothing to tune there; the arithmetic has to be wider than the data.
#
#   * `precision_floor(T)` -- how far above zero the ARITHMETIC's own noise
#     sits, relative to `‖L‖`.  This is what the gap test measures its ratio
#     from, and what floors the nullity ceiling.  It follows
#     `compute_eltype(T)`, so a Float16 tensor gets Float32's floor.
#
#   * `data_floor(T)` -- how finely the STORED data resolves anything, relative
#     to its own scale: `eps(T)`.  Rounding Γ to Float16 perturbs it by ~2e-4
#     relative, so an eigenvalue of the derivation operator below that is an
#     artifact of the rounding as easily as a feature of the tensor.  Computing
#     in Float32 makes such a value *converge*; it does not make it *true*.
#     This floor is therefore not used to place the cut -- it is used to refuse
#     to certify one, which is how Float16 stays honest instead of accurate.
#
# When the stored and computed types agree, `precision_floor` is 100x
# `data_floor` and the data floor never binds: Float64 and Float32 behave
# exactly as before this file existed.  The second floor exists for the mixed
# case alone.
#
# The measured tables behind every constant are in the docstrings below and in
# docs/design/Precision-Policy.md; the raw sweeps are
# bench/reports/precision-tune.csv and bench/reports/precision-study.md.
#

"""
    TOL_DEFAULT

The default RELATIVE ceiling on "zero" that every derivation entry point takes
as its `tol` keyword, unchanged from the original hard-coded value so that
Float64 results users already rely on do not move.  It is a ceiling only: the
policy floors it at `precision_floor(T)`, and the gap test may refine the count
below it but never above it.
"""
const TOL_DEFAULT = 1e-6

"""
    FLOOR_EPS

The precision floor, in units of `eps(T_compute)`, below which a value a solver
returns is indistinguishable from zero -- the constant `c` in
`c * eps(T_compute) * norm(L)`.

TUNED, NOT GUESSED.  `bench/reports/precision-tune.jl` stage A sweeps `c` over
1 .. 1e4 on the scrambled sphere (valence 3 at d = 10, 16; valence 4 at d = 6),
random dense tensors at valence 3 and 4, video-shaped boxes 20x20x10x3 and
40x40x20x3, and the near-degenerate `ramp + 1e-3*noise` case, in all three
element types, for `SVDSolver`, `GramSolver`, `ArpackSolver` and
`KrylovSolver` (bench/reports/precision-tune-a.csv has every row):

| c | nullity correct? | Float64 certified? | Float32/16 certified? |
|---|---|---|---|
| 1    | NO -- sphere d = 10 Float32 SVD gives 2 of 3, Arpack 1 of 3, Gram d = 16 gives 0 of 3 | yes | partly |
| 2    | yes | yes | yes |
| 3    | yes | yes | yes |
| 5    | yes | yes | yes |
| 10   | yes | yes | yes (video-40 at the edge) |
| 20   | yes | yes | NO -- video-40x40x20x3 loses it on SVD, Arpack and Krylov |
| 100  | yes | yes | NO -- both video shapes uncertified on every solver |
| 1000 | yes | NO | NO |
| 1e4  | NO -- swallows the first nonzero eigenvalue everywhere | - | - |

So the nullity plateau is `c` in 2 .. 1000 and the CERTIFICATION plateau is
`c` in 2 .. 10; **5** is the geometric centre of the intersection, 2.5x above
the lower edge and 2x below the upper.

WHY 100 WAS WRONG.  The old value was calibrated on Float64, where the floor
is 2.2e-14 and every real gap is 1e5..1e11, so any `c` within four decades
looks identical.  In Float32 the floor is 1e9 times closer to the data:
measured null values sit at 0.07..2.0 `eps(T)` relative (Float64 and Float32
alike, over every case above), so `c = 5` clears the largest of them by 2.5x
while `c = 100` puts the floor 50x above the noise it is meant to describe --
and since the gap is measured FROM the floor, that threw away 1.5 decades of
gap.  On the video shapes, whose first nonzero eigenvalue is 1.6e-4..2.7e-4
relative (against the sphere's 1.5e-3..7.7e-3), that was the difference
between a gap of 22 and a gap of 2250, i.e. between an uncertified and a
certified Float32 answer.  No Float64 verdict in the whole sweep changes
between `c = 5` and `c = 100`.

NO DIMENSION TERM.  The natural guess is that the floor should grow with the
matrix size, `sqrt(n)*eps` or the `max(m,n)*eps` that `LinearAlgebra.rank`
uses.  The data says otherwise for THESE operators: the largest relative null
values occur at the SMALLEST `n` (2.0 eps at n = 165, sphere d = 10 Float32)
and the smallest at the largest (0.11 eps at n = 3609, video-40 Float32).  The
null directions of a derivation operator are not generic vectors -- their
images are small, so their rounding is relative to a small quantity and not to
`norm(L)` -- and a `sqrt(n)` term merely re-broke the video cases it was added
to protect (at n = 3609 it makes the floor 60 eps, past the c = 20 edge).  The
`max(m,n)*eps` convention is still right for a rank-revealing FACTORIZATION,
where the classical backward-error bound does apply; that lives in
`rank_rtol`, and `LUSolver` uses it.
"""
const FLOOR_EPS = 5

"""
    GAP_RATIO

Default minimum multiplicative jump that certifies a nullity; see
`gap_verdict`.  Chosen from the sphere stratification data
(bench/reports/gap-verdict.md) and re-checked against the wider case set in
bench/reports/precision-tune-a.csv.  With the precision floor at
`FLOOR_EPS = 5` times `eps(T_compute)`:

- Float64 (floor 1.1e-15): the null cluster sits at 1e-21..3e-16 relative, so
  it is floored, and the first nonzero eigenvalue of the derivation operator
  is 1.5e-4 (video shapes) to 8e-3 (sphere) relative.  Floored gaps are
  1e11..1e13.
- Float32 (floor 6.0e-7): the null cluster is 2e-9..2.4e-7 relative, likewise
  under the floor, and the first nonzero is 1.6e-4..7.7e-3, giving floored
  gaps of 270..1.3e4.
- Float16 data computed in Float32: the null cluster is the same 5e-9..2.4e-7
  -- the rounding of Γ turns out not to raise it measurably -- so the gap is
  the same as Float32's.  What Float16 loses is not the gap but the right to
  believe the value above it; that is `data_floor`'s job, not this constant's.

100 sits above every within-cluster ratio the floor does not already collapse,
and at least 2.7x below the smallest measured true gap (270, video-40 in
Float32).  It cannot be raised much: 1e3 would leave the Float32 video shapes
uncertified.  Lowering it is what `FLOOR_EPS` was tuned to avoid needing.

THE FLOOR IS NOT OPTIONAL, and one case in the sweep proves it.  On the
valence-4 sphere at d = 6 in Float64, `ArpackSolver` returns
`7.4e-32, 2.8e-19, 2.9e-17, 7.0e-17 | 1.5e-3` -- a true nullity of 4 whose
null cluster spans THIRTEEN decades.  Unfloored, the largest consecutive ratio
is `2.8e-19/7.4e-32 = 3.8e12` at k = 1, so a gap test would certify nullity 1
on a 4-dimensional null space.  Floored at 1.1e-15 all four collapse, every
within-cluster ratio drops below 0.1, and the cut lands at 4.  This is also
why the floor must NOT be squared for a squared (Gram) spectrum, tempting as
that is from the `λ = σ²` scaling: squaring it would restore exactly this
failure.
"""
const GAP_RATIO = 100.0

"""
    compute_eltype(T) -> Type

The element type the ARITHMETIC should run in for data stored as `T`.

Identity except for `Float16`, which promotes to `Float32`.  This is not a
tuning choice, it is what the libraries support: no CPU BLAS or LAPACK has a
half-precision path, so in `Float16`

| route | result |
|---|---|
| `ArpackSolver`  | `MethodError: no method matching saupd(..., ::Base.RefValue{Float16}, ...)` |
| `KrylovSolver`  | `MethodError: no method matching hschur!(::SubArray{Float16,...})` |
| `CGSolver`      | `MethodError: no method matching eigen!(::Hermitian{Float16,...})` |
| `SVDSolver`, `GramSolver`, `LSMRSolver` | run, and are WRONG: nullity 32-41 against a truth of 3-4, reconstruction error 0.85-0.94 |

The dense routes run because Julia emulates half-precision arithmetic in
software -- 1.5x SLOWER than Float64 on this CPU, so Float16 is not a speed
route either -- but their noise floor is `eps16 * ‖L‖ ~ 0.09` relative, an
order of magnitude ABOVE the first nonzero eigenvalue of the derivation
operator (2.6e-3..7.2e-3 on the sphere), so the null cluster and the rest of
the spectrum are one indistinguishable blur.

Float16 is therefore supported as a STORAGE and DATA type, which is what video
needs: `Γ` is held in Float16, promoted once at the derivation entry point,
solved in Float32, and the answer is rounded back to Float16.  Measured on the
sphere at d = 10..40, Float16 data solved in Float32 or Float64 recovers to
2.0e-4 relative -- exactly the rounding error of Γ itself
(bench/reports/precision-study.md, Exp. 3), i.e. the data, not the arithmetic,
is the whole error budget.

`Float64` and `Float32` are returned unchanged, so nothing about the existing
paths moves.  Apple GPUs have no Float64 at all, so `compute_eltype` also makes
the GPU routes reachable from a Float16 tensor: `Float16 -> Float32` is exactly
what those kernels want.
"""
compute_eltype(::Type{Float16}) = Float32
compute_eltype(::Type{T}) where {T<:AbstractFloat} = T
compute_eltype(::Type{Complex{T}}) where {T<:AbstractFloat} = Complex{compute_eltype(T)}
# BigFloat, Rational, Integer and anything else a caller invents: leave alone.
# Nothing in this file can improve on a type it has not measured.
compute_eltype(::Type{T}) where {T<:Number} = T

"""
    precision_floor(T) -> Float64

The RELATIVE level below which a value returned by a solver on `T`-data is
indistinguishable from zero: `FLOOR_EPS * eps(compute_eltype(T))`.

Follows `compute_eltype`, so a Float16 tensor gets Float32's floor -- the
arithmetic really does resolve that finely once promoted.  Whether the *data*
justifies believing it is `data_floor`'s question.

| T | compute | `precision_floor` |
|---|---|---|
| Float64 | Float64 | 1.1e-15 |
| Float32 | Float32 | 6.0e-07 |
| Float16 | Float32 | 6.0e-07 |

There is deliberately no dimension term; see `FLOOR_EPS` for the measurements
that rule one out for these operators, and `rank_rtol` for the place where
`max(m,n)*eps` is the right rule.
"""
precision_floor(::Type{T}) where {T<:Number} =
    Float64(FLOOR_EPS) * Float64(eps(real(float(compute_eltype(T)))))

"""
    data_floor(T) -> Float64

The RELATIVE resolution of data STORED as `T`: `eps(T)`.  Unlike
`precision_floor` this does NOT follow `compute_eltype` -- it is a statement
about the numbers that were handed in, not about the arithmetic done to them.

| T | `data_floor` | `precision_floor` | binding? |
|---|---|---|---|
| Float64 | 2.2e-16 | 2.2e-14 | no (100x below) |
| Float32 | 1.2e-07 | 1.2e-05 | no (100x below) |
| Float16 | 9.8e-04 | 1.2e-05 | YES (80x above) |

WHAT IT IS FOR.  It never places a cut.  It decides whether a cut may be
CERTIFIED: a value that the solver returns above the cut is only evidence of a
nonzero eigenvalue if it is larger than the noise the stored data already
carries.  Rounding Γ to Float16 moves it by 2e-4 relative, and moves the
eigenvalues of the derivation operator with it, so a first-nonzero eigenvalue
at 1e-4 in a Float16 run is not a fact about the tensor -- rounding could have
put it there, or taken it away.  The honest verdict on such a spectrum is
"undecidable at this precision", and that is what `gap_verdict` returns:
nullity 3 with `certified = false` and `undecidable = 1`, rather than a
confident 4.

Because `precision_floor` is `FLOOR_EPS = 100` times `data_floor` whenever the
stored and computed types agree, this floor is inert for Float64 and Float32
and the verdicts on those paths are unchanged.  It binds only in the mixed
case that `compute_eltype` creates.
"""
data_floor(::Type{T}) where {T<:Number} = Float64(eps(real(float(T))))
data_floor(::Type{T}) where {T<:Integer} = 0.0

"""
    tol_default(T; tol = TOL_DEFAULT, squared = false) -> Float64

The RELATIVE ceiling on "zero" for element type `T`: the caller's `tol`,
floored at `precision_floor(T)`, because no rank, feasibility or verification
check can certify anything finer than the arithmetic's own noise.

`squared = true` says the values being filtered are `σ²` (the eigenvalues of a
Gram operator `AᵗA`, which `solve_nullspace` forms itself for the
eigensolvers).  A ceiling of `tol` on `σ²` admits every direction with
`σ/σ_max <= sqrt(tol)`, so the tolerance is squared first and the accepted
directions satisfy the `σ/σ_max <= tol` the caller asked for.  The floor is
applied after squaring, so this can only tighten the ceiling as far as `T` can
actually resolve -- in Float32 the floor binds and `tol²` never takes effect.

| T | `tol = 1e-6` | squared | note |
|---|---|---|---|
| Float64 | 1.0e-06 | 1.1e-15 | the caller's value; the floor is 9 decades below |
| Float32 | 6.0e-07 | 6.0e-07 | the floor binds either way |
| Float16 | 6.0e-07 | 6.0e-07 | Float32's floor -- `data_floor` carries the honesty limit |

TUNED, NOT GUESSED.  Sweeping `tol` over 1e-3 .. 1e-9 on the sphere at
valence 3 and 4, the video-shaped boxes and the near-degenerate case
(bench/reports/precision-tune.csv) the returned nullity is correct and
certified for every `tol` in 1e-4 .. 1e-8 in Float64 and for every `tol` at
all in Float32 and Float16, where the floor binds and `tol` is decorative.
1e-6 is kept because it is what Float64 callers already have, and it sits in
the middle of that plateau.
"""
function tol_default(::Type{T}; tol::Real = TOL_DEFAULT,
                     squared::Bool = false) where {T<:Number}
    rel = Float64(tol)
    rel = squared ? rel^2 : rel
    return max(rel, precision_floor(T))
end

"""
    iter_tol(T, tol) -> Float64

An iterative solver's own stopping tolerance, floored at
`ITER_TOL_EPS * eps(compute_eltype(T))`.

Every Krylov and CG method here stops on a residual norm, and no residual gets
below a small multiple of `eps(T) * ‖L‖`; asking for less does not buy accuracy,
it buys `maxiter`.  Measured on the sphere at d = 30..40 in Float32, block
Lanczos with KrylovKit's default `1e-12` relative took 7-38 s and allocated
19-50 GB per solve, against 0.3-1 s and 1-3 GB with the floor -- for the same
nullity and the same eigenvectors, because only the stopping test was
unreachable (bench/reports/precision-study.md, Exp. 5, 5b).

`ITER_TOL_EPS` is a SEPARATE constant from `FLOOR_EPS`, and the two were
measured apart because they answer different questions: `FLOOR_EPS` says how
far above zero a converged value sits, `ITER_TOL_EPS` says how hard to work to
get there.  `bench/reports/precision-tune-iter.jl` sweeps the multiplier over
1 .. 100 x `eps(T)` for `ArpackSolver` and `KrylovSolver` on the sphere at
valence 3 and 4, the video box and a random valence-4 tensor, in all three
element types (bench/reports/precision-tune-c.csv):

| multiplier | outcome | cost, video-20 Float32 block Lanczos |
|---|---|---|
| 1   | same nullity, but the `maxiter` pathology returns | 3.97 s |
| 3   | identical nullity and certification to 100 | 1.01 s |
| 10  | identical | 0.69 s |
| 30  | identical | 0.63 s |
| 100 | identical | 0.45 s |

Every multiplier from 3 to 100 gives the SAME nullity and the same
certification on every case, and the cost falls monotonically, so 100 is
chosen: it is 33x clear of the `maxiter` pathology at the bottom, has no
measured penalty at the top, and is the value the earlier study already
validated end to end.  In Float64 it is 2.2e-14, below every default in the
package, so Float64 timings and accuracies are untouched.

This is deliberately NOT `sqrt(eps(T))`: that is 1.5e-8 in Float64 and would
loosen Float64 recovery from 1e-13 to 1e-10..1e-12 for a 30% saving, which is
the wrong trade for the path that certifies answers.  Nor is it the `eps(T)`
that `LSMRSolver` uses for its inner least-squares tolerance: that solver
applies its projector repeatedly, so its accuracy is not capped by one pass.
See the comment in ext/DletoIterativeSolversExt.jl.
"""
const ITER_TOL_EPS = 100

iter_tol(::Type{T}, tol::Real) where {T<:Number} =
    max(Float64(tol), Float64(ITER_TOL_EPS) * Float64(eps(real(float(compute_eltype(T))))))

"""
    qd_tolerance(T, tol = TOL_DEFAULT) -> Float64

The working tolerance for the solve-and-lift methods (`FastDer3ValentMethod`,
`QuickDerMethod`): the caller's `tol`, floored at BOTH `sqrt(data_floor(T))` and
`precision_floor(T)`.

The `sqrt` is not decoration.  Every decision inside solve-and-lift -- the rank
of the restricted system, the feasibility of the lift, the Z-law verification --
compares a product of two data-sized quantities against zero, so its relative
error is `sqrt` of the elementwise one, and none of them can certify anything
finer than that.  Note it is the STORED type's `eps` under the root, not the
computed type's: the restricted matrix is assembled from the tensor's own
entries, so it is the data that sets this limit, and a Float16 tensor promoted
to Float32 arithmetic gains speed and stability but not information.

| T | `sqrt(data_floor)` | `precision_floor` | `tol = 1e-6` gives |
|---|---|---|---|
| Float64 | 1.5e-08 | 2.2e-14 | 1.0e-06  (the caller's value) |
| Float32 | 3.5e-04 | 1.2e-05 | 3.5e-04 |
| Float16 | 3.1e-02 | 1.2e-05 | 3.1e-02 |

Measured: at these tolerances QuickDer returns the correct nullity on the
sphere at valence 3 and 4 in all three types, including Float16 (where the
SylverLining routes are hopeless without promotion), and sweeping the floor
constant over 0.3x .. 3x of `sqrt(eps)` changes no nullity
(bench/reports/precision-tune.csv).  Below 0.1x the Float16 lift starts
accepting spurious solutions and the verification rejects the whole answer.
"""
qd_tolerance(::Type{T}, tol::Real = TOL_DEFAULT) where {T<:Number} =
    max(Float64(tol), sqrt(data_floor(T)), precision_floor(T))

"""
    rank_rtol(T, m, n) -> Float64

The relative pivot threshold for a rank-revealing factorization of an `m × n`
matrix of element type `T`: `max(m, n) * eps(compute_eltype(T))`, which is
`LinearAlgebra.rank`'s convention and the classical backward-error bound for a
factorization (as opposed to the residual of an iterative solve, which
`precision_floor` covers).

At n = 1395 in Float32 this is 1.7e-4, against null diagonals at 4.8e-8..1.3e-7
and the first independent one at 5.4e-2 -- the same 3-against-1392 split as
Float64.
"""
rank_rtol(::Type{T}, m::Integer, n::Integer) where {T<:Number} =
    Float64(max(m, n)) * Float64(eps(real(float(compute_eltype(T)))))

"""
    precision_policy(T, n = 0) -> NamedTuple

Every number this file decides, for one element type, in one place -- handy at
the REPL and in tests:

```julia
julia> precision_policy(Float16, 1395)
(store = Float16, compute = Float32, precision_floor = 5.96e-7,
 data_floor = 9.77e-4, tol = 5.96e-7, iter_tol = 1.19e-5, rank_rtol = 1.66e-4,
 qd_tol = 3.13e-2)
```
"""
precision_policy(::Type{T}, n::Integer = 0) where {T<:Number} =
    (; store = T, compute = compute_eltype(T),
       precision_floor = precision_floor(T), data_floor = data_floor(T),
       tol = tol_default(T), iter_tol = iter_tol(T, TOL_DEFAULT),
       rank_rtol = rank_rtol(T, n, n), qd_tol = qd_tolerance(T))
