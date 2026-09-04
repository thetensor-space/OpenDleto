# The precision policy: what "zero" means in Float64, Float32 and Float16

Date: 2026-09-04.  Machine: M4 Max, 16 CPU cores, 64 GB, Julia 1.12, run
through `bench/jl` (2 slots x 4 threads).  Source: `src/solvers/Precision.jl`.
Sweeps, all under `bench/reports/`:

| exp | script | question | data |
|---|---|---|---|
| 0 | `precision-probe0.jl` | what does each route do per type, end to end? | the log |
| A | `precision-tune.jl A` | the verdict constants `FLOOR_EPS`, `tol` | `precision-tune-a.csv`, 1664 rows |
| C | `precision-tune-iter.jl` | the iterative stopping tolerance | `precision-tune-c.csv` |
| D | `precision-frontier.jl` | the frontier in `d` and in nullity | `precision-frontier.csv` |
| E | `precision-lift-residual.jl` | QuickDer's lift residual vs `d` | `precision-lift-residual.csv` |
| F | `precision-qd-tol-frontier.jl` | is QuickDer's Float32 undercount a tolerance? | `precision-qd-tol-frontier.csv` |
| G | `precision-qd-law.jl` | what tolerance does it need, vs `d`? | `precision-qd-law.csv` |

Plus the earlier `precision-study.md`, whose section 7 proposals this work
applies, validates, and in two places corrects.

## 0. The problem, in one table

Every route, the scrambled sphere octant, true nullity = valence, default
`tol`.  `lsq` is the relative reconstruction error after stratification.
BEFORE is the state at the start of this work.

| route | valence | Float64 | Float32 | Float16 BEFORE | Float16 AFTER |
|---|---|---|---|---|---|
| SylverLining/SVDSolver  | 3 | 3, 1e-14 | 3, 5.6e-6 | **40**, 0.94 | 3, 1.4e-2 |
| SylverLining/GramSolver | 3 | 3, 3e-14 | 3, 1.2e-5 | **41**, 0.91 | 3, 2.4e-3 |
| SylverLining/ArpackSolver | 3 | 3, 1.5e-14 | 3, 6.3e-6 | **MethodError** | 3, 1.5e-3 |
| SylverLining/KrylovSolver | 3 | 3, 1.7e-13 | 3, 1.0e-5 | **34**, 0.92 | 3, 1.6e-3 |
| SylverLining/LSMRSolver | 3 | 3, 1.1e-14 | 3, 1.4e-6 | **20**, 0.84 | 3, 1.5e-3 |
| SylverLining/AutoSolver | 3 | 3, 1e-14 | 3, 6.1e-6 | **41**, 0.90 | 3, 4.6e-3 |
| QuickDer | 3 | 3, 6.3e-13 | 3, 1.7e-4 | 3, 1.5e-2 | 3, 4.1e-3 |
| SylverLining/SVDSolver | 4 | 4, 7.8e-15 | 4, 4.0e-6 | **32**, 0.93 | 4, 6.3e-3 |
| SylverLining/KrylovSolver | 4 | 4, 1.2e-13 | **3**, 1.8e-2 | **32**, 0.85 | 4, 1.8e-3 |
| SylverLining/LSMRSolver | 4 | 4, 6.7e-15 | 4, 1.9e-6 | **25**, 0.88 | 4, 1.6e-3 |
| ... (7 routes x 3 types x valence 3, 4) | | | | | |
| **totals** | | 14/14 | 12/14 | **0/14** | **14/14** |

42 of 42 correct nullities after, against 24 of 42 before.  The recovery error
in each type is what that type's data carries: 1e-15..1e-13 (Float64),
1.4e-6..2.3e-5 (Float32), 1.5e-3..1.4e-2 (Float16).

Float16 was not *failing*; it was answering confidently and wrongly.  Half
precision has no BLAS or LAPACK on any CPU this package targets, so the
eigensolvers threw `MethodError` (`saupd`, `hschur!`, `eigen!`) while the dense
routes ran in Julia's software emulation -- 1.5x *slower* than Float64 -- with
a noise floor of `eps16 * ‖L‖ ~ 0.09` relative, an order of magnitude ABOVE the
derivation operator's first nonzero eigenvalue (2.6e-3..7.7e-3).  The null
cluster and the rest of the spectrum were one blur.

## 1. Three types, not one

`src/solvers/Precision.jl`.  The type a tensor is STORED in, the type the
arithmetic RUNS in, and the resolution a VERDICT may claim are three different
things, and conflating them is what made Float16 dishonest.

| function | means | Float64 | Float32 | Float16 |
|---|---|---|---|---|
| `compute_eltype(T)` | what the arithmetic runs in | Float64 | Float32 | **Float32** |
| `precision_floor(T)` | the ARITHMETIC's noise, relative; `FLOOR_EPS·eps(compute)` | 1.8e-15 | 9.5e-07 | 9.5e-07 |
| `data_floor(T)` | what the STORED data resolves; `eps(T)` | 2.2e-16 | 1.2e-07 | **9.8e-04** |
| `tol_default(T)` | the relative ceiling, `max(tol, precision_floor)` | 1.0e-06 | 1.0e-06 | 1.0e-06 |
| `tol_default(T; squared=true)` | the same on a Gram map, where `λ = σ²` | 1.8e-15 | 9.5e-07 | 9.5e-07 |
| `iter_tol(T, tol)` | an iterative solver's stopping test, `ITER_TOL_EPS·eps` floor | 1.0e-06 | 1.2e-05 | 1.2e-05 |
| `qd_tolerance(T)` | solve-and-lift, floored on the STORED type | 1.0e-06 | 3.5e-04 | 3.1e-02 |
| `rank_rtol(T,m,n)` | a factorization's pivot cut, `max(m,n)·eps` | (n=1395) 3.1e-13 | 1.7e-04 | 1.7e-04 |

Two floors, two jobs:

* **`precision_floor` places the cut.**  It is where `gap_verdict` measures its
  ratio from, and what floors the nullity ceiling.  It follows
  `compute_eltype`, because once promoted the arithmetic really does resolve
  that finely.
* **`data_floor` never places a cut -- it vetoes CERTIFYING one.**  Rounding Γ
  to Float16 moves it by 2e-4 relative and carries the eigenvalues of its
  derivation operator along, so a first-nonzero eigenvalue below that could
  have been put there, or removed, by the rounding alone.  Values above the cut
  and below `data_floor` are counted in `NullVerdict.undecidable`, and a
  nonzero count clears `certified`.

Because `precision_floor` is `FLOOR_EPS` times `data_floor` whenever the stored
and computed types agree, the data floor is **inert for Float64 and Float32**.
It exists for the mixed case that `compute_eltype` creates, and no Float64 or
Float32 verdict changed when it was added.

Float16 in gives Float16 out: the promotion happens once at the derivation
entry point and the coordinates are rounded back, so the answer carries exactly
the precision the data justifies and no more.

## 2. `FLOOR_EPS`: 100 -> 8

The single most consequential number.  Stage A solves each case once per
(type, solver), keeps the spectrum, and scores every candidate `c` against it
offline -- so two constants are compared on the same numbers, not on two runs
of a randomised solver.  Cases: sphere valence 3 at d = 10, 16; sphere valence
4 at d = 6; random dense 12^3 and 6^4; video boxes 20x20x10x3 (n = 909) and
40x40x20x3 (n = 3609); `ramp + delta*noise` at 6^3 for delta = 1e-3, 1e-8.

| c | nullity correct? | Float64 certified? | Float32 / Float16 certified? |
|---|---|---|---|
| 1 | **NO** -- sphere d = 10 Float32 SVD 2 of 3, Arpack 1 of 3; Gram d = 16 and rand4-d6 0 of 3 | yes | partly |
| 2 | yes | yes | yes |
| 3 | yes | yes | yes |
| 5 | yes | yes | yes |
| **8** | **yes** | **yes** | **yes** |
| 10 | yes | yes | yes, video-40 at the edge |
| 20 | yes | yes | **NO** -- video-40 loses it on SVD, Arpack and Krylov |
| 30 | yes | yes | NO |
| 100 (old) | yes | yes | **NO** -- both video shapes uncertified on every solver |
| 300 | yes | yes | NO |
| 1000 | yes | **NO** | NO |
| 1e4 | **NO** -- swallows the first nonzero eigenvalue everywhere | - | - |

On these cases the nullity plateau is `c` in 2..1000 and the certification
plateau is `c` in 2..10.  Exp. D then pins it from both sides at once, over
~70 (d, valence, type) points on the sphere out to d = 96:

* LOWER BOUND `c > 4.92`.  The threshold must sit above the null cluster or a
  null vector is excluded.  The largest relative null value anywhere is
  **4.92 eps** (d = 32, valence 3, Float16).
* UPPER BOUND `c <= 12.6`.  The gap is measured FROM the floor, so
  `first_nonzero / (c·eps) >= GAP_RATIO`.  The tightest case is the sphere at
  d = 48 in Float32, first nonzero 1.50e-4 = 1260 eps, giving `c <= 12.6`;
  video-40 (1340 eps) gives 13.4.

Window `[4.92, 12.6]`, geometric centre 7.9, hence **`c = 8`**: 1.63x above the
worst null value and 1.58x below the tightest certification bound.

**That window is only 2.6x wide, and that is the honest headline.**  In Float64
the same two bounds are eleven decades apart, which is why any `c` within four
decades looked correct when this constant was last chosen.  Float32 and Float16
have no such slack; a tensor 2x worse conditioned than the sphere at d = 48
returns the right nullity and declines to certify it, and no `c` avoids that.
Widening the window means lowering `GAP_RATIO` -- see the note there.

Why 100 was wrong.  It was calibrated on Float64, where the floor is 2.2e-14
and every real gap is 1e5..1e11, so four decades of `c` look identical.  In
Float32 the floor is nine orders of magnitude closer to the data, and 100 eps
lands a factor of 8 outside the upper bound above.  Measured relative null
values, over every case and all three types:

| T | solver | measured null values, relative | in units of `eps(T)` |
|---|---|---|---|
| Float64 | SVD, video-20 | 2.4e-18 .. 1.6e-17 | 0.011 .. 0.07 |
| Float64 | Gram, sphere d = 16 | 2.3e-16 .. 2.7e-16 | 1.0 .. 1.2 |
| Float64 | Arpack, sphere 4-valent | 7.4e-32 .. 7.0e-17 | 3e-16 .. 0.32 |
| Float32 | SVD, sphere d = 10 | 3.8e-8 .. 2.4e-7 | 0.32 .. **2.0** |
| Float32 | Arpack, sphere d = 10 | 3.2e-8 .. 2.2e-7 | 0.27 .. 1.8 |
| Float32 | SVD, video-40 (n = 3609) | 1.7e-9 .. 1.3e-8 | 0.014 .. **0.11** |
| Float16 (computed Float32) | SVD, sphere d = 10 | 3.8e-8 .. 2.4e-7 | same as Float32 |

| Float16 (computed Float32) | Arpack/SVD, sphere d = 16..96 | 9.7e-8 .. 5.9e-7 | 0.81 .. **4.92** |

So the noise this constant describes never exceeds **4.92 `eps(T)`**, and it
does not trend with size: 1.67 eps at d = 16, 0.66 at d = 64, 2.80 at d = 96,
0.11 at n = 3609.  `c = 8` clears the worst of it by 1.6x; `c = 100` put the
floor 20x above the noise -- and since the gap is measured FROM the floor, that
threw away 1.1 decades of gap.  On the video shapes, whose first nonzero
eigenvalue is 1.6e-4..2.7e-4 relative (against the sphere's 1.5e-4..7.7e-3),
that was the difference between a gap of 22 and one of 1400, i.e. between an
uncertified and a certified Float32 answer.  **No Float64 verdict in either
sweep changes between 8 and 100.**

### No dimension term

The natural guess -- `sqrt(n)·eps`, or the `max(m,n)·eps` that
`LinearAlgebra.rank` uses -- was tried first and the data refutes it for the
null values of these operators.  They are flat in `d` from 16 to 96 and flat in
`n` from 84 to 13968, and the largest relative values occur at the SMALLEST
sizes (2.0 eps at n = 165, 4.92 eps at n = 1584, against 0.11 eps at
n = 3609).  A derivation operator's null directions are not generic vectors:
their images are small, so their rounding is relative to a small quantity, not
to `‖L‖`.  Adding `sqrt(n)` made the floor 60 eps at n = 3609 -- five times
past the upper bound -- and so re-broke exactly the video cases it was meant to
protect.

What DOES move with the problem is the *first nonzero* eigenvalue, which falls
from 2.4e-3 at d = 32 to 1.5e-4 at d = 48 and fluctuates more than 10x between
neighbouring `d`.  So the frontier is set by the tensor's conditioning, not by
the arithmetic's size -- which is exactly why the cut is decided by a GAP and
not by this constant alone.

`max(m,n)·eps` remains right for a rank-revealing **factorization**, where the
classical backward-error bound does apply.  That lives in `rank_rtol`, and
`LUSolver` and `LSMRSolver`'s revealing QR use it.

### The floor must not be squared, either

`solve_nullspace` is handed the Gram operator `AᵗA`, whose values are
`λ = σ²`, and it already squares the *ceiling* for that reason.  Squaring the
*floor* to match is tempting and wrong, and one case proves it: on the
valence-4 sphere at d = 6 in Float64, `ArpackSolver` returns
`7.4e-32, 2.8e-19, 2.9e-17, 7.0e-17 | 1.5e-3` -- a true nullity of 4 whose null
cluster spans thirteen decades.  Unfloored, the largest consecutive ratio is
`2.8e-19/7.4e-32 = 3.8e12` at k = 1, so a gap test would **certify nullity 1 on
a 4-dimensional null space**.  Floored at 1.1e-15 all four collapse, every
within-cluster ratio drops below 0.1, and the cut lands at 4.  Collapsing the
cluster is the floor's real job; a squared floor would not do it.

## 3. `ITER_TOL_EPS = 100`: measured separately, and kept

`FLOOR_EPS` says how far above zero a converged value sits; `ITER_TOL_EPS`
says how hard to work to get there.  Different questions, so they were swept
apart.  Stage C sweeps the multiplier for `ArpackSolver` and `KrylovSolver` on
the sphere at valence 3 and 4, the video box and a random valence-4 tensor, in
all three types.

| multiplier | nullity / certification | cost: video-20 Float32 block Lanczos | sphere-4 Float32 Krylov |
|---|---|---|---|
| 1 x eps | same | **3.97 s** | **1.78 s** |
| 3 x eps | identical to 100 | 1.01 s | 0.05 s |
| 10 x eps | identical | 0.69 s | 0.05 s |
| 30 x eps | identical | 0.63 s | 0.04 s |
| **100 x eps** | identical | **0.45 s** | 0.03 s |

Every multiplier from 3 to 100 gives the same nullity and the same
certification on every case, and the cost falls monotonically, while `1 x eps`
restores the `maxiter` pathology the earlier study found (40-190x).  So 100 is
kept: 33x clear of the pathology, no measured penalty above, and the value
already validated end to end in `precision-study.md`.

**`LSMRSolver` is the exception, and gets `eps(T)`.**  Its `atol`/`btol` are
relative tests *inside* the inner least-squares solve, and `project!` applies
the projector repeatedly, so each pass starts from a smaller `‖y‖` and the
composite accuracy is not capped by one pass the way a single Lanczos residual
is.  At `100 eps(Float32) = 1.2e-5` the sphere at d = 10 lost 270x of recovery
(1.4e-6 -> 4.0e-4 relative) for a 15% time saving, and the valence-4 case lost
a null vector outright (4 -> 3).  At `eps(Float32) = 1.2e-7` both are restored.
Its `rank_tol` (the pivot cut of its revealing QR, an absolute 1e-8 that sits
BELOW `eps(Float32)`) gets `rank_rtol` -- the factorization rule.

## 4. `qd_tolerance`: the solve-and-lift tolerance floors on the STORED type

Every check inside solve-and-lift -- the rank of the restricted system, the
feasibility of the lift, the Z-law verification -- compares a product of two
data-sized quantities against zero, so its relative error is the square root of
the elementwise one.  And it is the **stored** type's `eps` under the root:
the restricted matrix is assembled from the tensor's own entries, so the data
sets the limit, and promoting Float16 to Float32 arithmetic buys stability, not
information.

| T | `sqrt(data_floor(T))` | `precision_floor(T)` | `qd_tolerance(T)` |
|---|---|---|---|
| Float64 | 1.5e-08 | 1.8e-15 | **1.0e-06** (the caller's `tol`) |
| Float32 | 3.5e-04 | 9.5e-07 | **3.5e-04** |
| Float16 | 3.1e-02 | 9.5e-07 | **3.1e-02** |

Swept end to end through QuickDer over `tol` = 1e-9 .. 30x the floor: values
below the floor are clamped to it and all agree (the floor doing its job), and
the correct nullity holds from the floor up to ~10x it in every type.  This is
the one tolerance that was already type-relative before this work
(`_qd_tolerance`, which now delegates here), and it is why QuickDer was the
only route that got Float16 right from the start.

## 5. The frontier: what Float32 can certify, as a function of d and nullity

The most important table here, because the downstream data is Float32 video.
Exp. D (`precision-frontier.jl`, `precision-frontier.csv`) walks the scrambled
sphere matrix-free, through the same call `derTrOpsReduced(::SylverLiningMethod,
...)` makes, and records the verdict and the spectrum around the cut.  `1st` is
the first nonzero eigenvalue relative to `‖AᵗA‖` -- the number that sets
everything.  `null` is the largest null value in units of `eps(T_compute)`.

### 5.1 Nullity 3 (valence 3, `SymmetricOp`)

| d | n | 1st nonzero | Float64 | Float32 | Float16 | QuickDer F32 |
|---|---|---|---|---|---|---|
| 16 | 408 | 1.47e-3 | 3 cert (1.7 eps) | 3 **cert**, gap 2467 (3.6 eps) | 3 **cert**, gap 2468 | 3 |
| 24 | 900 | 2.94e-4 | 3 cert (0.6) | 3 **cert**, gap 493 (1.5) | 3, UNDECIDABLE(1) | 3 |
| 32 | 1584 | 2.43e-3 | 3 cert (0.7) | 3 **cert**, gap 4081 (1.6) | 3 **cert**, gap 4081 (4.9) | 3 |
| 48 | 3528 | 1.50e-4 | 3 cert (2.0) | 3 **cert**, gap 252 (1.2) | 3, UNDECIDABLE(3) | **2** |
| 64 | 6240 | 5.77e-4 | 3 cert (2.7) | 3 **cert**, gap 968 (0.7) | 3, UNDECIDABLE(1) | **2** |
| 96 | 13968 | 2.09e-4 | 3 cert (1.4) | 3 **cert**, gap 350 (2.8) | 3, UNDECIDABLE(1) | 3 |
| 140 | 29610 | - | - | - | - | **0** |

### 5.2 Nullity 4 (valence 4, `SymmetricOp`)

| d | n | 1st nonzero | Float64 | Float32 | Float16 |
|---|---|---|---|---|---|
| 6 | 84 | 1.50e-3 | 4 cert | 4 **cert**, gap 2513 | 4 **cert**, gap 2514 |
| 8 | 144 | 2.22e-3 | 4 cert | 4 **cert**, gap 3717 | 4 **cert**, gap 3717 |
| 10 | 220 | 1.26e-3 | 4 cert | 4 **cert**, gap 2119 | 4 **cert**, gap 2119 |
| 12 | 312 | 8.25e-4 | 4 cert | 4 **cert**, gap 1384 | 4, UNDECIDABLE(1) |
| 16 | 544 | 1.96e-4 | 4 cert | 4 **cert**, gap 328 | 4, UNDECIDABLE(1) |

### 5.3 Nullity 13 (valence 3, `UniversalOp` on the raw octant)

The same tensors, the same `d`, a 13-dimensional derivation space instead of 3.
This separates size from multiplicity.

| d | n | Float64 | Float32 | Float16 | QuickDer F16 |
|---|---|---|---|---|---|
| 12 | 432 | 13 cert, gap 2.9e10 | 13, **uncert**, gap 54 | 13, uncert, UNDECIDABLE(5) | **14** |
| 16 | 768 | 13 cert, gap 1.5e11 | 13 **cert**, gap 271 | 13, uncert, UNDECIDABLE(11) | **14** |
| 24 | 1728 | 13 cert, gap 1.3e10 | **12**, uncert, gap **1.17** | **12**, uncert, UNDECIDABLE(20) | **20** |
| 32 | 3072 | 13 cert, gap 4.0e10 | 13, **uncert**, gap 75 | 13, uncert, UNDECIDABLE(10) | **19** |

### 5.4 The three rules that fall out

1. **Nullity is set by `d` not at all; certification is set by the tensor's
   conditioning.**  The null cluster is FLAT: 0.27..4.92 `eps(T_compute)` over
   every point above, `d` from 6 to 96 and `n` from 84 to 13968, in all three
   types.  What moves is the first nonzero eigenvalue -- 1.5e-4 to 2.4e-3 on the
   sphere, fluctuating more than 10x between neighbouring `d` -- and since the
   certificate needs `1st / (FLOOR_EPS·eps) >= 100`, that is what decides.
   **Float32 certifies when the first nonzero eigenvalue exceeds
   `100·8·eps(Float32)` = 9.5e-5 relative.**  Every sphere point from d = 16 to
   96 clears it; both video shapes (1.6e-4, 2.7e-4) clear it by 1.6x and 2.8x.
2. **Float16 certifies when the first nonzero eigenvalue exceeds
   `eps(Float16)` = 9.8e-4** -- and reports UNDECIDABLE otherwise, with the
   right nullity.  That is `data_floor` working exactly as designed, and the
   correlation is perfect over the 11 points above: certified at
   1.47e-3, 2.43e-3, 1.50e-3, 2.22e-3, 1.26e-3; undecidable at 2.94e-4,
   1.50e-4, 5.77e-4, 2.09e-4, 8.25e-4, 1.96e-4.  No exceptions.
3. **Larger nullity is harder, in Float32 and Float16 both.**  Going from
   nullity 3 to 13 on the same tensors drops the Float32 gap from 252..4081 to
   1.17..271: the 13th and 14th eigenvalues are far closer together than the
   3rd and 4th are.  At d = 24 the gap is 1.17 and Float32 undercounts to 12.
   So Float32's usable envelope is **nullity <= 4 comfortably, nullity ~13
   only where the gap happens to be wide** -- and it says so rather than
   guessing.

## 5.5 QuickDer in Float32 undercounts past d ~ 32, and it IS a tolerance

The d = 500 report (valence 3, Float32, whitened matrix-free QuickDer: nullity
1 of 3, residual 2.23e-4, uncertified, identically on CPU and GPU) reproduces
five times smaller and is fully bracketed here.  Exp. E
(`precision-lift-residual.jl`) and Exp. F (`precision-qd-tol-frontier.jl`):

| d | r | QuickDer Float64 | QuickDer Float32 | QuickDer Float16 |
|---|---|---|---|---|
| 16 | 8x8x8 | 3 | 3 | 3 |
| 32 | 11x11x11 | 3 | 3 | 3 |
| 48 | 13x13x13 | 3 | **2** | 3 |
| 64 | 15x15x15 | 3 | **2** | 3 |
| 100 | 19x19x19 | 3 | **1** | 3 |
| 140 | 22x22x22 | - | **0** | - |
| 500 | ~40 | 3 (residual 3.2e-13) | **1** (residual 2.23e-4) | - |

Note which type fails: **Float16, with a 90x COARSER `qd_tolerance` (3.1e-2
against 3.5e-4), gets the right answer at every `d`.**  That is the whole clue,
and Exp. F confirms it -- sweeping `tol` at the failing sizes:

| tol passed | `qd_tolerance` in force | nullity at d = 64 | at d = 100 |
|---|---|---|---|
| 1e-6, 1e-5, 1e-4, 3.45e-4 | 3.45e-4 (the floor) | **2** | **1** |
| 1e-3 | 1e-3 | **3** | **3** |
| 3e-3, 1e-2, 3e-2 | as passed | 3 | 3 |

A clean step, so there is a threshold, and Exp. G (`precision-qd-law.jl`)
measures it on a 1/3-decade grid:

| d | smallest `tol` that recovers nullity 3 | as a multiple of `sqrt(eps(Float32))` |
|---|---|---|
| 16 | <= 4.64e-4 | <= 1.34 |
| 32 | <= 4.64e-4 | <= 1.34 |
| 48 | 6.81e-4 | 1.97 |
| 64 | <= 4.64e-4 | <= 1.34 |
| 100 | 1.00e-3 | 2.90 |
| 140 | <= 4.64e-4 | <= 1.34 |

**The requirement is essentially FLAT in `d`** -- between 1.3x and 2.9x of
`sqrt(eps(Float32))` from d = 16 to d = 140 -- so this is not a size effect at
all.  `qd_tolerance = sqrt(eps(T))` simply sits ON the cliff edge: it is
within 1.3x of the value that works, and whether a given `d` falls on the
right side of it is luck.

### Why the constant was NOT simply raised

The obvious fix is a multiplier: `C·sqrt(eps(T))` with `C >= 2.9`.  The
nullity-13 rows of section 5.3 rule it out.  At d = 24, `UniversalOp`,
Float32, the spectrum around QuickDer's cut is

    2.46e-6, 3.01e-6, 2.49e-4 | 4.56e-4, 6.76e-4

with the threshold at 3.45e-4.  A 4x coarser tolerance swallows 4.56e-4 and
6.76e-4 and overcounts; `C = 2.9` would already be marginal.  So the lower
bound from nullity 3 (2.9x) sits ABOVE the upper bound from nullity 13
(~1.3x): **no single multiplier serves both**, and raising the constant would
trade a silent undercount at low nullity for a silent overcount at high
nullity.  Float16 already demonstrates the overcount half: with its 90x
coarser tolerance it returns 14, 14, 20 and 19 where the truth is 13.

The asymmetry that makes the undercount worse than the overcount is worth
naming.  QuickDer's lift filter is a HARD cut --
`nullspace(Rall; atol = qd_tolerance(T), rtol = 0)` -- applied before
`solve_nullspace` ever sees a spectrum, so the gap test cannot rescue what it
discards: the d = 100 Float32 run returns 1 of 3 with no way to know.  The
overcount, by contrast, goes through `gap_verdict` and comes back
**UNCERTIFIED**, which is at least honest.  The fix therefore belongs in that
filter -- a gap test on the scale-normalized residual spectrum of `Rall`, which
is already computed, instead of a fixed cut -- and not in this constant.  That
is inside `_qdn_solve_and_lift`, which is being rewritten for the whitened
restriction, so it is left to that rewrite rather than patched here.

**Until then, the operational guidance is:** Float32 QuickDer is reliable to
d ~ 32 at nullity 3; past that pass `tol = 1e-3` explicitly (which recovers
the full nullity at every `d` measured, up to 140), or use
`:SylverLining` with `:ArpackSolver`, which is correct AND certified in
Float32 at every `d` up to 96 and at both video shapes.

## 5.6 What each type can and cannot do

## 6. Solver defects the sweep exposed (not tolerance problems)

1. **Block Lanczos below Float64 at small n.**  With the default request
   (nev = 24) `KrylovKit.BlockLanczos` in Float32 returned nullity **0 of 4** at
   n = 84 and 1 of 3 at n = 165, was correct at n = 408, 909 and 3609, and was
   correct at all of them in Float64.  Sweeping its stopping tolerance over
   1..100 x eps changed nothing, so this is block re-orthogonalization losing
   rank in half a mantissa.  It mattered more than a wrong count usually would:
   the values it returned were 3e-4..6.5e-3, all above the threshold, so the
   verdict was a **certified nullity 0** on a 4-dimensional null space.  Fixed
   by widening its dense-`eigen` gate 4x below Float64 (`8*(nev+1)` = 200:
   past the largest observed failure at 165, below the smallest observed
   success at 408).  Its Float32 `DomainError` / `ArgumentError` breakdowns on
   the near-degenerate tensors now fall back to single-vector Arnoldi with a
   warning rather than crashing.
2. **`GramSolver` below Float64 past n ~ 900.**  Its null values sit at
   `n·eps(T)` -- 9e-5 relative at n = 909, 1.8e-4 at n = 3609 -- which is *at*
   the video shapes' first nonzero eigenvalue, so it reports nullity 0 and
   cannot be rescued by any floor: a floor high enough to admit its nulls also
   admits real eigenvalues.  Left as is, because `GramSolver` is opt-in
   (`matrix_free_solvers` never selects it, so `AutoSolver` never does either)
   and it already warns.  Its own docstring says as much: "in Float32 the
   Gram's own roundoff is n*eps and a system this large may simply need
   Float64".
3. **`QuickSylver` is Float64-only** by construction (`RHS = zeros(Float64,
   size(G))`), independent of `eltype(Γ)`.  Untouched here; its default
   tolerances now come from the policy but its arithmetic does not respond to
   the element type.

## 7. Files

* `src/solvers/Precision.jl` -- the policy, with the measured tables in the
  docstrings
* `test/TestPrecision.jl` -- 396 assertions, parametrized over the three types
* `bench/reports/precision-probe0.jl` -- the per-route, per-type end-to-end
  probe of section 0
* `bench/reports/precision-tune.jl` -- stage A (verdict constants, 1664 rows in
  `precision-tune-a.csv`), stage B (LSMR, QuickDer)
* `bench/reports/precision-tune-iter.jl` -- stage C, `precision-tune-c.csv`
* `bench/reports/precision-frontier.jl` -- stage D, the frontier in `d` and
  nullity, `precision-frontier.csv`
* `bench/reports/precision-lift-residual.jl` -- stage E, QuickDer's lift
  residual against its tolerance, `precision-lift-residual.csv`
* `bench/reports/precision-qd-tol-frontier.jl` -- stage F, `precision-qd-tol-frontier.csv`
* `bench/reports/precision-qd-law.jl` -- stage G, `precision-qd-law.csv`
* `bench/reports/precision-tune-report.py` -- renders the sweep CSVs as the
  tables above
* `bench/reports/precision-study.md` -- the earlier Float32/Float16 study whose
  section 7 proposals this work applies, validates and in two places corrects
