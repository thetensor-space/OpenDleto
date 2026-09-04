# QuickDer-W: does whitening the restricted system buy iterations?

2026-09-04.  Raw numbers: `whitened.csv` (regenerate the tables below with
`bench/jl bench/reports/2026-09-04/whitened/summarise.jl`).  Harness:
`bench/WhitenedRestriction.jl`.  Design: `docs/design/QuickDer-valence-n.md`
section 2a.  Code: `src/solvers/QuickDerN.jl`, `_qdn_whiten_axis` and friends.

## The question

Above d ≈ 200 at valence 3 the dense/Gram route does not fit and the
MATRIX-FREE restricted branch is the only one left.  Session 3 recorded that
branch as "converging slowly" on structured tensors, with ARPACK hitting its
iteration cap.  Since the dense branch is at the BLAS floor, iteration count is
the only lever.

The claim under test: the ill-conditioning of the restricted system is almost
entirely SKETCH-INDUCED, and can be removed exactly and cheaply.

## The structure (checked first, before any implementation)

In this file's convention the axis-`a` column block of the restricted system is

    A_a = I_{r_a} ⊗ M_a,   M_a := unfold_a(S_a)ᵗ   (R_a × d_a)

up to the row permutation `_qdn_row_perm` bookkeeps, so the diagonal blocks of
the restricted Gram are Kronecker:

    A_aᵗ A_a = c_a · (I_{r_a} ⊗ M_aᵗ M_a),   c_a = Σ_ρ P[ρ,a]²

Measured relative error of that identity, over the whole assembled matrix:

| case | worst axis |
|---|---|
| dims (8,9,10), UniversalChisel(3) | 2.3e-16 |
| dims (8,9,10), CentroidChisel(3) (3 rows, ±1) | 1.4e-16 |
| dims (5,6,7,8), UniversalChisel(4) | 5.1e-16 |
| dims (5,6,7,8), CentroidChisel(4) (6 rows) | 3.8e-16 |
| dims (8,9,10), AdjointChisel(3,1,2) (axis 3 disengaged) | 2.7e-16 |

So the whole sketch-induced conditioning lives in `d_a × d_a` Grams, and a thin
QR `M_a = Q_a R_a` with the substitution `Ỹ_a = R_a Y_a` removes it exactly.
(The design note writes `Y_a` as `r_a × d_a` and the substitution as
`Ỹ_a = Y_a R_aᵗ`; the code's `Y_a` is `d_a × r_a`, so it is `Ỹ_a = R_a Y_a`.)

**Where the conditioning actually is.**  Effective condition number
`σ_max/σ_{k+1}` of the restricted operator (`k` = nullity), unwhitened vs
whitened, which is what a Krylov method sees:

| case | max cond(M_a) | eff. cond plain | eff. cond whitened | gain |
|---|---|---|---|---|
| scrambled sphere v3 d = 20 | 20 | 3.2e3 | 8.9e2 | 3.6x |
| scrambled sphere v3 d = 30 | 26 | 4.1e2 | 8.5e1 | 4.8x |
| scrambled sphere v3 d = 60 | 60 | 5.4e5 | 2.7e4 | 20x |
| scrambled sphere v4 d = 14 | 79 | 1.0e3 | 4.8e1 | 21x |
| scrambled sphere v4 d = 18 | 153 | 4.8e3 | 1.4e2 | 35x |
| smooth video 20x20x10x3 | **3.5e16** | -- | -- | rank deficient |
| randn 20^3 | 3.2 | 1.5e1 | 1.0e1 | 1.4x |
| randn 40^3 | 3.0 | 2.4e1 | 1.7e1 | 1.4x |

Two predictions fall out, and both are confirmed below: a STRUCTURED tensor
gains a lot, a GENERIC one gains ~1.4x because there was nothing to remove, and
a SMOOTH tensor is not ill-conditioned at all but numerically rank deficient --
a different problem, handled by truncation.

## Measurement: the matrix-free branch

Every row forces the matrix-free branch (`QDN_DENSE_BUDGET_BYTES = 0`) so both
settings meet the same solver at the same size.  `applies` counts forward +
adjoint applications of the restricted map (`Dleto.QDN_APPLY_COUNT`), so it is
two ticks per ARPACK application of `AᵗA`.  Float64, 5 threads, `bench/jl`
budget; wall times are noisy because a second job shared the machine, which is
why the apply count is the headline number.  ARPACK is QuickDer's own default
on this branch (`_qdn_default_free_solver`), so the `AutoSolver` column IS the
current default.

### Scrambled sphere octant, valence 3 (SymmetricOp, oracle nullity 3)

| d | plain applies | plain result | whitened applies | whitened result | iter |
|---|---|---|---|---|---|
| 30  | 53632  | **ARPACK hit its cap, nullity 0** | 23732 | nullity 3, resid 2.8e-13 | 2.3x |
| 40  | 55196  | **ARPACK hit its cap, nullity 0** | 30796 | nullity 3, resid 1.2e-12 | 1.8x |
| 100 | 66900  | **ARPACK hit its cap, nullity 0** | 34636 | nullity 3, resid 7.8e-11 | 1.9x |
| 150 | 177142 | **ARPACK hit its cap, nullity 0** | 39544 | nullity 3, resid 1.7e-12 | 4.5x |
| 200 | 65182  | **ARPACK hit its cap, nullity 0** | 38262 | nullity 3, resid 3.3e-10 | 1.7x |

The apply ratios (1.7x-4.5x) UNDERSTATE the result, because the plain column
never produced an answer: its count is the cap, not a convergence cost.  The
honest statement is that the unwhitened matrix-free branch does not converge on
this tensor family at any d that was tried, and the whitened one converges at
every one, verified by the Z-law, at a whitening cost of 0.01 s in an 18 s
solve.

For scale: session 3's dense Gram route answered d = 200 in 22 s with an 8 GB
peak and a 2.2 GB restricted matrix.  Whitened matrix-free does it in 18.4 s at
2.1 GB peak and no matrix at all.

### Random dense, valence 3 (UniversalOp, oracle nullity 2) -- the control

The only family where the unwhitened branch converges, so the only one with a
finite iteration ratio:

| d | plain applies | plain seconds | whitened applies | whitened seconds | iter | time |
|---|---|---|---|---|---|---|
| 150 | 14618 | 5.6 | 10156 | 3.3 | 1.44x | 1.7x |
| 250 | 22974 | 16.7 | 15712 | 11.4 | 1.46x | 1.5x |

1.4x, exactly what the effective-condition table predicts for a generic tensor.
This is the negative control that says the sphere result is the whitening and
not an accident of the harness.

### Valence 4 and the video shape -- the cases with a finite ratio and a right answer

Here the unwhitened branch also fails, but the whitened counts are small enough
that the ratio against the cap is itself at or above 3x -- so these are the rows
that meet the "at least 3x fewer iterations" bar on its own terms, with the
plain column's count being only a lower bound on what it would have needed:

| case | oracle | plain applies / result | whitened applies / result | iter |
|---|---|---|---|---|
| sphere v4 d = 60 (60^4 Float64) | 4 | 61994 / **cap, nullity 0** | 12808 / nullity 4, resid 1.1e-13, 2.8 s | 4.8x |
| sphere v4 d = 80 (80^4 Float64) | 4 | 64584 / **cap, nullity 0** | 19718 / nullity 4, resid 2.4e-13, 5.5 s | 3.3x |
| video 200x200x100x3 Float32 | 3 | 56980 / **cap, nullity 0** | 11182 / nullity 3, resid 1.7e-5, 3.2 s | 5.1x |
| degenerate 40^3, mode-1 rank 38 | 82 | 59640 / **nullity 26 of 82** | 4024 / nullity 82, resid 8.4e-16 | 14.8x |

The degenerate row is the sharpest confirmation of the mechanism: 26 is exactly
`2 + 2·r_1` with `r_1 = 12`, the `W`-visible slice the analysis predicts the
unwhitened branch can see, and 82 is exactly `2 + 2·d`, the whole trivial space
plus the two scalars.  The Float32 video row is `uncertified` (the verdict's
gap test cannot clear a Float32 precision floor at this nullity) but lands the
oracle nullity at the Float32 residual bound.

### The frontier

Whitened, ARPACK, Float64, matrix-free, valence 3, scrambled sphere:

| d | restricted system | applies | seconds | peak RSS | nullity | residual | plain at the same size |
|---|---|---|---|---|---|---|---|
| 200 | 17576 x 15600 | 38262 | 18.4 | 2.1 GB | 3 | 3.3e-10 | 65182 applies, **cap** |
| 300 | 29791 x 27900 | 77482 | 105.0 | 3.8 GB | 3 | 1.8e-10 | 66604 applies, **cap** |
| 500 | 64000 x 60000 | 51116 | 136.1 | 11.4 GB | 3 | 3.2e-13 | (queued) |

**d = 300 and d = 500 at valence 3 are new.**  Session 3's frontier was d = 200
on the dense Gram route (22 s, 8 GB peak, a 2.2 GB restricted matrix).  The
whitened matrix-free branch does d = 200 in 18.4 s at 2.1 GB with no matrix at
all, and then keeps going: d = 500 in 136 s at a Z-law residual of 3.2e-13.
The original goal for this size was "500..1000 within an hour".

At d = 500 the 11.4 GB peak is the TENSOR, not the solve -- `build_sphere` holds
the sphere, its orthogonal scramble and the nondeg tensor, three 1 GB arrays,
and `_qdn_pair_tensor` `permutedims` the full tensor once per lift axis.  That
is what puts d = 1000 (8 GB per copy) out of reach of a 12 GB process, and it
is a memory problem in the HARNESS and the sketch pass, no longer a convergence
problem in the solver.

Note what the ratio column does at d = 300: the whitened run uses MORE applies
(77482) than the failed plain run (66604).  That is not a regression, it is the
plain count being a cap rather than a convergence cost -- which is exactly why
this report leads with verdicts and not with ratios.

### Eigensolver choice on the whitened operator

The hypothesis was that with the spectrum in `[0, Σ_a c_a]` LOBPCG might win
outright.  It does not.  `:CGSolver` is LOBPCG on `AᵗA`, unpreconditioned, with
block size `nv + max(8, nv÷2)`:

| case | solver | whiten | applies | seconds | nullity (oracle 3) | verdict |
|---|---|---|---|---|---|---|
| sphere v3 d = 30 | ARPACK | 0 | 53632 | 2.5 | 0 | cap |
| sphere v3 d = 30 | ARPACK | 1 | 23732 | 2.0 | 3 | ok |
| sphere v3 d = 30 | LOBPCG | 0 | 168034 | 9.9 | **2** | wrong, uncertified |
| sphere v3 d = 30 | LOBPCG | 1 | 53124 | 3.0 | 3 | ok |
| sphere v3 d = 100 | LOBPCG | 0 | 24040 | 4.6 | **0** | wrong, silent |
| sphere v3 d = 100 | LOBPCG | 1 | 65348 | 14.1 | **0** | wrong, uncertified |
| sphere v3 d = 150 | LOBPCG | 0 | 24040 | 7.6 | **0** | wrong, silent |
| sphere v3 d = 200 | LOBPCG | 0 | 24040 | 29.1 | **0** | wrong, silent |
| sphere v3 d = 200 | LOBPCG | 1 | 72040 | 122.5 | **0** | wrong, uncertified |

Two things to keep.  LOBPCG does benefit from the whitening where it works at
all (168034 → 53124 applies at d = 30, and it becomes CORRECT there), so the
gain is a property of the operator and not of ARPACK's internals.  But LOBPCG's
failure mode is to return the wrong nullity while reporting success -- `nullity
0` at d ≥ 100 in both settings -- and it is 2-4x more applies than ARPACK when
it does answer.  **ARPACK-first stays the right default on this branch**, and
the `nullity 0` rows are also a reminder of what `solve_nullspace`'s own
docstring warns: "a non-converging iterative solver reports 'no solutions'
rather than 'I failed'".

## Degenerate modes: a correctness fix, not only a conditioning one

A smooth or separable tensor has numerically rank-deficient mode unfoldings
(`cond(M_a) = 3.5e16` on the smooth 20x20x10x3 box above).  There the QR has no
inverse, so the whitening takes an SVD, truncates to the range at the standard
`min(size)·eps` cut, and writes the dropped directions down as what they are:
the trivial derivations `X_a ×_a Γ = 0`, checked against Γ itself rather than
trusted.

That also fixes an undercount in the unwhitened path.  It sees those directions
only as `Y_a` with columns in `ker(M_a)`, i.e. `X_a = Y_a W_aᵗ`, and its
rank-deficient lift returns a minimum-norm `Z_a` with no kernel component.  On
a 12³ tensor with a mode-1 rank of 10:

| | reported nullity | true |
|---|---|---|
| plain | 16 (an arbitrary, `W`-dependent slice) | 26 |
| whitened | 26 (2 scalars + the whole 2·12 trivial space) | 26 |

and the whitened answer CONTAINS the plain one to 4.6e-15.  Degenerate modes
are meant to be removed upstream by `nondeg`; this is the safety net.

## Correctness

`bench/jl test/runtests.jl` with `whiten = true` as the DEFAULT: exit 0, no
failures, 13213 assertions.  New in `test/TestQuickDerN.jl` section 8 (135
assertions):

| testset | asserts | what it pins |
|---|---|---|
| the restricted Gram has Kronecker diagonal blocks | 40 | the identity to 1e-12 on 4 chisel/valence combinations, `‖A_w‖ ≤ sqrt(Σ c_a)`, `M_a·un == Q_a`, and `unfold(fold(Q_aᵗ)) == Q_aᵗ` |
| same nullity and same span as unwhitened | 34 | principal-angle residual both ways < 1e-8 on spheres v3/v4, random dense v3/v4, a centroid chisel, and a video-shaped 20x20x10x3 |
| degenerate mode is counted exactly | 52 | nullity `2 + 2d` at d = 8, 12; whitened ⊇ unwhitened; every column Z-law verified |
| matrix-free branch agrees and counts its applies | 9 | forced matrix-free, both settings land nullity 3 and the same span, and `QDN_APPLY_COUNT` counts |

## What did not fit this machine

`bench/jl bench/WhitenedRestriction.jl estimate` prints sizes and an RSS
estimate before anything is allocated.  Against a 12 GB per-process kill line,
and remembering that `build_sphere` holds the sphere, its scramble and the
nondeg tensor at once (≈3 copies):

| case | tensor estimate | verdict |
|---|---|---|
| sphere v3 d = 500 | 2.8 GB | runs |
| sphere v3 d = 1000 | 22.4 GB | over budget, not attempted |
| sphere v4 d = 100 Float64 | 2.2 GB | runs |
| sphere v4 d = 150 Float64 | 11.3 GB | over budget, not attempted |
| sphere v4 d = 200 Float64 | 35.8 GB | over budget, not attempted |
| sphere v4 d = 150 Float32 | 5.7 GB | would run, exploratory only |
| sphere v4 d = 200 Float32 | 17.9 GB | over budget, not attempted |

So the valence-4 sweep the brief asked for (d = 100, 150, 200) is capped at
d = 100 in Float64 by the TENSOR, not by the solver: a `d^4` Float64 array is
1.6e9 entries at d = 200.
