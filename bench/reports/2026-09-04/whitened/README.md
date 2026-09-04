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
| sphere v4 d = 100 (100^4 Float64) | 4 | 65858 / **cap, nullity 0** | 13188 / nullity 4, resid 3.7e-13, 7.2 s, 11.7 GB peak | 5.0x |
| video 200x200x100x3 Float32 | 3 | 56980 / **cap, nullity 0** | 11182 / nullity 3, resid 1.7e-5, 3.2 s | 5.1x |
| degenerate 40^3, mode-1 rank 38 | 82 | 59640 / **nullity 26 of 82** | 4024 / nullity 82, resid 8.4e-16 | 14.8x |

The degenerate row is the sharpest confirmation of the mechanism: 26 is exactly
`2 + 2·r_1` with `r_1 = 12`, the `W`-visible slice the analysis predicts the
unwhitened branch can see, and 82 is exactly `2 + 2·d`, the whole trivial space
plus the two scalars.  The Float32 video row is `uncertified` (the verdict's
gap test cannot clear a Float32 precision floor at this nullity) but lands the
oracle nullity at the Float32 residual bound.  At valence 4, d = 100 is the
ceiling on this machine and it is the TENSOR that sets it: 11.7 GB peak against
a 12 GB kill line, for a 100^4 Float64 array that `build_sphere` holds three
copies of.

### The frontier

Whitened, ARPACK, Float64, matrix-free, valence 3, scrambled sphere:

| d | restricted system | applies | seconds | peak RSS | nullity | residual | plain at the same size |
|---|---|---|---|---|---|---|---|
| 200 | 17576 x 15600 | 38262 | 18.4 | 2.1 GB | 3 | 3.3e-10 | 65182 applies, **cap** |
| 300 | 29791 x 27900 | 77482 | 105.0 | 3.8 GB | 3 | 1.8e-10 | 66604 applies, **cap** |
| 500 | 64000 x 60000 | 51116 | 136.1 | 11.4 GB | 3 | 3.2e-13 | 66902 applies, **cap** |

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

**Part 2 below measures that claim and half of it is wrong.**  The peak is
indeed the harness, but the dominant term is `nondeg`'s per-axis SVD, not the
three copies; `_qdn_pair_tensor`'s transpose is 80 MB at this size (it is 8 GB
at d = 1000); and the largest single item on the video shapes is the
BENCHMARK's own Z-law check.

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

# Part 2 (afternoon): the memory wall

The morning above moved the frontier to d = 500 and left a MEMORY wall.  This
half measures that wall, removes it, and takes valence 3 to d = 1000.  Raw
numbers in the same `whitened.csv`; the stage-by-stage memory comes from
`bench/MemoryProfile.jl`.

## Memory: where the frontier's bytes actually were

After the whitening the matrix-free branch converges and the wall is memory.
The morning's numbers said d = 500 peaked at 11.4 GB and projected ~22 GB at
d = 1000, and named two suspects from reading the code: `build_sphere` holding
three copies of the tensor, and `_qdn_pair_tensor` doing a full `permutedims`
of it once per lift axis.  `bench/MemoryProfile.jl` measures instead of
guessing -- a stage boundary between every step, and three numbers per stage,
because they disagree and each is needed:

* **alloc** `Base.gc_bytes()` over the stage: every allocation, survivor or
  not.  CHURN.
* **live** `Base.gc_live_bytes()` at the end of it: what is still reachable.
  RETENTION.
* **peak** `Sys.maxrss()`, the process high-water mark, monotone across
  stages.  The stage where it first reaches its final value is the one that
  set the peak, and the peak is the kill-line number.

### Before: d = 500 valence 3, one copy = 0.93 GB

| stage | alloc GB | live GB | peak GB |
|---|---|---|---|
| `sphere_octant` | 1.878 | 1.928 | 2.799 |
| nonzero count | 1.863 | 2.859 | 3.731 |
| `randomize_tensor` | 2.828 | 3.796 | 5.623 |
| `nondeg` | 25.214 | 6.605 | **10.299** |
| convert + ops | 0.000 | 6.605 | 10.299 |
| (after a full GC) | 0.000 | 4.736 | 10.299 |
| sketches + whitening + solve (all of QuickDer) | 243.1 | | **12.141** |

**The peak is the harness, and the entire solve adds 1 GB on top of it.**  Six
copies live at the worst moment for a pipeline that needs one.  Where they go:
`Array(S, fr...)` for the nonzero count is a second full copy, `ITensor(A,
frames...)` copies the array it is handed, `randomize_tensor` contracts three
`ITensor`s in sequence, and `nondeg` -- a full `d^{n-1} x d` SVD per axis,
which forms the unfolding AND its left factor -- churns 25 GB and adds 4.7 GB
of peak by itself.

Both of the named suspects were half right.  `build_sphere` does hold copies,
but the dominant one is `nondeg`'s SVD, not the three tensors.  And
`_qdn_pair_tensor`'s transpose is 80 MB at d = 500 -- invisible here.  It is
real, and it is fatal, only at d = 1000, where the same transpose is 8 GB.

### The fixes

| what | where | why |
|---|---|---|
| no `permutedims` of the input, ever | `_qdn_ttm` | axis 1 and axis `N` ARE reshapes of the tensor, so they are single GEMMs; a middle axis runs `back` GEMMs on contiguous `(front, d, back)` slices, reading the tensor once and writing only the small output |
| `_qdn_ttm!(out, G, M, a)` | new | a chain of mode products in two buffers instead of a fresh `d^n` array per step |
| `_qdn_ttm_square!(G, M, a)` | new | the same chain in ONE buffer when the matrices are square, which the scramble and the `nondeg` change of basis both are: a block of slices into a 64 MB buffer and copy back |
| `_tsqr_axis_basis` | `bench/SphereHarness.jl` | the `nondeg` axis basis by tall-skinny QR: `R` is `d x d`, the working set is one 64 MB block rather than three copies of the tensor, `svd(R)` gives `V` and the singular values at full precision, and it is `2 m d²` flops against `gesdd`'s `8 m d²` |
| `itensor`, not `ITensor` | `build_sphere_lean` | the capitalised constructor copies the array it is handed |
| nonzero count off the array | `build_sphere` | `ITensors.array` in storage order is a view |
| `keep_S = false` | `build_sphere` | `reconstruction` needs the original tensor and no frontier run scores |
| the trivial space FACTORED, capped at 256 MB | `_qdn_trivial_ders`, `QDN_TRIVIAL_FACTORED` | `(d - rank)·d` operator tuples of `n` dense `d x d` matrices is 3 GB at d = 500 with one degenerate mode, to describe a `d x (d - rank)` matrix |

Above `SPHERE_LEAN_BYTES` (256 MB per copy) `build_sphere` takes the lean
path.  It is the same input family and NOT the same array -- the SVD's sign
ambiguity, and `nondeg` applies its basis transposed (`Array(V, r, a)` is
`(link) x (a)` and it is handed to `ITensor(_, a, a_nondeg)`), where this path
applies the `V` that actually diagonalises the unfolding.  Both are per-axis
orthogonal, so nullity, verdict and the ORDER of the Z-law residual carry over
while the iteration count and the residual's digits do not.  `-lean` in a case
name marks those rows.  `bench/jl bench/MemoryProfile.jl compare <d>` checks
the TSQR singular values against `ITensors.svd` (agreeing to 1.2e-14) and the
two builds' nullity side by side.

## Honesty: three gaps closed and one diagnosis corrected

### A failed iterative solve now says so

`solve_nullspace`'s bracket rule cannot tell "found the whole null space and
then some" from "converged to nothing at all" -- both look like values above
the cut -- so a non-converging iterative solver reported an empty null space
rather than a failure, silently.  Measured on the whitened restricted map,
matrix-free, valence 3: block Lanczos converges NOTHING at d = 300 and d = 500,
returns its sixteen Ritz values stalled at 1.1e-9 relative, and the gap test
reads that spectrum -- correctly, on its own terms -- as `nullity 0,
certified`.  Since nullity 0 is a legitimate answer it cannot be escalated
away; the only defence is for the solver to say whether it converged.

So it does: `ArpackSolver` from ARPACK's `nconv`, `CGSolver` from LOBPCG's own
flag, `KrylovSolver` from `info.converged`.  `NullVerdict` carries it as
`status` (`:ok` / `:unconverged` / `:capped`), deliberately separate from
`certified`, which is and stays a statement about the spectrum alone -- the
whole point is that a stalled solve produces a spectrum with a clean gap in it.
`_qdn_solve_and_lift` declines a non-`:ok` solve that returned nothing, so
`AutoDerMethod` falls back to SylverLining instead of reporting "no
derivations"; a nonzero count from a non-converged solve is kept and judged on
its own Z-law residual, which is the stronger test.

### The d = 300 apply anomaly is not the nv escalation

d = 300 uses 76810 applies where d = 500 uses 51116, and the standing
suspicion was the escalation loop doubling one extra time there.  It does
double one extra time -- requests `[16, 32]` at d = 200 and d = 500 against
`[16, 32, 64]` at d = 300 -- but that is a symptom.  The trace (one `@debug`
line per request, and `bench/WhitenedRestriction.jl` reports the sequence on
every row) reads

    request 16 -> nullity 10 of 16    request 22 -> nullity 12 of 22
    request 27 -> nullity 13 of 27    request 33 -> nullity 13 of 33

The RESTRICTED system at d = 300 has a 13-dimensional numerical near-null space
against 3 true derivations; the loop chases it correctly and the lift's
consistency filter cuts 13 down to 3.  So the cost is that spurious dimension
and the escalation rule's only job is to get past it in as few solves as it
can: measured both ways, a gentler `k + max(4, k÷4)` step takes FOUR solves and
99170 applies where doubling takes three and 76810.  Doubling stays.  The lever
on d = 300 is the restriction size (its system is 29791 x 27900, 6.8%
overdetermined, against d = 200's 12.7%) or the null threshold -- not `nv`.

### The lift filter cuts on a gap; and the Float32 undercount is elsewhere

The filter took a null space of the lift-residual matrix at a hard relative
cutoff, `atol = max(tol, sqrt(eps(T)))`.  That is the level a lift residual
BOTTOMS OUT at -- the accuracy of the triangular solve that produced `Z`,
measured at 1.3 to 2.9 times `sqrt(eps(Float32))` for genuine directions -- so
the cutoff sits inside the population it is meant to keep, and in Float32 it
cuts into it (at d = 140 the spectrum reaches 1.1e-3 against a cutoff of
3.5e-4).  The rule now uses the same constant as a FLOOR and cuts at the
largest consecutive ratio above it, `gap_verdict`'s rule, with a ceiling of
`32*sqrt(eps(T))` bounding eligibility.  The floor has to be `sqrt(eps)` and
not `100*eps`: with the tighter floor the genuine cluster spans decades above
it, its internal ratios clear `gap_ratio`, and the cut lands inside the
cluster -- measured, that undercounts the raw sphere at 11 of 13 in Float64.

It is not, however, where the Float32 undercount lives.  At d = 48 Float32,
matrix-free, fresh process:

    restricted solve   nullity 13 (uncertified)
    lift filter        cut 13 of 13   -- under BOTH the old rule and the new
    restrict_to_ops    13 -> 2

The filter passes everything it is given; the loss is in
`_fastder_restrict_to_ops`, cutting 13 universal derivations down to the
SymmetricOp space.  Fresh process, one call each: old rule 2, new rule 2.

**And any before/after table on those sizes measures noise.**  ARPACK's start
vector comes from a seed SAVED inside the library across calls, so the same
input, the same `Random.seed!` and the same knobs give different restricted
null spaces on repeated calls in one process.  The control -- the same setting
three times -- reads

| case | same setting, x3 (old rule) | same setting, x3 (new rule) |
|---|---|---|
| scrambled v3 d = 48 Float32 | 2, 3, 3 | 2, 2, 3 |
| scrambled v3 d = 64 Float32 | 2, 3, 3 | 3, 3, 2 |
| scrambled v3 d = 48, 64 Float64 | 3, 3, 3 | 3, 3, 3 |
| raw sphere d = 24, 40, both eltypes | 13, 13, 13 | 13, 13, 13 |

Every Float64 and every raw-sphere case is stable; the Float32 matrix-free ones
are not.  Measure them one data point per PROCESS, or with a deterministic
solver.  On the dense branch with `SVDSolver` -- no randomised subspace and no
ARPACK saved state -- both rules give the oracle at d = 48, 64, 100 in both
eltypes and 13 for the raw sphere at d = 24, 40, which is what the new test
pins.

## The frontier, part 2

Whitened, ARPACK, Float64 unless said otherwise, matrix-free, after the memory
work.  `-lean` rows were built by `build_sphere_lean` (the same input family up
to a per-axis orthogonal change of basis, so their applies and residual digits
are not comparable with a classic row at the same d; their nullity and verdict
are).  `nv` is `solve_nullspace`'s request sequence.

| case | restricted system | r | applies | nv | seconds | peak RSS | nullity | residual |
|---|---|---|---|---|---|---|---|---|
| v3 d = 200 | 17576 x 15600 | 26 | 38262 | 16, 32 | 20.6 | 2.06 GB | 3 | 3.25e-10 |
| v3 d = 300 | 29791 x 27900 | 31 | 76810 | 16, 32, 64 | 118.2 | 4.49 GB | 3 | 1.88e-11 |
| **v3 d = 700** -lean | 103823 x 98700 | 47 | 103982 | 16, 32, 64 | 546.0 | 17.04 GB | 3 | 7.45e-13 |
| v3 d = 500 -lean | 64000 x 60000 | 40 | 49556 | 16, 32 | 204.4 | **3.87 GB** | 3 | 2.38e-11 |
| **v3 d = 1000** -lean | 178752 x 168000 | 57,56,56 | 56388 | 16, 32 | 552.6 | **13.70 GB** | 3 | 7.19e-13 |
| **v4 d = 150** -lean | 10000 x 6000 | 10 | 16024 | 16, 32 | 31.0 | 7.29 GB | 4 | 1.63e-13 |
| **v4 d = 200** -lean | 14641 x 8800 | 11 | 14430 | 16, 32 | 181.4 | 17.35 GB | 4 | 1.33e-11 |
| video 640x480x90x3 F32 | 7920 x 25680 | 20,20,11,3 | 19482 | -- | 26.3 | 2.03 GB | 3 | 9.7e-6 |
| video 640x480x300x3 F32 | 13200 x 30000 | 20,20,11,3 | 26554 | -- | 43.9 | 4.06 GB | 3 | 1.5e-5 |

**d = 1000 at valence 3 and d = 200 at valence 4 are new**, against the
morning's d = 500 and d = 100 -- and the morning's own estimate table put
valence 4 at d = 200 Float64 at 35.8 GB and "over budget, not attempted".  d = 1000 answers in 9.2 minutes at a Z-law
residual of 7.2e-13 and a peak of 13.7 GB -- the goal set for this size was
"500..1000 within an hour", and the morning's own estimate for the BUILD alone
was 22.4 GB.  Note that d = 1000 costs FEWER applies than d = 700 (56388
against 103982) and less peak (13.7 GB against 17.0): both differences are the
preallocated apply and the rewritten Z-law check, which landed between the two
rows, plus d = 1000's request sequence stopping at [16, 32] where d = 700's
went to 64.  The valence-4 ceiling moved for the same reason it was there:
`build_sphere` held six copies of a 3.8 GB tensor and then the Z-law check
asked for `2(n+1)` more.

At valence 4 the two rows show where the cost now sits: d = 200's solve is
9.6 s of its 181 s, so what is left is the BUILD (a 12.8 GB tensor, the three
TSQR passes over it) and the Z-law check.  The 17.35 GB peak is one tensor plus
the GC's headroom, which is the floor for a `d^4` Float64 array on this
machine.

The requirement this half was set was that d = 500 drop "well below" the
morning's 11.4 GB: it is 3.87 GB, a 2.9x cut, with the same nullity, the same
verdict and a residual of the same order (2.4e-11 against 3.2e-13 -- the lean
build is a different member of the family, so the digits differ).

d = 200 reproduces the morning's row to the digit (38262 applies, 3.25e-10),
which is the regression check on every change in this half -- the mode-product
rewrite, the in-place kernels and the preallocated matrix-free apply are all
numerically identical, not merely close.  The two video rows are the shape the
user actually cares about: at F = 300 the morning's harness was killed at
13.1 GB and this one peaks at 4.06 GB.

The d = 700 row was taken before the matrix-free apply got its preallocated
scratch and before the Z-law check was rewritten; its 17 GB is mostly GC slack
and one Float64 promotion, not a live set (the same measurement's live set is
under 8 GB).  Read it as the frontier's REACH at the time, not as its cost now.

### d = 1000 at valence 3

The build fits comfortably.  `bench/MemoryProfile.jl sphere 1000 3`, one copy
= 7.45 GB:

| stage | alloc GB | live GB | peak GB |
|---|---|---|---|
| lattice array | 7.517 | 7.570 | 8.335 |
| nonzero count | 0.000 | 7.570 | 8.335 |
| scramble, in place | 0.245 | 7.816 | 8.604 |
| `nondeg` bases (TSQR) | 62.266 | 7.833 | 9.152 |
| `nondeg` apply, in place | 0.132 | 7.965 | 9.152 |
| convert + ops | 0.045 | 8.010 | 9.190 |
| after a full GC | 0.000 | **7.571** | 9.190 |

`live` at the end is 7.571 GB against a 7.451 GB tensor: ONE copy, which is the
floor.  The morning's estimate for the same build was 22.4 GB and the classic
path would in fact have wanted more than 40.

And the whole run fits: 13.70 GB peak, 552.6 s, nullity 3, residual 7.19e-13.
It did not fit on the first three tries, and the three things that closed the
gap are worth naming because they are all the same mistake in different
places -- 25.7 GB with fresh temporaries in the lift's residual, 21.3 GB after
`AZ .-= B`, and 13.7 GB once the matrix-free apply stopped allocating ~13 MB
per apply-pair and the Z-law check stopped promoting a full copy to Float64.
Between them, RSS was tracking the GC's heap target and not the live set.
