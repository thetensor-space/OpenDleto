# Orthogonally hidden sphere: diagnosis and command-line experiment

> Historical diagnosis and measurements from the original notebook and solver
> prototype. The revised [SphereLab](../../../labs/SphereLab.ipynb) now exposes
> sphere construction, orthogonal hiding, noise, and recovery directly. Its
> solver is available through the public
> [`:SymmetricGram` API](../../../docs/SymmetricGram.md); use a finite positive
> convergence tolerance (for example `tol=1e-6`). Older `tol=Inf` commands below
> describe the prototype and are not accepted by the core method.

The reference is `docs/null_patterns.pdf`, section 8.4 (page 18): solve
`Der(C, Γ) ∩ Ω` with symmetric operators, whose eigendecompositions give
orthogonal changes of basis. This is an exact statement; a finite-width shell
or additive noise changes the near-null problem.

## Revised notebook validation (2026-09-14)

Executed every default code cell of the revised notebook in order with Julia
1.12.3 and four BLAS threads, and inspected the original/hidden/recovered plots.
The optional noise sweep remains off by default. This is the public
`stratify(...; method=:SymmetricGram, nd=3, tol=1e-6, seed=50)` route, with
bounded amplitudes, dimension 50, and fixed hiding/noise draws.

| Input | Relative recovery error | Energy on clean support | Solve wall time |
|---|---:|---:|---:|
| Exact | 2.705e-14 | 100% | 31.67 s |
| Relative noise 0.001 | 9.852e-4 | 99.999904% | 15.18 s |

Both signed-permutation checks and transformation-identity assertions passed.
The first solve includes compilation, and these are single-run timings. The
plots use known hiding matrices only to align coordinates for evaluation.
The existing local `DletoPlotsExt` method-overwrite precompilation warning
appeared; Julia fell back to loading it and all notebook cells completed.

## What differed in the original SphereLab

- `stratify(Γ)` defaults to **universal operators**. Use
  `stratify(SymmetricOps(Γ), UniversalChisel(3), Γ; ...)` explicitly.
- The literal `d=50, r=48, cutoff=1.5` integer shell has **48 occupied entries**
  and only **19 active slices per axis**. At the notebook's `r=50`, there are
  actually 52 entries per axis, 171 occupied entries, and 41 active slices.
  `nondeg` can remove unused slices; it cannot supply missing surface samples.
- The shell includes squared-radius defects of -1, 0, or +1. A shell cutoff
  in squared coordinate units is not a solver-relative residual tolerance.
- The final notebook plotting cell refers to `𝕊_den_sym` while the immediately
  preceding solve assigns `𝕊_strat_sym`; it also plots earlier random draws.
- The first hand-built sphere cell subtracts the center inside `f` and again
  in `f(x-a,y-b,z-c)`. That shifts the intended center twice.

There is a stronger identifiability obstruction than the point count alone.
Exact rational elimination of the diagonal equations `u_i+v_j+w_k=0` gives
**18 independent diagonal derivations** for the 50-cube shell after removing
unused slices, and **15** for the notebook's 52-cube shell. All these diagonal
operators remain symmetric under orthogonal scrambling. Thus restricting to
symmetric operators does not isolate a single sphere direction here: a random
linear combination can produce many different coordinate orderings. These
counts are computed independently by `python3 bench/SymmetricSphereSampling.py`
and recorded in `sampling.csv`. They do not depend on the solver tolerance.

The independent demonstration uses an exact densor sample by default:
`x_i = sqrt(i/(d-1))` for zero-based `i`, and support `i+j+k=d-1`.
It has 1,275 populated entries at d=50, all on `x²+y²+z²=1`.
Its array-index support is a plane; its physical-coordinate support is a
sphere octant. This sampling change is explicit, not a claimed recovery of
the under-sampled integer shell. The `lattice` option retains that failure case.

## Solver support and tolerance semantics

| Route | Symmetric operators | Relevant limitation |
|---|---|---|
| QuickDer | Yes, intersects universal lifted solutions with Ω | Approximate universal solutions followed by intersection need not give the smallest approximate symmetric solutions. |
| FastDer3Valent / QuickDer3 | Yes, same intersection approach | Three engaged axes, valence 3. |
| QuickSylver | No | Requires UniversalOp and exactly two engaged axes; unsuitable for this three-axis universal chisel. |
| SylverLining | Yes, solves directly in Ω | May be substantially slower because it applies the full operator. |
| Auto | QuickDer when applicable, otherwise SylverLining | QuickDerDeclined triggers a potentially expensive fallback. |

Orthogonal scrambling has no noise-tolerance parameter: it uses computed
orthogonal matrices. Use Float64 and verify orthogonality and the transformation
identity. Do not replace it with a generic invertible scramble: conjugation then
need not preserve symmetry.

A universal three-axis chisel always has two scalar derivations. A successful
exact densor sphere typically adds one non-scalar direction. Scalar-only output
is not recovery, even though the solver has correctly found a nonempty kernel.
QuickDer floors its working tolerance at sqrt(eps(Float64)), about 1.49e-8.
SylverLining accounts for its squared operator when applying tolerance.
Automatic spectral-gap selection can keep only exact scalar derivations when
noise lifts the geometric direction away from zero. For a known one-surface
model, `nd=3, tol=Inf` on the **direct symmetric SylverLining route** requests three
smallest directions, including the two scalars. This is a model-order assumption,
not a certification of an exact nullspace. A finite SylverLining tolerance still
limits the retained directions even with positive `nd`; `Inf` removes that
ceiling. QuickDer's `nd` is imposed before
intersection and is not equivalent for constrained operators.

The experiment checks recovery against the known hidden input as well as the
full derivation residual. A tiny residual alone does not establish recovery:
scalar operators have zero residual on every tensor.

## Running the experiment

From the repository root:

```sh
julia --project=. --threads=4 labs/SymmetricSphereDemo.jl \
  --dim=50 --seed=50 --noise=0 --tol=1e-6 --method=QuickDer \
  --out=/tmp/symmetric-sphere/results.csv
```

For the comparison sweep (single Julia process, four BLAS threads, a 15-minute
wall-time limit, CSV checkpoints and an audible/desktop completion notice):

```sh
python3 bench/SymmetricSphereSweep.py
```

The wrapper owns the process and its timeout; no agent is needed to monitor it.
`current-case.txt`, `comparison.log`, and `completion.txt` live alongside the CSV.
Override `SPHERE_TIMEOUT_SECONDS` if needed. The first small run pays compilation
cost and is labelled warmup. Reported `seconds` measures the solve, while the log
also gives whole-case elapsed time. The script exports original, scrambled, and
recovered point clouds plus a three-panel SVG. Recovered physical coordinates
are aligned using the known transformation chain; this is an evaluation oracle,
not input to the solver. Thresholding in the pictures is for display only; the
support/reconstruction metrics use the full tensor.

## Measured precision failure and tuned exact recovery

Float64 tensors, 50³, four BLAS threads, deterministic independent streams for
support, noise, orthogonal scramble, and QuickDer. These are single-run wall
times, not statistically controlled speed benchmarks; some experiment jobs
overlapped. Warmup/compilation is excluded from the rows below.

| Gaussian seed | Default mixed Gram: returned directions | Mixed reconstruction error | Full Gram: returned directions | Full reconstruction error | Mixed / full solve seconds |
|---|---:|---:|---:|---:|---:|
| 17 | 0 | no recovery | 3 | 9.24e-14 | 4.95 / 8.58 |
| 50 | 3 | 9.87e-4 | 3 | 3.26e-11 | 14.74 / 8.94 |
| 91 | 3 | 3.68e-6 | 3 | 4.04e-14 | 13.46 / 6.57 |

The narrowed Gram can find a plausible universal subspace yet lose the
symmetric intersection. Its verification precedes the final Ω projection;
therefore the demo independently measures the **returned symmetric** operators
against the full rectangular chisel map. On seed 50, the mixed result has a
full residual 4.64e-5 despite requesting tol=1e-6. Full Gram reduces it to 9.25e-12.
This is not evidence that every mixed-precision problem fails, but it disproves
that the optimization is harmless on every small symmetric sphere.

For full Gram, tol=1e-8, 1e-6, and 1e-4 give the same seed-50 reconstruction
error (3.26e-11). For mixed Gram, tightening to 1e-8 returns zero directions;
loosening to 1e-4 increases error to 7.74e-2. Precision comes before tolerance.

Bounded random support amplitudes, with magnitude in [0.5,1.5), improve
conditioning without changing the support. Full Gram then gives reconstruction
errors 3.73e-14, 3.76e-14, and 3.10e-14 on seeds 17, 50, and 91, with residuals
below 3.4e-15 and solve times 5.45–8.46 seconds. This is the recommended exact
illustration:

```sh
julia --project=. --threads=4 labs/SymmetricSphereDemo.jl \
  --dim=50 --seed=50 --amplitudes=bounded --noise=0 \
  --gram-precision=full --tol=1e-6 \
  --out=/tmp/sphere-exact.csv
```

`--gram-precision=full` is the demo default. It scopes and restores the existing
`QDN_GRAM_MIXED_PRECISION` setting; production solver defaults were not changed.
`--dense-solver=svd` forces the restricted dense SVD for comparison.
`--solver` alone controls QuickDer's matrix-free solver, not its dense branch.
The scoped global settings require serial calls within a Julia process.

Raw evidence: `results.csv` / `sweep.log` for the original mixed baseline,
`full-precision.csv` / `full-precision.log` for the full-precision run.
The older baseline labels zero returned directions `scalar-only`; read its
`nullity` column. The final demo distinguishes `no-directions`. The baseline
also records a prototype lattice-wrapper error, fixed in the final demo;
it is not evidence of a solver failure on that row.

FastDer3Valent recovered d=10 in the preliminary check, but the 50³ run was
stopped after at least 111 seconds without a result (`slow-method-limit.txt`).
That is a lower bound, not a completed runtime. The sweep makes this slow
comparison opt-in with `SPHERE_SLOW=1`.

## Direct symmetric approximation for noisy data

On the full-precision QuickDer path, relative Gaussian noise of 1e-8, 1e-6,
and 1e-4 caused automatic selection to keep only two directions, for both
tol=1e-6 and tol=1e-3. The exact-kernel task has changed: requesting a larger
numerical ceiling does not force the gap classifier to retain a geometric mode.

`--method=SymmetricGram --nd=3` is a separate, direct symmetric calculation
added to this demonstration for the known one-surface model. It assembles the
symmetric normal matrix from mode covariances and pairwise tensor contractions,
then computes its three smallest modes by oversampled shifted inverse iteration.
The final Rayleigh–Ritz step uses the **unsquared** existing rectangular chisel
map and an SVD, rather than measuring residuals only on the normal matrix.
`--symmetric-solver=eigen` retains the dense eigenvalue calculation as an oracle. It uses Frobenius-orthonormal
symmetric coordinates (off-diagonal units divided by sqrt(2)), so the least
squares objective is invariant under the orthogonal hiding transformations.
It converts the result back to Dleto coordinates and uses the same `stratify`
and full-residual checks as the other methods. It does not use original sphere
coordinates, support, or hidden transforms during the solve.

This avoids forming the 125,000 × 3,825 rectangular chisel matrix at d=50,
and avoids constructing its normal matrix one full-map column at a time.
The stored normal matrix is 3,825 × 3,825 (about 117 MB). This is a dense,
three-way, equal-dimension prototype intended for these modest sizes; its
memory grows as O(d⁴). Normal equations still square the condition number.
It does not replace the general Dleto solvers.

The positive `nd` is a model-order assumption, not an exact-nullspace verdict;
`tol` does not choose the count for this prototype. Two of the three directions
are scalar. Selecting too few directions or applying the rank-three assumption
to the under-sampled lattice is not a justified reconstruction method.
The weighted normal matrix and eigenpairs agree with the existing rectangular
`sylvesterLM` at d=3 and d=4: **20/20 checks passed**, tolerance 1e-12.

```sh
julia --compile=min --project=. bench/TestSymmetricSphereGram.jl
```

The first 50³ direct comparison took **91.67 s** with the dense eigenvalue
oracle and **11.08 s** with inverse iteration (about **8.3× faster**).
Reconstruction error improved from 2.88e-13 to 1.97e-14. The inverse run spent
0.68 s assembling the normal matrix, 4.64 s on Cholesky, 1.62 s on subspace
iteration, and 3.93 s on the unsquared Ritz step. This locates the original
bottleneck in the eigensolve, not in symmetric-matrix support. These are warm,
single-run timings; use the logged stage times rather than extrapolating them
to larger tensors.

The inverse mode also recovers the noisy sphere (Gaussian support, seed 50):

| Relative additive noise | Reconstruction error | Energy on recovered support | Solve seconds |
|---:|---:|---:|---:|
| 0 | 1.97e-14 | 100% | 11.08 |
| 1e-6 | 9.84e-7 | >99.9999999999% | 9.77 |
| 1e-3 (0.1%) | 9.84e-4 | 99.999904% | 12.48 |
| 1e-2 (1%) | 9.91e-3 | 99.990267% | 12.29 |

At 1% input noise, recovery error remains approximately the noise level.
The demonstration does not remove the additive noise; it recovers the basis
in which the noiseless signal has its surface support. The errors and support
fractions are measured on all entries, including those hidden by plot cutoffs.

Recommended noisy illustration:

```sh
julia --project=. --threads=4 labs/SymmetricSphereDemo.jl \
  --dim=50 --seed=50 --amplitudes=bounded --noise=0.001 \
  --method=SymmetricGram --symmetric-solver=inverse --nd=3 --tol=Inf \
  --out=/tmp/sphere-noisy.csv
```

Use `--help` for all options. The core demo writes SVG and point-cloud CSV
without requiring a plotting backend. To produce the three-dimensional PNGs
shown with this report, run the optional Matplotlib helper:

```sh
python3 bench/SymmetricSpherePlot.py /path/to/demo_points.csv
```

`exact-recovery.png` shows the bounded-amplitude exact example. `noisy-recovery.png`
shows the 0.1%-noise direct symmetric example. The physical coordinates of the
recovered points use the explicitly stated transformation-chain alignment.

The independent runs preserve the user's existing notebook and plotting edits;
all implementation here is in new demo/benchmark files.

The bounded-amplitude, 0.1%-noise direct test passed on all three seeds:
reconstruction errors 9.8421e-4, 9.8426e-4, and 9.8471e-4 (seeds 17, 50, 91),
with valid permutations and more than 99.99990% support energy. Solve times
were 9.15–13.75 seconds. The full direct validation finished successfully;
`inverse.csv`, `inverse.log`, and `inverse-completion.txt` contain its evidence.

The literal lattice comparison reduced to 19³ and failed the rank-three
recovery model: reconstruction error 0.9424, support energy 11.19%, and no
valid permutation. Its residual was only 2.20e-4, underscoring why residual
size alone is insufficient. This row uses an intentionally inappropriate
three-mode assumption on a shell already known to have at least 18 diagonal
derivations; it is not a claim that this prototype resolves that larger
nullspace. Recovering the intended sphere ordering requires better sampling
or additional geometric constraints.
