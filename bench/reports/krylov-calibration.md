# Krylov-type null solvers on the sphere stratification: resilience, cost, calibration

Date: 2026-09-03.  Branch `signed-sealed-delivered/der-and-derivation-law`.
Input: `bench/SphereHarness.jl`, `build_sphere(d; seed)` (Float64, `SymmetricOp`,
universal chisel).  The operator handed to the null solver is the square
symmetric PSD derivation operator of dimension `N = 3 d(d+1)/2` (1395 at d=30,
2460 at d=40, 4788 at d=56, 24768 at d=128); the derivation space is
3-dimensional.  Metric: nullity must be 3 and `lsq_err` ~1e-13.

All timings were taken with `julia -t 2` on an 8-core machine shared with
other agents.  Wall times are only comparable *within* one table; the
contention-free cost measure is the number of map applications ("matvecs"),
counted with a wrapper around the `LinearMap`.  Scripts and CSVs:
`exp1-grid.jl/.csv`, `exp1-truth.csv`, `exp2-seeds.jl/.csv`,
`exp3-endtoend.jl` -> `exp3-seeds.csv`, `exp3-nv0.csv`, `exp3-scaling.csv`.

## 1. The spectrum being solved (dense ground truth)

`eigen(Symmetric(Matrix(L)))`, seed = d.  The three null eigenvalues sit at
1e-14; the gap ratio `lambda_4 / lambda_max` decides how hard the problem is
for a regular-mode Lanczos on the small end (iterations ~ 1/sqrt(gap)).

| d  | N    | lambda_1..3 | lambda_4 | lambda_5 | lambda_max | lambda_4/lambda_max | dense eigen (s) |
|----|------|-------------|----------|----------|------------|---------------------|-----------------|
| 24 | 900  | ~1e-13      | 2.98e-2  | 1.18e-1  | 102        | 2.9e-4              | 0.6  |
| 32 | 1584 | ~1e-14      | 2.53e-1  | 8.13e-1  | 103        | 2.5e-3              | 2.4  |
| 36 | 1998 | ~1e-14      | 7.88e-2  | 3.40e-1  | 123        | 6.4e-4              | 7.4  |
| 40 | 2460 | ~1e-14      | 4.25e-1  | 5.81e-1  | 164        | 2.6e-3              | 7.7  |
| 48 | 3528 | ~1e-13      | 2.23e-2  | 9.91e-2  | 151        | 1.5e-4              | 19.5 |

The gap is a property of the random support coefficients, not of d: over
seeds 1..5 at d = 32/36/40 it ranged from 4.6e-3 down to **5e-6** (seed 5),
five times the relative threshold `tol = 1e-6`.  Seed 2 has a genuine fourth
near-derivation at `lambda/lambda_max ~ 5e-7`, *under* tol: every solver,
dense included, reports nullity 4 there and the recovery degrades to
`lsq_err ~ 1e-6`.  So do not read seed 2 (or seed 56 at d = 56) as a solver
failure; the benchmark's tolerance is marginal for those inputs, and if the
sphere is the target `tol = 1e-8` would be the safer harness setting.

## 2. Diagnosis of the KrylovSolver nullity-2 miss

**It is the multiple-eigenvalue problem, not the gap, the tolerance, the
`info.converged` check or the retry loop.**  The null space is a 3-fold
eigenvalue 0.  A Krylov space grown from one start vector `x0` contains
exactly one direction of that eigenspace (the projection of `x0` onto it);
the other copies enter only through roundoff and are amplified only by the
restart polynomial.  Whether they have emerged by the time `howmany` Ritz
pairs have converged is luck.  Evidence, `exp1-grid.csv` (seed = d):

| KrylovKit `eigsolve(:SR)`   | d=32 | d=36 | d=40 | what came back for nev=4, kd=10 |
|-----------------------------|------|------|------|-----------------------|
| nev=4 (kd 10/16/64, any tol)| 1-2  | 1-2  | 1-2  | `0, 0.2525, 0.8125, 0.9256`: one null vector, then three *genuine* nonzero eigenvalues, all converged |
| nev=8                       | 2-3  | 2-3  | 2-3  | 20 of 54 configs correct |
| nev=16                      | 3    | 2-3  | 3    | 46 of 54 configs correct |

Because the extra returned values are genuine eigenvalues above threshold,
`solve_nullspace` sees a legitimate bracket and returns nullity 1 or 2 with
no warning.  Arnoldi and Lanczos (`issymmetric=true`) behave identically;
the old extension in fact ran **Arnoldi** on the symmetric map (KrylovKit only
picks Lanczos when told, and a `LinearMap` is not an `AbstractMatrix`), with
`krylovdim = 2 nev = 32` and an *absolute* `tol = 1e-8`.  ARPACK is not immune
either: `nev=4, ncv=20` missed a null vector in 4 of 15 seeded runs
(`exp2-seeds.csv`) and once in the 72-config grid; its reliability at `nev=16`
comes from doing more restarts before declaring convergence.

### Candidate fixes, 5 seeds x d in (32, 36, 40) (`exp2-seeds.csv`)

"correct" = nullity 3 (4 for seed 2, matching the dense truth).

| strategy                          | correct | matvecs mean (range) |
|-----------------------------------|---------|----------------------|
| Arpack nev=16 ncv=128 tol=1e-10   | 15/15   | 489 (437-645)  |
| Arpack nev=16 ncv=33              | 15/15   | 526 (409-684)  |
| Arpack nev=4  ncv=20              | 11/15   | 804 (503-1420) |
| Arpack nev=4  ncv=64              | 14/15   | 811 (531-1312) |
| Arpack nev=4  ncv=128             | 15/15   | 1066 (607-2484) |
| KK single Lanczos nev=4 kd=32     |  0/15   | 248 |
| KK single Lanczos nev=4 kd=128    |  2/15   | 260 |
| KK single Lanczos nev=4 kd=256    | 11/15   | 366 |
| KK single Lanczos nev=16 kd=64    | 11/15   | 385 |
| KK Lanczos + deflation guard nev=2| 15/15   | 813 (613-1146) |
| KK Lanczos + deflation guard nev=4| 15/15   | 805 (633-1192) |
| **KK BlockLanczos block 4, kd=128** | **15/15** | **434 (389-545)** |
| KK BlockLanczos block 4, kd=64    | 15/15   | 485 (401-653) |
| KK BlockLanczos block 2, kd=64    |  3/15   | 346 |
| KK BlockLanczos block 8, kd=128   | 15/15   | 752 (633-857) |

Block Lanczos with a block at least the multiplicity is the textbook fix and
costs no more than ARPACK; the deflation guard (re-solve on `L + c V V'`
until the smallest value is above threshold) is also sound but 1.7x dearer.
Null-vector residuals `|Lv|/|L|` were 5e-16..1e-15 for ARPACK and
1e-15..5e-13 for block Lanczos at relative tol 1e-10; both give
`lsq_err ~1e-13` end to end.

Shift-invert on the matrix-free map was assessed and rejected: the inner
solve is CG on `L + shift I` with condition `|L|/shift`; a shift small
enough to separate 0 from `lambda_4` (gap down to 5e-6) needs ~1/sqrt(gap)
CG steps *per application*, more than the whole `:SM` solve.  This is why
`ShiftInvertSolver` (shift 1e-10 relative, 100 CG steps) found nothing.

## 3. What was changed

`ext/DletoKrylovKitExt.jl` -- `KrylovSolver`:
- `KrylovKit.BlockLanczos` with block = request (`blocksize` kw to override),
  `krylovdim = max(128, 8 blocksize)`, `maxiter = 200`.
- `tol` is now **relative** to a 10-matvec power-iteration estimate of `|L|`
  (default 1e-12); KrylovKit's `tol` is an absolute residual norm.
- Arnoldi only as a fallback for a non-symmetric map; dense `eigen` for
  `N <= max(32, 2(nev+1))`, where a block cannot fit.
- `Dleto.initial_request(::KrylovSolver) = 4`.

`ext/DletoArpackExt.jl` -- `ArpackSolver`:
- Requests `max(nv, 16)` pairs (`min_request` kw), `ncv = 8 nev` (128 by
  default), `tol = 1e-10`, `maxiter = 300`, and returns the `nv` smallest by
  magnitude, so the caller's doubling rule stays sound.  Dense `eigen` for
  `N <= 32` (ARPACK needs `nev < N-1`).

`src/solvers/NullSolvers.jl`:
- `SQUARE_DENSE_LIMIT = 400`; `dense_is_cheap` densifies a *square* map only
  below that side when any matrix-free solver is registered, otherwise the
  old `DENSE_LIMIT = 1000` / 1 GB byte gate applies (unchanged for
  rectangular maps, i.e. `den`).
- `matrix_free_solvers(L)`: square -> `[:ArpackSolver, :KrylovSolver,
  :CGSolver, :LSMRSolver]`; rectangular -> the old `[:LSMRSolver, :CGSolver,
  :ArpackSolver, :KrylovSolver]`.  The zero-argument form is kept.
- `solve(::AutoSolver, L)` squares the map only if the chosen solver
  `wants_square` -- previously it squared unconditionally, so `LSMRSolver`
  (whose whole design is the rectangular map) received `L'L`.  Behaviour
  change for a matrix-free `den`: LSMR now gets the rectangular map, as
  documented.  Verified on a 60x40 rank-37 map: nullity 3, `|Av| = 1.4e-14`.
- `initial_request(solver[, L])` trait, default 16; `solve_nullspace`'s
  `nv0` defaults to it (`AutoSolver` forwards to the solver it will use).

## 4. Request size (`exp3-nv0.csv`, through `solve_nullspace`, tol 1e-6)

Matvecs to return the null space with a bracket, d=32 and 40, seeds d and 5:

| nv0 | Arpack (4 cases)   | KrylovSolver block Lanczos (4 cases) |
|-----|--------------------|--------------------------------------|
| 4   | 563, 659, 572, 665 | 471, 523, 523, 575  |
| 8   | 565, 660, 572, 665 | 831, 831, 943, 887  |
| 16  | 470, 661, 571, 665 | 1711, 1599, 2047, 1839 |
| 32  | 497, 682, 688, 698 | 2399, 2271, 2879, 2879 |

ARPACK's cost is flat in the request (keep 16: it is what makes it
resilient); block Lanczos is linear in the block, hence its initial request
of 4 and reliance on the doubling rule (which costs at most one extra solve).

## 5. End-to-end resilience (`exp3-seeds.csv`, `run_stratify`, tol 1e-6)

30 runs, 5 seeds x d in (32, 36, 40) x {ArpackSolver, KrylovSolver}: 30/30
returned the dense-truth nullity (3; 4 at seed 2), `lsq_err` 3e-14..4e-12
(seed 2: 1e-6..9e-6 for every solver, see section 1).  Arpack wall time
0.5-0.8 / 1.1-1.2 / 1.5-2.1 s at d = 32/36/40; KrylovSolver in that run was
still using block 16 (2.1-2.3 / 3.5-6.1 / 5.0-7.2 s); with the final block-4
default it measured 0.58 s at d=32 and 1.73 s at d=40, on par with Arpack.

## 6. Scaling (`exp3-scaling.csv`, seed = d, `run_stratify`)

| d  | N    | Arpack s / matvecs | Krylov (block 16) s / matvecs | dense SVD s | CG (LOBPCG) s | gap    | peak RSS GB |
|----|------|--------------------|-------------------------------|-------------|---------------|--------|------|
| 24 | 900  | 0.27 / 358  | 0.98 / 1311  | 0.71  | 8.5   | 2.9e-4 | 1.68 |
| 32 | 1584 | 0.75 / 564  | 2.81 / 1727  | 3.34  | 8.9   | 2.5e-3 | 1.68 |
| 40 | 2460 | 1.68 / 571  | 6.30 / 2031  | 10.4  | 32.5  | 2.6e-3 | 1.87 |
| 48 | 3528 | 2.79 / 561  | 9.68 / 1839  | 26.1  | --    | 1.5e-4 | 2.24 |
| 56 | 4788 | 4.20 / 563  | 16.2 / 2271  | 57.0  | --    | --     | 2.43 |

(d = 56, seed 56, has a fourth near-derivation; all three solvers agree on
nullity 4.)  Bytes allocated: Arpack 0.8 -> 10 GB, Krylov(16) 2.4 -> 38 GB,
SVD 1.7 -> 87 GB, CG 23 -> 73 GB over the table.  RSS is dominated by the
tensor machinery, not the solvers; the Krylov bases are 25 MB at d = 128
(128 x 24768 doubles).

Least-squares fits `t = c d^p` (contended, 2 threads):

| solver     | p    | t(64)  | t(96)  | t(128)  |
|------------|------|--------|--------|---------|
| Arpack     | 3.26 | 7 s    | 26 s   | **67 s** |
| Krylov(16) | 3.28 | 26 s   | 99 s   | 255 s (block 4: ~Arpack) |
| dense SVD  | 5.16 | 116 s  | 16 min | 69 min (and a 4.9 GB matrix) |
| CG         | 2.5 (3 points, unreliable) | 81 s | 226 s | 466 s |

Decomposition: Arpack's matvec count is flat (~560 from d = 32 to 56), so the
whole exponent is the map application, 0.76 ms at d = 24 to 7.5 ms at d = 56
(~d^2.8; the contraction is O(d^4) asymptotically, so expect the exponent to
drift up).  At d = 128 that is 45-75 ms per application, 560 applications:
**25-45 s for the null solve, 45-70 s for the full stratification** by the two
extrapolations.  The one-minute target is borderline, and the eigensolver is
no longer the lever: 560 applications is already within a factor ~2 of the
un-restarted Lanczos floor for these gaps.  The lever is the map:
`BlockLanczos` applies the operator to a block of 4 (or 16) vectors per step,
and a `sylvesterLM` variant that accepts a *matrix* of vectors would replace
`bs` BLAS-2 contractions by one BLAS-3 contraction -- typically 2-4x on this
kind of kernel -- which would put d = 128 comfortably under a minute and
would make the block solver, not ARPACK, the fastest.  More BLAS threads help
the same term.

Memory is not a concern at d = 128: tensor 16 MB, Krylov basis 25 MB, no
dense matrix.

## 7. AutoSolver calibration (implemented; see section 3)

- Dense gate: square map -> dense only for `N <= 400` when an eigensolver is
  loaded.  Crossover data: N = 165 SVD 0.02 s vs Arpack 0.05; N = 360, 0.12
  vs 0.06; N = 630, 0.30 vs 0.17; N = 900, 0.71 vs 0.27; N = 2460, 7.1 vs 1.6.
  Rectangular map -> unchanged byte gate (LSMR is ~100x slower than a dense
  SVD that fits).
- Element type: the byte gate keeps `sizeof(eltype)`; the square gate is a
  dimension because both sides (N^3 factorisation vs ~560 O(N^2)-per-step
  applications) scale identically with the element size.  Not done here
  (another agent owns element-type handling): for `Float32` maps the
  extension defaults `tol = 1e-12` (KrylovKit) / `1e-10` (ARPACK) are below
  `eps(Float32)` and should be floored at about `sqrt(eps(RT))`.
- Order for square symmetric maps: Arpack, Krylov, CG, LSMR.  For the
  rectangular densor map: LSMR, CG, Arpack, Krylov (unchanged; `den` intact).
- `initial_request`: 16 (Arpack, others), 4 (KrylovSolver).

Not changed but worth noting: `LSMRSolver` asks for `nv + 8` projections
with one refinement each; on the square map it is 100x Arpack for that
reason, and a smaller candidate count would make it competitive at large d
because its iteration count scales with `gap^-1/4` rather than `gap^-1/2`.

## 8. Tests

`julia -t 2 --project=. -e 'using Pkg; Pkg.test()'` after all edits above:
**"Testing Dleto tests passed"**, every test set Pass = Total (the suite
reaches the solvers through `AutoSolver`, whose small maps stay on the dense
path).  Smoke run of the new paths (`build_sphere`): AutoSolver d=8 dense
0.008 s; AutoSolver d=24 -> Arpack 0.27 s (SVD 0.78 s); KrylovSolver block 4
d=32 0.58 s, d=40 1.73 s; all nullity 3, `lsq_err` <= 6e-12.
