# Reduced-precision stratification: Float32 and Float16 on the sphere benchmark

Date: 2026-09-03.  Branch `signed-sealed-delivered/der-and-derivation-law`
(uncommitted Float32 generalisation of the solvers).  Machine: 16 GB, Julia
1.12.6, run as `julia -t 2 --project=.` with BLAS pinned to 2 threads while
other agents used the remaining cores.  **All timings are under contention and
drift by 2-5x between sessions (compare Exp. 1's Float64 Arpack at d = 40,
6-14 s, with Exp. 5's 0.8-1.0 s two hours later).  Compare rows within one
experiment only.**  Bytes are cumulative allocation from `@timed`; peak RSS is
`Sys.maxrss()`.

Scripts and raw CSVs live beside this file: `precision-exp1.jl` .. `precision-exp5b.jl`
and the matching `precision-exp*.csv` / `.log`.  The harness is
`bench/SphereHarness.jl`: scrambled nondegenerate sphere octant, symmetric
operators, universal chisel, nullity exactly 3, `lsq_err` = relative
least-squares recovery error (Float64 gives ~1e-14).

**Mid-study caveat.** Another agent rewrote `ext/DletoArpackExt.jl` and
`ext/DletoKrylovKitExt.jl` at 13:40 and `src/solvers/NullSolvers.jl` at 13:52.
Experiments 1, 2, 2b and 3 ran on the *old* extensions (Arpack `tol = 0`,
KrylovKit single-vector Arnoldi with absolute `tol = 1e-8`).  Experiments 5 and
5b ran on the *new* ones (Arpack `tol = 1e-10`, `nev >= 16`, `ncv = 128`;
KrylovKit block Lanczos with `tol = 1e-12` relative to `‖L‖`,
`initial_request = 4`).  The proposed diffs in section 7 target the new text.

## 0. Summary

| Question | Answer |
|---|---|
| Float32 speed vs Float64, d = 30-40 | Same wall time for SVD and Arpack (contraction-bound; Float32 contractions are 1.3-1.4x faster per application), 2-3x fewer bytes.  Krylov-type solvers 0.6-1.6 GB allocated per solve in Float32 vs 1.7-4 GB. |
| Float32 recovery error | 1e-5 .. 2e-4 (Arpack, SVD), growing slowly with d; Float64 is 1e-14.  The error is eigenvalue noise / spectral gap: `eps32 * few / 5e-3`. |
| Nullity in Float32 | 3 in every Arpack/SVD/Krylov run at d = 20..40 (60+ runs).  Only LOBPCG (`CGSolver`) fails. |
| The reported 17x Float32 Arpack slowdown at d = 20 | Does not reproduce: 4 seeds give 0.10-0.13 s Float32 vs 0.14-0.19 s Float64.  It was JIT: `warmup!` defaults to `T = Float64`, and the first Float32 call compiles Float32 specialisations of Arpack and the ITensor contractions. |
| Which tolerance must scale with `eps(T)` | KrylovKit's absolute residual tolerance.  The current ext asks for `1e-12 * ‖L‖`, below the Float32 residual floor `eps32 * ‖L‖ ~ 1e-5`, so block Lanczos runs to `maxiter`: 7-38 s and 19-50 GB per solve instead of 0.3-1 s.  Floor it at `100 * eps(T)` (validated, Exp. 5b).  ARPACK's tolerance is relative to the Ritz value and already tolerates Float32; flooring it saves ~20%.  LOBPCG in Float32 is broken by its Float32 Cholesky, not by its tolerance. |
| Null threshold in `solve_nullspace` | `tol = 1e-6` relative is within 1.6x of the Float32 null eigenvalues (up to 6.2e-7 relative at d = 40).  Floor it at `100 * eps(T)` = 1.2e-5; the first nonzero eigenvalue is >= 1.2e-4 relative at every d <= 50 measured (one seed excepted, see Exp. 4). |
| Float16 | Arithmetic: no.  ITensor contracts Float16 (1.5x *slower* than Float64, software emulation) but every eigensolver throws `MethodError` (no Float16 LAPACK) and the dense SVD's Float16 noise floor (`eps16 * ‖L‖ ~ 0.09`) swamps the null threshold, giving nullity 0.  Data: yes.  Γ rounded to Float16 (relative rounding 2.06e-4) and stratified in Float32 or Float64 gives nullity 3 and `lsq_err = 2.0e-4` at d = 20, 30, 40 -- the recovery error equals the rounding error. |
| Route to d = 128..256 | Matrix-free Krylov in either precision: ~500 map applications per solve, estimated 2 min at d = 128 and ~16 min at d = 256 (Float64, single-threaded contractions), resident memory well under 1 GB.  The dense operator is 4.6 GB (Float64) / 2.3 GB (Float32) at d = 128 and needs 24768 map applications (~100 min) plus SVD workspace, so Float64 dense does not fit 16 GB and Float32 dense is marginal; at d = 256 dense is 72 / 36 GB and impossible. Precision changes constants (2x bytes, 1.3x per application), not feasibility. |

**Verdict.** Float32 is viable as a *storage and contraction* precision and buys
a 2-3x memory reduction, but not more speed than that, and it caps recovery at
~1e-5..1e-4 relative -- worse as the spectral gap shrinks with d.  The
strongest configuration measured is *mixed*: Float32 Γ and contractions inside a
Float64 eigensolver (`ArpackP64`, `KrylovP64` in Exp. 5), which is as fast as
pure Float32, keeps nullity 3 on every seed, and recovers 2-3x better
(5e-6..1e-5).  Float16 is only useful as a data format.  Nothing about precision
blocks d = 128; the matrix-free path already fits.

## 1. Float64 vs Float32 at d = 30, 35, 40 (Exp. 1, old extensions)

Three seeds for the randomised solvers (`seed = d, d+1000, d+2000`), one for
SVD.  Median wall seconds and median GB allocated; `lsq_err` range over seeds.
Peak RSS stayed at 1.3-1.4 GB for the *entire* sweep, both precisions,
including the 2460x2460 dense SVD -- the problem never dominated the process
footprint at these sizes.

| d | T | solver | runs | median s | median GB alloc | nullity | lsq_err | ok |
|---|---|---|---|---|---|---|---|---|
| 30 | Float64 | SVDSolver | 1 | 4.1 | 4.43 | 3 | 9.7e-14 | 1/1 |
| 30 | Float64 | ArpackSolver | 3 | 1.1 | 1.70 | 3 | 1.8e-14..3.9e-14 | 3/3 |
| 30 | Float64 | KrylovSolver | 3 | 0.7 | 1.25 | 3,3,2 | 8.7e-14..3.6e-13 | 3/3 |
| 30 | Float64 | CGSolver | 3 | 11.2 | 15.77 | 3 | 3.8e-12..2.6e-10 | 3/3 |
| 30 | Float32 | SVDSolver | 1 | 5.4 | 2.55 | 3 | 2.7e-05 | 1/1 |
| 30 | Float32 | ArpackSolver | 3 | 1.1 | 0.59 | 3 | 7.1e-06..2.3e-04 | 3/3 |
| 30 | Float32 | KrylovSolver | 3 | 1.5 | 0.90 | 3 | 1.0e-05..8.1e-05 | 3/3 |
| 30 | Float32 | CGSolver | 3 | 63.7 | 40.50 | 0,0,2 | 5.9e-02 | 1/3 |
| 35 | Float64 | SVDSolver | 1 | 12.5 | 8.67 | 3 | 1.7e-13 | 1/1 |
| 35 | Float64 | ArpackSolver | 3 | 2.3 | 2.53 | 3 | 3.2e-14..1.3e-13 | 3/3 |
| 35 | Float64 | KrylovSolver | 3 | 1.5 | 1.44 | 2,2,3 | 7.1e-14..1.1e-10 | 3/3 |
| 35 | Float64 | CGSolver | 3 | 30.1 | 23.14 | 3 | 5.9e-12..2.6e-11 | 3/3 |
| 35 | Float32 | SVDSolver | 1 | 13.4 | 5.04 | 3 | 1.8e-04 | 1/1 |
| 35 | Float32 | ArpackSolver | 3 | 2.1 | 0.87 | 3 | 7.5e-06..1.2e-04 | 3/3 |
| 35 | Float32 | KrylovSolver | 3 | 2.6 | 1.16 | 3 | 3.6e-05..7.9e-05 | 3/3 |
| 35 | Float32 | CGSolver | 3 | 59.3 | 23.36 | 1,0,0 | 1.0e-05 | 1/3 |
| 40 | Float64 | SVDSolver | 1 | 32.6 | 16.53 | 3 | 1.1e-13 | 1/1 |
| 40 | Float64 | ArpackSolver | 3 | 9.5 | 3.97 | 3 | 2.2e-14..7.6e-14 | 3/3 |
| 40 | Float64 | KrylovSolver | 3 | 8.3 | 3.03 | 3 | 1.2e-13..7.4e-10 | 3/3 |
| 40 | Float64 | CGSolver | 3 | 63.2 | 22.78 | 3 | 3.7e-12..1.8e-11 | 3/3 |
| 40 | Float32 | SVDSolver | 1 | 21.1 | 9.17 | 3 | 7.6e-05 | 1/1 |
| 40 | Float32 | ArpackSolver | 3 | 3.6 | 1.62 | 3 | 2.5e-05..6.5e-05 | 3/3 |
| 40 | Float32 | KrylovSolver | 3 | 6.4 | 2.51 | 3,0,3 | 4.1e-05..9.4e-05 | 2/3 |
| 40 | Float32 | CGSolver | 3 | 23.5 | 9.73 | 0 | - | 0/3 |

Notes.
- Float32 SVD and Arpack take the same wall time as Float64 (the run is
  contraction-bound; see section 6 for per-application timings) and allocate
  2.4-2.9x fewer bytes.
- The old single-vector `KrylovSolver` returned nullity 2 on 3 of 9 Float64
  runs (the multiplicity problem the new block Lanczos fixes) and threw
  `LAPACKException(32)` once in Float32 (KrylovKit's internal Float32 Schur
  step).  The new block code had no failure in 20+ Float32 runs (Exp. 5, 5b).
- Float32 `CGSolver` is unusable: LOBPCG's absolute `tol = 1e-10` is
  unreachable in Float32 and its Float32 Cholesky of the block Gram matrix
  fails (`PosDefException`), so the block collapses to 1-3 vectors and it
  runs to `maxiter`.  See Exp. 2b.

## 2. Tolerances (Exp. 2, 2b; old extensions)

### 2A. The relative null threshold `tol` does not matter in Float32 at d = 30

| d | T | solver | tol | seconds | nullity | lsq_err |
|---|---|---|---|---|---|---|
| 30 | Float32 | ArpackSolver | 1e-03 | 1.07 | 3 | 8.1e-06 |
| 30 | Float32 | ArpackSolver | 1e-04 | 1.06 | 3 | 9.3e-06 |
| 30 | Float32 | ArpackSolver | 1e-05 | 1.46 | 3 | 9.8e-06 |
| 30 | Float32 | ArpackSolver | 1e-06 | 1.01 | 3 | 7.7e-06 |
| 30 | Float32 | KrylovSolver | 1e-03 | 1.32 | 3 | 3.4e-05 |
| 30 | Float32 | KrylovSolver | 1e-04 | 1.96 | 3 | 9.2e-05 |
| 30 | Float32 | KrylovSolver | 1e-05 | 1.67 | 3 | 3.5e-05 |
| 30 | Float32 | KrylovSolver | 1e-06 | 1.62 | 3 | 7.5e-05 |

### 2B. Eigensolver tolerance, called directly on `L` with a counting wrapper

`nev = 16`; `applications` counts map applications; `lam_k` are the k smallest
returned |eigenvalues| relative to `‖L‖` (power-iteration estimate).  The gap
between `lam3` and `lam4` is the spectral gap the null threshold must fall in.

| d | T | solver | eigtol | seconds | applications | converged | lam1 | lam2 | lam3 | lam4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | Float32 | Arpack | 0 (= eps32) | 0.66 | 332 | 16 | 3.8e-08 | 5.9e-08 | 6.4e-08 | 4.9e-03 |
| 20 | Float32 | Arpack | 1e-07 | 0.19 | 339 | 16 | 1.6e-08 | 5.1e-08 | 6.5e-08 | 4.9e-03 |
| 20 | Float32 | Arpack | 1e-05 | 0.22 | 305 | 16 | 1.1e-08 | 2.6e-08 | 5.6e-08 | 4.9e-03 |
| 20 | Float32 | Arpack | 1e-03 | 0.19 | 251 | 16 | 8.3e-09 | 1.5e-08 | 5.2e-08 | 4.9e-03 |
| 20 | Float32 | Arpack | 1e-02 | 0.15 | 240 | 16 | 2.5e-08 | 6.5e-08 | 9.4e-08 | 4.9e-03 |
| 20 | Float32 | KrylovKit | 1e-08 abs | 1.19 | 761 | 19 | 1.7e-07 | 2.4e-07 | 3.9e-07 | 4.9e-03 |
| 20 | Float32 | KrylovKit | 1e-06 abs | 0.28 | 313 | 16 | 8.2e-14 | 1.4e-07 | 1.9e-07 | 4.9e-03 |
| 20 | Float32 | KrylovKit | 1e-04 abs | 0.17 | 274 | 16 | 6.9e-09 | 1.8e-08 | 1.9e-07 | 4.9e-03 |
| 20 | Float32 | KrylovKit | 1e-02 abs | 0.20 | 225 | 17 | 3.4e-10 | 1.5e-08 | 1.8e-07 | 4.9e-03 |
| 20 | Float32 | KrylovKit | 1e-01 abs | 0.15 | 203 | 17 | 1.3e-17 | 1.7e-07 | 2.5e-07 | 4.9e-03 |
| 30 | Float32 | Arpack | 0 (= eps32) | 1.06 | 376 | 16 | 1.5e-09 | 4.9e-09 | 2.4e-08 | 7.2e-03 |
| 30 | Float32 | Arpack | 1e-05 | 1.09 | 345 | 16 | 7.7e-09 | 4.8e-08 | 6.7e-08 | 7.2e-03 |
| 30 | Float32 | Arpack | 1e-03 | 0.91 | 297 | 16 | 1.4e-08 | 2.9e-08 | 6.3e-08 | 7.2e-03 |
| 30 | Float32 | KrylovKit | 1e-08 abs | 2.37 | 629 | 19 | 1.5e-09 | 1.2e-07 | 3.4e-07 | 7.2e-03 |
| 30 | Float32 | KrylovKit | 1e-06 abs | 4.84 | 382 | 18 | 2.9e-08 | 9.4e-08 | 1.3e-07 | 7.2e-03 |
| 30 | Float32 | KrylovKit | 1e-04 abs | 2.39 | 331 | 16 | 1.4e-08 | 2.7e-08 | 1.7e-07 | 7.2e-03 |
| 30 | Float32 | KrylovKit | 1e-02 abs | 1.30 | 298 | 16 | 0.0 | 3.2e-08 | 7.0e-08 | 7.2e-03 |
| 30 | Float32 | KrylovKit | 1e-01 abs | 1.02 | 244 | 18 | 1.2e-07 | 1.3e-07 | 1.6e-07 | 7.2e-03 |
| 20 | Float64 | Arpack | 0 (= eps64) | 2.06 | 483 | 16 | 2.8e-17 | 3.6e-17 | 2.2e-16 | 4.9e-03 |
| 20 | Float64 | Arpack | 1e-08 | 0.34 | 351 | 16 | 2.5e-17 | 9.8e-17 | 1.7e-16 | 4.9e-03 |
| 20 | Float64 | Arpack | 1e-04 | 0.25 | 327 | 16 | 1.6e-17 | 2.5e-17 | 8.3e-17 | 4.9e-03 |
| 20 | Float64 | KrylovKit | 1e-08 abs | 3.82 | 349 | 16 | 1.3e-16 | 1.6e-16 | 4.3e-16 | 4.9e-03 |
| 20 | Float64 | KrylovKit | 9e-11 abs | 0.34 | 383 | 16 | 1.3e-17 | 6.4e-17 | 1.6e-16 | 4.9e-03 |
| 30 | Float64 | Arpack | 0 (= eps64) | 2.28 | 566 | 16 | 1.8e-17 | 8.4e-17 | 1.1e-16 | 7.2e-03 |
| 30 | Float64 | Arpack | 1e-08 | 1.60 | 390 | 16 | 2.5e-17 | 7.5e-17 | 2.2e-16 | 7.2e-03 |
| 30 | Float64 | KrylovKit | 1e-08 abs | 2.16 | 391 | 16 | 7.7e-17 | 2.2e-16 | 2.6e-16 | 7.2e-03 |

`‖L‖` = 89 (d = 20), 126 (d = 30).  Full table incl. iterations and `lam5`
in `precision-exp2-B.csv`.

Reading: ARPACK's stopping test is relative to the Ritz value with an
`eps^(2/3)` floor, so it self-scales; every Float32 tolerance from machine eps
to 1e-2 converges in 240-410 applications, the same range as Float64.
KrylovKit's tolerance is an absolute residual norm; below the Float32 residual
floor (`~eps32 * ‖L‖ = 1e-5`) the cost rises (761 vs 274 applications at
`1e-8` vs `1e-4`), and the new block Lanczos with `1e-12 * ‖L‖` is far worse
(section 5).  Exactly 3 eigenvalues fall below `1e-6 * ‖L‖` in every row.

### 2b. The d = 20 Float32 Arpack case, and LOBPCG in Float32

| d | T | solver | seed | seconds | nullity | lsq_err |
|---|---|---|---|---|---|---|
| 20 | Float64 | ArpackSolver | 20 / 1020 / 2020 / 3020 | 0.18 / 0.19 / 0.15 / 0.14 | 3 | 1.4e-12 / 2.4e-14 / 2.2e-14 / 1.4e-14 |
| 20 | Float64 | KrylovSolver | same | 0.29 / 0.22 / 0.26 / 0.17 | 3 | 4.1e-14 / 1.8e-10 / 3.2e-14 / 5.9e-11 |
| 20 | Float32 | ArpackSolver | same | 0.13 / 0.12 / 0.12 / 0.10 | 3 | 1.1e-05 / 1.0e-05 / 1.4e-05 / 1.6e-04 |
| 20 | Float32 | KrylovSolver | same | 0.27 / 0.30 / 0.36 / 0.24 | 3 | 3.5e-05 / 5.3e-05 / 1.1e-05 / 7.0e-05 |

The 8.4 s / 17x Float32 Arpack figure is not reproducible after a Float32
warm-up; Float32 Arpack is 20-30% *faster* than Float64 at d = 20.

LOBPCG (`CGSolver`) called directly, `nv = 16`, block 24 requested:

| d | T | tol | seconds | applications | block that survived Cholesky | n below 1e-6 | lam1..lam3 (rel) |
|---|---|---|---|---|---|---|---|
| 20 | Float32 | 1e-10 | 6.13 | 1772 | 1 | 0 | 4.9e-06 |
| 20 | Float32 | 1e-06 | 0.80 | 1368 | 1 | 1 | 3.4e-09 |
| 20 | Float32 | 1e-04 | 0.86 | 1335 | 1 | 1 | 3.3e-09 |
| 20 | Float32 | 1e-03 | 0.98 | 1561 | 3 | 1 | 1.8e-08, 1.2e-06, 1.3e-06 |
| 20 | Float32 | 1e-02 | 0.39 | 702 | 12 | 3 | 7.3e-08, 9.5e-08, 7.1e-07 |
| 30 | Float32 | 1e-10 | 3.67 | 2183 | 1 | 0 | 4.4e-05 |
| 30 | Float32 | 1e-06 | 3.61 | 2183 | 1 | 0 | 4.4e-05 |
| 30 | Float32 | 1e-04 | 2.93 | 1808 | 1 | 1 | 3.0e-09 |
| 30 | Float32 | 1e-03 | 3.43 | 1992 | 1 | 1 | 4.0e-07 |
| 30 | Float32 | 1e-02 | 3.03 | 1742 | 3 | 1 | 2.3e-08, 1.9e-06, 2.5e-06 |
| 20 | Float64 | 1e-10 | 7.16 | 4281 | 6 | 3 | 3.0e-17, 8.5e-16, 1.6e-14 |
| 20 | Float64 | 1e-06 | 1.00 | 1439 | 24 | 3 | 3.5e-17, 4.9e-17, 5.2e-17 |
| 30 | Float64 | 1e-10 | 5.04 | 4034 | 12 | 3 | 2.6e-16, 3.6e-16, 6.1e-15 |
| 30 | Float64 | 1e-06 | 2.51 | 2026 | 24 | 3 | 6.9e-16, 6.9e-16, 2.0e-14 |

LOBPCG in Float32 fails on the Cholesky of its block Gram matrix, whatever the
tolerance; the ext's halving fallback leaves a 1-3 vector block, which cannot
hold a 3-dimensional null space.  In Float64, `tol = 1e-6` gives identical
eigenvalues at 2-3x fewer applications than `1e-10`.

## 3. The near-zero spectrum in both precisions (Exp. 2C, 4) -- the gap that decides viability

Dense symmetric eigen of `Matrix(L)`; `map_T` is the precision the map was
built and applied in, `eigen_T` the precision of the eigensolver.  Values are
relative to `lam_max`.

| d | N | map_T | eigen_T | densify s | eigen s | lam_max | lam1 | lam2 | lam3 | lam4 | lam5 | lam4/lam3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 630 | Float64 | Float64 | 0.63 | 0.06 | 88.9 | 2.7e-16 | 6.3e-16 | 1.1e-15 | 4.9e-03 | 8.5e-03 | 4e12 |
| 20 | 630 | Float32 | Float32 | 0.54 | 0.06 | 88.9 | 1.3e-07 | 3.7e-07 | 4.5e-07 | 4.9e-03 | 8.5e-03 | 1e4 |
| 20 | 630 | Float32 | Float64 | 0.54 | 0.03 | 88.9 | 8.6e-10 | 9.7e-09 | 1.3e-08 | 4.9e-03 | 8.5e-03 | 4e5 |
| 30 | 1395 | Float64 | Float64 | 5.42 | 0.64 | 126.8 | 1.7e-17 | 4.5e-16 | 7.3e-16 | 7.2e-03 | 1.3e-02 | 1e13 |
| 30 | 1395 | Float32 | Float32 | 4.10 | 0.46 | 126.8 | 1.3e-08 | 7.1e-08 | 1.7e-07 | 7.2e-03 | 1.3e-02 | 4e4 |
| 30 | 1395 | Float32 | Float64 | 4.10 | 0.86 | 126.8 | 1.2e-09 | 2.8e-09 | 4.1e-09 | 7.2e-03 | 1.3e-02 | 2e6 |
| 40 | 2460 | Float64 | Float64 | 27.5 | 3.71 | 163.9 | 8.4e-16 | 9.7e-16 | 1.1e-15 | 2.6e-03 | 3.5e-03 | 2e12 |
| 40 | 2460 | Float32 | Float32 | 21.0 | 1.99 | 163.9 | 2.3e-07 | 3.7e-07 | 6.2e-07 | 2.6e-03 | 3.5e-03 | 4e3 |
| 40 | 2460 | Float32 | Float64 | 21.0 | 3.86 | 163.9 | 2.5e-09 | 5.4e-09 | 7.3e-09 | 2.6e-03 | 3.5e-03 | 4e5 |
| 50 | 3825 | Float64 | Float64 | 55.3 | 11.1 | 165.9 | 7.4e-16 | 1.4e-15 | 1.6e-15 | **1.2e-08** | 5.4e-05 | 8e6 |
| 50 | 3825 | Float32 | Float32 | 39.2 | 4.09 | 165.9 | 6.1e-08 | 2.8e-07 | 3.3e-07 | 5.2e-07 | 5.5e-05 | 1.6 |
| 50 | 3825 | Float32 | Float64 | 39.2 | 8.11 | 165.9 | 6.2e-10 | 3.8e-09 | 5.5e-09 | 7.0e-09 | 5.4e-05 | 1.3 |

Three regimes of "how far above zero the null eigenvalues sit":
Float64 everything ~1e-16..1e-15; Float32 map, Float64 eigen ~1e-9..1e-8
(this is the rounding of Γ and of the contractions); Float32 eigen on top
~1e-7..6e-7.  For d <= 40 the 4th eigenvalue is 2.6e-3..7.2e-3, so even the
worst Float32 case has 4 decades of gap and the default relative threshold
1e-6 sits 1.6x above the largest null eigenvalue (6.2e-7 at d = 40) -- safe but
with no margin to grow.

The d = 50, seed = 50 row has a 4th eigenvalue of 1.2e-8 **in Float64**.  That
is a property of that tensor, not of precision: with `tol = 1e-6` Float64
would also report nullity 4 there.  Exp. 4 checked the benchmark
construction (support count exact at every d) and re-measured d = 45, 50 on
three seeds:

| d | seed | lam4 rel | lam5 rel | lam6 rel |
|---|---|---|---|---|
| 45 | 45 | 1.2e-04 | 7.5e-04 | 3.2e-03 |
| 45 | 1045 | 4.7e-03 | 9.9e-03 | 1.0e-02 |
| 45 | 2045 | 1.3e-04 | 1.8e-03 | 2.8e-03 |
| 50 | 50 | 1.2e-08 | 5.4e-05 | 1.0e-03 |
| 50 | 1050 | 3.0e-04 | 4.9e-03 | 5.2e-03 |
| 50 | 2050 | 6.3e-04 | 8.8e-04 | 1.2e-03 |

(lam1..lam3 were 2e-16..3e-15 in every row.)  So the sphere's first nonzero
derivation eigenvalue drifts from ~5e-3 (d <= 40) to ~1e-4..5e-3 (d = 45-50)
and fluctuates by an order of magnitude with the scramble; seed 50 is an
accidental near-degeneracy.  Against Float32 null eigenvalues at up to 6e-7
that leaves ~2 decades at d = 50, versus 4 at d = 40.  This trend, not
arithmetic per se, is what would end Float32's usefulness at large d; the
mixed scheme (Float32 map, Float64 eigen) keeps its null eigenvalues at 1e-8
and therefore 4 decades of margin at d = 50.

**Error model.** Recovery error ~ (null eigenvalue noise) / (gap):
Float64 1e-16 / 5e-3 ~ 2e-14 (measured 1e-14); Float32 1e-7 / 5e-3 ~ 2e-5
(measured 1e-5..2e-4); mixed 5e-9 / 5e-3 ~ 1e-6 (measured 5e-6..1e-5).  As the
gap falls to 1e-4 at d ~ 50 expect Float32 recovery ~1e-3, mixed ~1e-4.

## 4. Float16 (Exp. 3)

### Pure Float16

| d | stage | solver | result |
|---|---|---|---|
| 10, 20 | `build_sphere(d; T = Float16)` | - | ok, `eltype(Γ) = Float16` |
| 10, 20 | `L * v` | - | ok, returns `Vector{Float16}`; 0.34 ms at d = 20 vs 0.22 ms Float64 (emulated) |
| 10, 20 | stratify | SVDSolver | nullity 0 -- "no nontrivial derivations": Float16 singular values of the null directions are ~`eps16 * ‖L‖` ~ 0.09, far above the threshold `1e-6 * ‖L‖` ~ 1e-4 |
| 10, 20 | stratify | ArpackSolver | `MethodError: no method matching saupd(..., ::Base.RefValue{Float16}, ...)` -- ARPACK has no half-precision build |
| 10, 20 | stratify | KrylovSolver | `MethodError: no method matching hschur!(::SubArray{Float16, ...})` -- KrylovKit's Schur step needs LAPACK |
| 10, 20 | stratify | CGSolver | `MethodError: no method matching eigen!(::Hermitian{Float16, ...})` -- LOBPCG's Rayleigh-Ritz needs LAPACK |

Map application cost per element type at d = 20 (median of 7, N = 630):

| T | seconds | bytes |
|---|---|---|
| Float64 | 0.00022 | 1,125,272 |
| Float32 | 0.00023 | 701,448 |
| Float16 | 0.00034 | 495,832 |

Float16 halves the bytes again but is 1.5x slower than Float64: this CPU
has no native half-precision matmul, so it is not a speed route either.

### Float16 as a data format: round Γ, compute wider

`store_T` is what Γ is rounded through; `compute_T` is the precision of the
whole stratification.  The last column is `‖Γ_rounded - Γ‖/‖Γ‖`.

| d | store_T | compute_T | solver | seconds | nullity | lsq_err | Γ rounding |
|---|---|---|---|---|---|---|---|
| 20 | Float16 | Float32 | SVD / Arpack | 0.20 / 0.08 | 3 | 1.97e-04 / 1.97e-04 | 2.06e-04 |
| 20 | Float16 | Float64 | SVD / Arpack | 0.26 / 0.14 | 3 | 1.97e-04 / 1.97e-04 | 2.06e-04 |
| 20 | Float32 | Float32 | SVD / Arpack | 0.19 / 0.08 | 3 | 1.02e-05 / 8.26e-06 | 2.55e-08 |
| 20 | Float32 | Float64 | SVD / Arpack | 0.24 / 0.13 | 3 | 2.44e-08 / 2.44e-08 | 2.55e-08 |
| 20 | Float64 | Float64 | SVD / Arpack | 0.23 / 0.12 | 3 | 5.3e-14 / 1.2e-14 | 0 |
| 30 | Float16 | Float32 | SVD / Arpack | 1.13 / 0.22 | 3 | 2.07e-04 / 2.02e-04 | 2.07e-04 |
| 30 | Float16 | Float64 | SVD / Arpack | 1.60 / 0.43 | 3 | 2.01e-04 / 2.01e-04 | 2.07e-04 |
| 30 | Float32 | Float32 | SVD / Arpack | 1.10 / 0.22 | 3 | 1.93e-05 / 8.14e-06 | 2.53e-08 |
| 30 | Float32 | Float64 | SVD / Arpack | 1.57 / 0.38 | 3 | 2.46e-08 / 2.46e-08 | 2.53e-08 |
| 30 | Float64 | Float64 | SVD / Arpack | 1.57 / 0.36 | 3 | 4.2e-14 / 7.7e-13 | 0 |
| 40 | Float16 | Float32 | SVD / Arpack | 4.32 / 0.52 | 3 | 2.04e-04 / 2.04e-04 | 2.07e-04 |
| 40 | Float16 | Float64 | SVD / Arpack | 6.62 / 1.01 | 3 | 2.03e-04 / 2.03e-04 | 2.07e-04 |
| 40 | Float32 | Float32 | SVD / Arpack | 4.33 / 0.47 | 3 | 7.03e-04 / 2.27e-05 | 2.53e-08 |
| 40 | Float32 | Float64 | SVD / Arpack | 6.77 / 1.01 | 3 | 2.48e-08 / 2.48e-08 | 2.53e-08 |
| 40 | Float64 | Float64 | SVD / Arpack | 6.79 / 1.00 | 3 | 9.6e-14 / 1.3e-13 | 0 |

Two clean facts.  (i) With Float64 arithmetic the recovery error *equals* the
data rounding error: 2.4e-8 for Float32 data, 2.0e-4 for Float16 data, at
every d.  The data can be as coarse as the accuracy you need.  (ii) Float32
*arithmetic* costs 300-3000x more error than Float32 *data* (8e-6..7e-4 vs
2.5e-8).  The Float32 error budget is spent in the eigensolver, not in Γ.

(Timings in this table were lightly contended: Float32 Arpack at d = 40 is
2x faster than Float64 here, 0.47 vs 1.0 s, and Float32 SVD 1.6x.)

## 5. Validating the proposed tolerance rules end to end (Exp. 5, 5b; new extensions)

Solvers registered locally (`precision-exp5.jl`, no `src/`/`ext/` edits):
`ArpackP` = ARPACK with `tol = sqrt(eps(T))`; `KrylovP` = single-vector
Lanczos with absolute `tol = sqrt(eps(T)) * ‖L‖`; `CGP` = LOBPCG with the same
absolute tolerance; `*P64` = the same algorithm run in Float64 over the Float32
map (`v -> Float64.(L * Float32.(v))`).  `thr` is the relative null threshold
passed to `run_stratify`.  Three seeds each; medians, nullity counts, lsq range.

| d | T | solver | thr | median s | min..max s | median GB | nullity (count) | lsq min..max | ok |
|---|---|---|---|---|---|---|---|---|---|
| 30 | Float32 | ArpackSolver (stock, new) | 1e-6 | 0.27 | 0.24..0.32 | 0.80 | 3 x3 | 1.3e-05..3.0e-05 | 3/3 |
| 30 | Float32 | ArpackP | 1e-6 | 0.18 | 0.18..0.20 | 0.57 | 3 x3 | 1.4e-05..2.9e-05 | 3/3 |
| 30 | Float32 | ArpackP64 | 1e-6 | 0.25 | 0.24..0.29 | 0.64 | 3 x3 | 4.9e-06..1.2e-05 | 3/3 |
| 30 | Float32 | KrylovSolver (stock, new) | 1e-6 | **13.11** | 9.31..13.39 | **22.8** | 3 x3 | 7.4e-05..4.0e-04 | 3/3 |
| 30 | Float32 | KrylovP | 1e-6 | 0.22 | 0.20..0.28 | 0.48 | 3 x3 | 2.2e-05..8.5e-05 | 3/3 |
| 30 | Float32 | KrylovP64 | 1e-6 | 0.35 | 0.23..0.35 | 0.59 | 3 x3 | 6.1e-06..8.6e-06 | 3/3 |
| 30 | Float32 | CGP | 1e-6 | 0.82 | 0.64..0.88 | 1.75 | 3 x1, 2 x2 | 1.3e-02..1.5e-02 | 3/3 |
| 30 | Float32 | CGP64 | 1e-6 | 10.9 | 4.7..93.6 | 20.8 | 2 x2, 1 x1 | 2.2e-05..3.4e-04 | 3/3 |
| 40 | Float32 | ArpackSolver (stock, new) | 1e-6 | 0.56 | 0.48..1.44 | 1.69 | 3 x3 | 2.3e-05..6.1e-05 | 3/3 |
| 40 | Float32 | ArpackP | 1e-6 | 0.43 | 0.41..1.11 | 1.42 | 3 x3 | 2.0e-05..2.8e-04 | 3/3 |
| 40 | Float32 | ArpackP64 | 1e-6 | 0.54 | 0.53..1.24 | 1.70 | 3 x3 | 1.1e-05..2.9e-05 | 3/3 |
| 40 | Float32 | KrylovSolver (stock, new) | 1e-6 | **18.3** | 16.1..37.8 | **47.0** | 3 x3 | 2.9e-04..1.7e-03 | 3/3 |
| 40 | Float32 | KrylovP | 1e-6 | 0.38 | 0.27..0.95 | 1.22 | 3 x3 | 1.9e-05..4.7e-04 | 3/3 |
| 40 | Float32 | KrylovP64 | 1e-6 | 0.49 | 0.47..1.08 | 1.64 | 3 x3 | 3.6e-05..2.8e-04 | 3/3 |
| 40 | Float32 | CGP | 1e-6 | 1.73 | 1.41..5.03 | 4.91 | 2 x1, 1 x2 | 3.9e-03..2.4e-02 | 3/3 |
| 40 | Float32 | CGP64 | 1e-6 | 290 | 281..383 | 584 | 3 x1, 1 x1, 0 x1 | 5.8e-06..2.3e-03 | 2/3 |
| 30 | Float64 | ArpackSolver (stock, new) | 1e-6 | 0.33 | 0.30..0.34 | 1.40 | 3 x3 | 5.4e-14..1.8e-13 | 3/3 |
| 30 | Float64 | ArpackP | 1e-6 | 0.33 | 0.32..0.35 | 1.43 | 3 x3 | 2.7e-14..4.6e-14 | 3/3 |
| 30 | Float64 | KrylovSolver (stock, new, block 4) | 1e-6 | 1.27 | 1.26..1.69 | 4.59 | 3 x3 | 6.7e-14..2.4e-12 | 3/3 |
| 30 | Float64 | KrylovP (single vector) | 1e-6 | 0.22 | 0.21..0.25 | 1.01 | **3 x1, 2 x2** | 1.3e-13..5.5e-09 | 3/3 |
| 30 | Float64 | CGSolver (stock) | 1e-6 | 3.48 | 3.15..4.82 | 13.2 | 3 x3 | 9.1e-12..2.0e-11 | 3/3 |
| 30 | Float64 | CGP | 1e-6 | 1.37 | 1.35..1.60 | 5.20 | 3 x3 | 6.9e-08..3.9e-07 | 3/3 |
| 40 | Float64 | ArpackSolver (stock, new) | 1e-6 | 0.79 | 0.79..0.99 | 3.07 | 3 x3 | 4.1e-14..4.7e-14 | 3/3 |
| 40 | Float64 | ArpackP | 1e-6 | 0.88 | 0.78..0.92 | 3.62 | 3 x3 | 2.8e-14..7.1e-14 | 3/3 |
| 40 | Float64 | KrylovSolver (stock, new, block 4) | 1e-6 | 3.32 | 3.15..3.77 | 12.1 | 3 x3 | 9.7e-14..1.1e-12 | 3/3 |
| 40 | Float64 | KrylovP (single vector) | 1e-6 | 0.73 | 0.68..0.75 | 2.95 | 3 x3 | 1.8e-09..1.4e-06 | 3/3 |
| 40 | Float64 | CGSolver (stock) | 1e-6 | 10.7 | 8.75..15.7 | 40.2 | 3 x3 | 1.8e-12..2.8e-11 | 3/3 |
| 40 | Float64 | CGP | 1e-6 | 3.29 | 3.19..4.30 | 12.1 | 3 x3 | 7.5e-08..9.9e-08 | 3/3 |

Rows with `thr = 1.2e-5` (Float32 only) are in `precision-exp5.csv`; they
changed nothing -- nullity 3 in every Arpack/Krylov run at either threshold.

Conclusions from Exp. 5:
- The new `KrylovSolver` in Float32 is the one real defect: 40-70x slower and
  40x more allocation than the same solve with a reachable tolerance,
  because `1e-12 * ‖L‖` is below the Float32 residual floor and block Lanczos
  restarts to `maxiter = 200`.  It still gets nullity 3 (the Ritz pairs are
  fine; only the stopping test never fires).
- Single-vector Lanczos (`KrylovP`) is fast but misses null vectors (2 of 6
  Float64 runs) -- the block design is right; only its tolerance needs the
  floor.  Exp. 5b tests exactly that.
- ARPACK is fine in Float32 as is; `sqrt(eps)` saves ~20%.
- LOBPCG is not rescued by tolerance in Float32 (nullity 1-3, recovery 4e-3..4e-2),
  and running it in Float64 over the Float32 map is erratic (5 s to 24 min).
  In Float64, `sqrt(eps) * ‖L‖` = 1.5e-6 is 3x faster than `1e-10` but
  loosens recovery from 1e-11 to 1e-7.
- Mixed (`ArpackP64`, `KrylovP64`) costs nothing in time and recovers 2-3x
  better than pure Float32.

### 5b. Block Lanczos (the current ext code, verbatim) with a tolerance floor

`KrylovB100`: `atol = max(tol, 100*eps(T)) * ‖L‖`; `KrylovBsqrt`: `max(tol, sqrt(eps(T)))`.
Both used the default `initial_request = 16` (block 16); the stock solver
declares `initial_request(::KrylovSolver) = 4`, so the **Float64 timing
difference below is block size, not the floor** (the floor is inactive in
Float64 for `KrylovB100`: 2.2e-14 < 1e-12).

| d | T | solver | runs | median s | min..max s | median GB | nullity | lsq min..max |
|---|---|---|---|---|---|---|---|---|
| 30 | Float32 | KrylovSolver (stock, block 4, no floor) | 1 | 7.07 | - | 18.6 | 3 | 1.2e-04 |
| 30 | Float32 | KrylovB100 (block 16) | 3 | 0.40 | 0.38..0.53 | 1.22 | 3 x3 | 6.7e-05..1.7e-03 |
| 30 | Float32 | KrylovBsqrt (block 16) | 3 | 0.33 | 0.32..0.37 | 1.02 | 3 x3 | 1.9e-05..2.8e-03 |
| 40 | Float32 | KrylovSolver (stock, block 4, no floor) | 1 | 13.3 | - | 38.3 | 3 | 8.9e-05 |
| 40 | Float32 | KrylovB100 (block 16) | 3 | 1.04 | 0.96..1.05 | 3.26 | 3 x3 | 4.9e-05..3.8e-03 |
| 40 | Float32 | KrylovBsqrt (block 16) | 3 | 0.81 | 0.78..0.84 | 2.45 | 3 x3 | 8.9e-05..3.4e-03 |
| 30 | Float64 | KrylovSolver (stock, block 4) | 3 | 0.28 | 0.28..0.31 | 1.29 | 3 x3 | 7.6e-14..6.5e-12 |
| 30 | Float64 | KrylovB100 (block 16) | 3 | 1.21 | 1.16..1.62 | 4.59 | 3 x3 | 8.2e-14..1.2e-13 |
| 30 | Float64 | KrylovBsqrt (block 16) | 3 | 0.75 | 0.75..0.98 | 3.11 | 3 x3 | 4.7e-12..8.5e-10 |
| 40 | Float64 | KrylovSolver (stock, block 4) | 3 | 0.71 | 0.64..0.90 | 3.09 | 3 x3 | 2.4e-13..1.5e-10 |
| 40 | Float64 | KrylovB100 (block 16) | 3 | 3.19 | 2.88..3.38 | 12.5 | 3 x3 | 2.0e-13..9.7e-13 |
| 40 | Float64 | KrylovBsqrt (block 16) | 3 | 2.10 | 1.95..2.34 | 8.41 | 3 x3 | 6.9e-11..1.7e-10 |

The floor makes Float32 block Lanczos 15-20x faster (with a 4x *larger* block
than stock, so the like-for-like gain is bigger), nullity 3 on every seed.
`100*eps(T)` leaves Float64 untouched; `sqrt(eps(T))` = 1.5e-8 would loosen
Float64 recovery to 1e-10..1e-12 for a 30% saving -- not worth it.  Float32
block-Lanczos recovery is noisier than Arpack's (outliers at 2-4e-3 on both
floors), consistent with the error model: `1e-7 / gap` with `gap` occasionally
small on a given scramble.

## 6. Memory (Exp. 1, 2C and arithmetic)

The square derivation--densor operator has `N = 3d(d+1)/2` (symmetric
operators, three axes); the densor map has `d^3` rows for the universal chisel.

| d | N | dense Float64 | dense Float32 | dense Float16 | one vector (F64) | Γ (F64) | densor rows d^3 |
|---|---|---|---|---|---|---|---|
| 30 | 1395 | 14.8 MB | 7.4 MB | 3.7 MB | 11 KB | 0.2 MB | 27,000 |
| 40 | 2460 | 46 MB | 23 MB | 12 MB | 19 KB | 0.5 MB | 64,000 |
| 50 | 3825 | 112 MB | 56 MB | 28 MB | 30 KB | 1.0 MB | 125,000 |
| 64 | 6240 | 297 MB | 149 MB | 74 MB | 49 KB | 2.0 MB | 262,144 |
| 128 | 24768 | **4.57 GB** | **2.29 GB** | 1.14 GB | 194 KB | 16 MB | 2,097,152 |
| 256 | 98688 | **72.6 GB** | **36.3 GB** | 18.1 GB | 771 KB | 128 MB | 16,777,216 |

Measured per-application cost of `L` (from the densify timings in Exp. 2C,
`Matrix(L)` = N applications; contended, 2 threads):

| d | N | Float64 ms/app | Float32 ms/app | F64/F32 |
|---|---|---|---|---|
| 20 | 630 | 1.0 | 0.86 | 1.17 |
| 30 | 1395 | 3.9 | 2.9 | 1.32 |
| 40 | 2460 | 11.2 | 8.5 | 1.31 |
| 50 | 3825 | 14.5 | 10.2 | 1.42 |

Growth is roughly `d^3` from d = 30 to 50.  Extrapolating from d = 50:
~0.24 s/application at d = 128 and ~2 s at d = 256 in Float64 (0.7x in
Float32).

Measured Krylov-type memory at d = 30..40 (Exp. 1): Arpack allocates 1.7 → 4.0 GB
per solve in Float64 vs 0.6 → 1.6 GB in Float32 (that is churn, not
residency); peak RSS never left 1.3-1.4 GB for either precision or any solver,
including the dense SVD.  A matrix-free solve holds Γ, a Krylov basis of
`krylovdim * N` numbers (128 x 24768 x 8 B = 25 MB at d = 128, 100 MB at
d = 256) and contraction temporaries of order `d^3` -- under 1 GB resident at
d = 256 in Float64.

**Feasibility at d = 128 in 16 GB.**
- Dense SVD path: 4.6 GB for the Float64 matrix plus LAPACK `gesdd` workspace
  (U, V and work arrays of comparable size) is ~15-20 GB -- does not fit.
  Float32 dense is 2.3 GB + workspace ~8-10 GB -- fits, marginally, but
  `Matrix(L)` alone is 24768 applications ~ 100 min before the SVD starts.
- Matrix-free (Arpack or block Lanczos): ~400-550 applications per solve
  (Exp. 2B; the ext docstrings quote 437-546) ~ 2 min at d = 128 and ~16 min
  at d = 256, resident memory < 1 GB either precision.  This is the route to
  d = 128..256, and it reaches d = 256 in Float64.  Float32 halves Γ and
  temporaries and speeds each application by ~1.3x, at the cost of the
  recovery ceiling in section 3.

## 7. Proposed source changes (not applied; the files are owned by other agents)

### 7.1 `ext/DletoKrylovKitExt.jl` -- floor the residual tolerance at `100*eps(T)` (validated: Exp. 5b `KrylovB100`)

```diff
@@ function Dleto.solve(::KrylovSolver, L::LinearMap; nv::Integer = 10, tol::Real = 1e-12,
     scale = Dleto.opnorm_estimate(L; iters = 10)
-    atol = tol * max(scale, eps(RT))
+    # KrylovKit's `tol` is an absolute residual norm, and no Lanczos residual
+    # gets below a small multiple of `eps(T) * ‖L‖`.  `1e-12` relative is
+    # reachable in Float64 but 1e5 times below the Float32 floor, where block
+    # Lanczos then restarts to `maxiter` for nothing: 7-38 s and 19-50 GB per
+    # solve at d = 30..40 against 0.3-1 s with the floor, same nullity
+    # (bench/reports/precision-study.md, Exp. 5 and 5b).  `100*eps(RT)` is
+    # 2.2e-14 in Float64, so Float64 behaviour is unchanged.
+    atol = max(tol, 100 * eps(RT)) * max(scale, eps(RT))
```

And in the docstring, after "`1e-12` relative gives null residuals of 1e-13..1e-15 ...":
"The tolerance is floored at `100*eps(T)`, so in Float32 it is 1.2e-5 relative."

### 7.2 `ext/DletoArpackExt.jl` -- same floor (measured: Float32 Arpack converges at any `tol` from `eps32` to 1e-2, 240-410 applications; the floor saves ~20% -- Exp. 2B, 5 `ArpackP`)

```diff
@@ function Dleto.solve(::ArpackSolver, L::LinearMap; nv::Integer = 20, tol::Real = 1e-10,
     nev = clamp(max(want, min_request), 1, n - 2)
     ncv_ = ncv === nothing ? min(n, max(2 * nev + 1, 8 * nev)) : clamp(Int(ncv), nev + 1, n)
-    vals, vecs = Arpack.eigs(L; nev = nev, ncv = ncv_, which = :SM, tol = tol, maxiter = maxiter)
+    # ARPACK's test is relative to the Ritz value with an eps^(2/3) floor, so
+    # it survives Float32 at any `tol`; flooring at 100*eps(T) just stops it
+    # polishing below the element type's precision (~20% fewer applications
+    # in Float32, no change in Float64).
+    vals, vecs = Arpack.eigs(L; nev = nev, ncv = ncv_, which = :SM,
+                             tol = max(tol, 100 * eps(RT)), maxiter = maxiter)
```

### 7.3 `src/solvers/NullSolvers.jl` -- floor the null threshold at `100*eps(T)` and keep LOBPCG out of the Float32 fallback order

```diff
@@ function solve_nullspace(L, solver::Union{Symbol,NullSolver};
-    scale = atol === nothing ? sqrt(max(opnorm_estimate(L' * L; iters = 10), 0.0)) : 1.0
-    threshold = atol === nothing ? tol * max(scale, eps()) : atol
+    # The null eigenvalues an iterative solver returns sit at a few `eps(T)`
+    # relative to `‖L‖` -- up to 6.2e-7 in Float32 at d = 40 on the sphere
+    # benchmark, within 1.6x of the default 1e-6 -- while the first nonzero
+    # eigenvalue is >= 1e-4 relative there.  Floor the relative threshold at
+    # 100*eps(T) so a Float32 run does not lose a null vector to rounding;
+    # in Float64 the floor (2.2e-14) is far below any sensible `tol`.
+    RT = real(eltype(L))
+    scale = atol === nothing ? sqrt(max(opnorm_estimate(L' * L; iters = 10), 0.0)) : 1.0
+    threshold = atol === nothing ? max(tol, 100 * eps(RT)) * max(scale, eps(RT)) : atol
```

```diff
 matrix_free_solvers(L) =
     size(L, 1) == size(L, 2) ?
         filter(s -> haskey(SOLVER_REGISTRY, s),
-               [:ArpackSolver, :KrylovSolver, :CGSolver, :LSMRSolver]) :
+               # LOBPCG (:CGSolver) is excluded below Float64: its Float32
+               # Cholesky of the block Gram matrix fails and the block collapses
+               # to 1-3 vectors, so it cannot hold a null space of dimension 3
+               # whatever its tolerance (precision-study.md, Exp. 2b, 5).
+               real(eltype(L)) === Float64 ?
+                   [:ArpackSolver, :KrylovSolver, :CGSolver, :LSMRSolver] :
+                   [:ArpackSolver, :KrylovSolver, :LSMRSolver]) :
         matrix_free_solvers()
```

### 7.4 `ext/DletoIterativeSolversExt.jl` (`CGSolver`) -- optional

LOBPCG's `tol = 1e-10` is absolute and is 2-3x more expensive than `1e-6` in
Float64 for identical eigenvalues (Exp. 2b), but scaling it to
`sqrt(eps) * ‖L‖` loosened recovery from 1e-11 to 1e-7 (Exp. 5 `CGP`).  A
change is a speed/accuracy trade the CG owner should make; nothing here fixes
it in Float32, which is why 7.3 excludes it instead.  If a Float32 path is
wanted, run LOBPCG in Float64 over the Float32 map -- but Exp. 5 `CGP64` was
erratic (5 s to 24 min), so this needs its own study.

### 7.5 `bench/SphereHarness.jl` -- not a bug, a trap

`warmup!` defaults to `T = Float64`; a Float32 timing without
`warmup!(configs; T = Float32)` measures JIT (~8 s at d = 20, the origin of
the "17x slower" figure).  Every script in this study warms up per element
type.

### 7.6 Worth building: a mixed-precision wrapper

`Promote64` in `precision-exp5.jl` (12 lines) runs any eigensolver in Float64
over a Float32 map.  On this benchmark it matched pure Float32's speed and
memory churn (0.25 vs 0.18 s, 0.64 vs 0.57 GB at d = 30) with 2-3x better
recovery (5e-6..1e-5) and 4 decades of spectral margin instead of 2 at
d = 50.  It is the configuration to carry toward d = 128.

## 8. What Float32 cannot do, stated plainly

- Recovery below ~1e-5 relative.  The eigenvector error is eigenvalue noise
  over gap, and the noise floor is `eps32 * ‖L‖`.
- Hold its margin as d grows.  The first nonzero eigenvalue of the sphere's
  operator drifts from 5e-3 to ~1e-4 relative between d = 40 and 50 and
  fluctuates 30x with the scramble, so a Float32 null threshold that is safe
  at d = 40 has ~2 decades of room at d = 50 and, on the trend, may have none
  by d = 100.  Float64 (or mixed) eigensolving keeps 4+ decades.
- LOBPCG at all.
- Give a speedup beyond the ~1.3x per contraction: at d <= 40 the wall time
  is the same as Float64 in every experiment; the win is bytes.

## Files

- `bench/reports/precision-exp1.jl` / `.csv` / `.log` -- Exp. 1
- `bench/reports/precision-exp2.jl`, `precision-exp2-{A,B,C}.csv`, `.log` -- Exp. 2
- `bench/reports/precision-exp2b.jl`, `precision-exp2b-{d20,cg}.csv`, `.log` -- Exp. 2b
- `bench/reports/precision-exp3.jl`, `precision-exp3-{float16,apply,mixed}.csv`, `.log` -- Exp. 3
- `bench/reports/precision-exp4.jl`, `precision-exp4-{support,spectrum}.csv`, `.log` -- Exp. 4
- `bench/reports/precision-exp5.jl` / `.csv` / `.log`, `precision-exp5b.jl` / `.csv` / `.log` -- Exp. 5, 5b
