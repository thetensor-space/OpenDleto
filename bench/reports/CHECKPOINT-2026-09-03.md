# Checkpoint: stratification solver study, 2026-09-03

Handoff note so another agent or person can continue. Branch
`signed-sealed-delivered/der-and-derivation-law`, nothing committed today yet.
Detailed reports live next to this file; memory notes for Claude sessions are
in `~/.claude/projects/-Users-algeboy-CODE-OpenDleto/memory/`.

## Goal (from the user)

Make Dleto's stratification of a sphere-octant tensor resilient and fast at
d = 30..40, then d = 40..50, eventually d = 128 (target under one minute) and
maybe 256. Recalibrate `AutoSolver` (the default null solver) from
measurements. Compare precision (Float64, Float32, Float16). Compare
`:QuickDer` against the null-solver path. Print live results; exclude JIT
warm-up from timings; always include Arpack.

## Where things stand

### Done and verified (tests passed after each; uncommitted)

| area | file(s) | what | report |
|---|---|---|---|
| Benchmark harness | `bench/SphereHarness.jl` | `build_sphere(d; T, seed, ops)`, `run_stratify(inp; solver/method, tol)`, `reconstruction` (least-squares recovery up to per-axis signed permutation), `warmup!` | header comments |
| Sweeps | `bench/StratifySolverProfile.jl` (all 10 solvers, standalone), `bench/StratifyLeadersProfile.jl` (leaders, uses harness; `--only=`, `--csv=`, `--tol=`, `--seeds=`) | results in `bench/stratify-solver-profile.csv`, `bench/stratify-leaders-profile.csv` | this dir |
| LUSolver | `src/solvers/NullSolvers.jl` (LU section) | was not rank revealing (partial-pivot LU), gave nullity 1 / wrong basis; now column-pivoted QR, 3-4x faster than SVD | `lu-solver-fix.md` |
| Krylov | `ext/DletoKrylovKitExt.jl` | nullity miss was the 3-fold eigenvalue 0 in single-vector Krylov; now block Lanczos (block = request, krylovdim >= 128, relative tol floored at 100 eps(T)) | `krylov-calibration.md` |
| Arpack | `ext/DletoArpackExt.jl` | nev >= 16, ncv = 8 nev, which=:SM, tol 1e-10 floored at 100 eps(T); 30/30 seeded runs correct | `krylov-calibration.md`, `precision-study.md` |
| AutoSolver | `src/solvers/NullSolvers.jl` | square maps densify only below N = 400 when an eigensolver is loaded; square order Arpack, Krylov, CG, LSMR (CG excluded below Float64); rectangular (`den`) unchanged; squares only if solver `wants_square`; `initial_request` trait; null threshold floored at 100 eps(T) | `krylov-calibration.md` sec. AutoSolver, `precision-study.md` sec. 7 |
| QuickDer | `src/solvers/FastDer3Valent.jl` | universal derivation space of the sphere is 13-dim (3 diagonal + 10 nilpotent, by degree grading); `derTrOpsReduced(::FastDer3ValentMethod, Ω, ...)` now intersects the universal basis with any valency-3 operator space Ω, so it matches the symmetric solvers (nullity 3, 4e-14); tolerances scale with eps(T) | `quickder-sphere.md` |
| Precision | measurements only | Float32: same wall time, 2.5x fewer bytes, recovery 1e-5..1e-4, loses spectral margin by d ~ 50; Float16 arithmetic impossible (no LAPACK), Float16 DATA fine (2e-4); mixed Float32 map + Float64 eigensolver is the configuration to carry forward | `precision-study.md` |

### In flight when this checkpoint was written

- **Gap-based nullity verdict** (Sonnet agent, editing `solve_nullspace` in
  `src/solvers/NullSolvers.jl`): a `NullVerdict` struct was drafted (see
  `git diff src/solvers/NullSolvers.jl`, section "the verdict"). Design: sort
  returned values, floor at 100 eps(T) relative, cut at the largest
  multiplicative jump if it exceeds `gap_ratio` (default ~1e3), else fall back
  to the threshold rule and mark uncertified; return `(; vals, vecs, verdict)`
  so `(vals, vecs) = solve_nullspace(...)` still works; warn when the floor is
  binding. Must be validated on d=50 seed 50 (genuine near-derivation at
  1.2e-8 relative: Float64 should certify nullity 3, Float32 must not silently
  return 4). Report goes to `gap-verdict.md`. If the agent did not finish:
  check the package loads, run `Pkg.test()`, and either finish or revert the
  verdict hunk before committing.

### Not started / paused

1. **Commit and push** (user asked for this next). Plan: commit A = source
   (`src/`, `ext/`, `test/`), commit B = `bench/SphereHarness.jl`,
   `bench/Stratify*.jl`, `bench/reports/`, `bench/stratify-*.csv`. Exclude
   `*.log` and `labs/WWEIA2.ipynb` (user's notebook). Run
   `julia -t 4 --project=. -e 'using Pkg; Pkg.test()'` first. Push to
   `origin` same branch. Commit trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
2. **AutoSolver vs QuickDer numbers** (user asked for this after the push):
   `julia -t 4 --project=. bench/StratifyLeadersProfile.jl 20 30 40 50 64 --seeds=2 --only=Auto/F64,QuickDer/F64 --csv=bench/reports/auto-vs-quickder.csv`
   on a quiet machine, then summarize seconds/bytes/lsq_err per d.
3. **Block (BLAS-3) apply in `sylvesterLM`** (`src/SylverLining/SylverLining.jl`):
   the lever for d = 128. Eigensolvers need ~500 applications regardless of d;
   each application is a few contractions costing ~d^2.8 and allocating
   5-10 MB. Profile stages, turn each axis contraction into one GEMM (universal
   chisel axis has length 1), preallocate, add an n x bs block apply for block
   Lanczos. Agent prompt is in the session transcript; nothing was edited.
4. Refine the dense gate to account for allocation churn of `Matrix(L)`
   (150 GB at d=64 caused swap stalls on the 16 GB machine).
5. `bench/StratifySolverProfile.jl` still duplicates harness code; could
   `include("SphereHarness.jl")`.

## Key numbers (quiet machine, 4 threads, Float64, tol 1e-8, `stratify-leaders-profile.csv`)

| d | Krylov | Arpack | Auto | QuickDer | LU-CPQR | SVD |
|---|---|---|---|---|---|---|
| 30 | 0.30 s | 0.25-0.34 s | 0.25-0.33 s | 0.28-0.33 s | 1.3 s | 1.6 s |
| 40 | 0.76-0.92 s | 0.74-0.98 s | 0.75-0.99 s | 0.64-0.77 s | 4.7 s | 6.2 s (400 s swapping) |
| 50 | 1.4-1.6 s | 1.8 s | 1.7 s | 2.3-2.4 s | 14-431 s | 20-113 s |
| 64 | 3.3-3.8 s | 3.6-8.3 s | 3.2-4.2 s | 6.1 s | 49-72 s | 73-835 s |

All recover the sphere (nullity 3, error <= 1e-11). QuickDer allocates 5-30x
less than anything else. Dense paths swap beyond d ~ 40 on 16 GB. Arpack
scaling ~d^3.26 extrapolates to 35-80 s at d = 128.

## Gotchas learned

- The literal SphereLab lattice sphere is a curve's worth of points and
  ill-posed after `nondeg` at these sizes; the harness samples x^2 equally
  spaced instead (exact, nullity 3). See `project-sphere-benchmark-construction` memory.
- Spectral gap (4th eigenvalue / top) is seed dependent: 2.5e-3 to 8e-5 at
  d=32; seed 50 at d=50 has 1.2e-8. Fixed tol 1e-6 is wrong; use 1e-8 or the
  gap verdict.
- `warmup!` defaults to Float64; a Float32 timing without `warmup!(cfgs; T=Float32)` measures JIT.
- `-t 2` does not cap BLAS threads; concurrent Julia jobs contend heavily.
  Only within-run ratios are trustworthy from the agent reports; the serial
  leaders sweep is the clean timing.
- Machine: 8 cores, 16 GB; dense `Matrix(L)` churn causes swap stalls.
- User policies: no Terminal.app windows (keep output inside VS Code: run in
  background, log to `bench/`, `code -r <log>`); subagents on Sonnet/Haiku,
  Fable only for the orchestrator; keep the user updated as results land.

## How to reproduce quickly

```
julia -t 4 --project=. -e 'using Pkg; Pkg.test()'
julia -t 4 --project=. bench/StratifyLeadersProfile.jl 20 30 40 --seeds=1 --only=Auto/F64,QuickDer/F64,Arpack/F64
```
