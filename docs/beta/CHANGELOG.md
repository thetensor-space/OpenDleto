# OpenDleto `beta` — change log against `main`

Written 2026-09-08 from the commit history `main..beta` (113 commits, 2026-09-02 to
2026-09-04), the beta tags, and `docs/CONTEXT.md`. Companion documents:
[REVIEW.md](REVIEW.md) (the code review), [DESIGN-CHOICES.md](DESIGN-CHOICES.md) (the
design decisions and whether they hold up).

`main` is at `40b20d5` (2026-08-04, "Cleaned up labels"). `beta` is at `2d87dc3`, tagged
`v1.5-beta-2026-09-04`. `beta` contains `main` entirely; nothing on `main` is missing from
`beta`. Source-tree size of the change: 40 files under `src/ ext/ test/`, +11,713 / −638 lines.
The whole diff (with `bench/`, `labs/`, `docs/`) is 184 files and about 515k inserted lines,
of which ~86k are one committed notebook (`labs/WWEIA2.ipynb`) and most of the rest are CSV
and log outputs under `bench/`.

---

## 0. What `main` was

A working but thin package. One derivation solver (`SylverLining`, in a file misspelled
`SylverLininig.jl`), one `NullSolvers.jl` in which five of seven named solvers raised
`UndefVarError`, `den` an `@assert false` stub, `der` exported but undefined, `Random` a
weak dependency so `TestFastDer3Valent.jl` never ran, `TestSylverLining.jl` and
`TestDerivations.jl` commented out of `runtests.jl`. The test suite errored after 2,460
passes.

---

## 1. Summary of what `beta` adds, by theme

| Theme | What landed | Where |
|---|---|---|
| **Correctness repairs** | `den` implemented; `der` defined; five broken null solvers fixed; `FastDer3Valent` re-transcribed from the reference; dead code and dead exports removed; `nd` truncation bug; trivial Z-sets no longer errors; `stratify` retag bug; `ivec` default | `src/Densors.jl`, `src/Derivations.jl`, `src/solvers/NullSolvers.jl`, `src/solvers/FastDer3Valent.jl`, `src/chisels/ChiselImpls.jl` |
| **New derivation methods** | `QuickDer` (any valence: sketch-restrict / solve / lift / verify), `QuickSylver`, `AutoDer` (`:Auto`, now `stratify`'s default) | `src/solvers/QuickDerN.jl`, `QuickSylver.jl`, `AutoDer.jl` |
| **Null-solver layer** | Solver registry; `AutoSolver`, `SVDSolver`, `LUSolver`, `ShiftInvertSolver`, `GramSolver`, `LSMRSolver`; byte-based dense gate; adaptive `nv` escalation; `NullVerdict` with `status`; seeded solves | `src/solvers/NullSolvers.jl`, `ext/Dleto{Arpack,KrylovKit,IterativeSolvers}Ext.jl` |
| **Precision policy** | One module deciding what "zero" means per element type; `data_floor`, `precision_floor`, `qd_tolerance`, `iter_tol`, `rank_rtol`, tuned by experiment (`FLOOR_EPS = 8`) | `src/solvers/Precision.jl`, `docs/design/Precision-Policy.md` |
| **Diagnostics** | `DerivationReport` (the verdict as a value, `return_diagnostics = true`); `Dleto.der_residual` (the Z-law check) in the library; tagged progress reporting | `src/solvers/DerivationReport.jl`, `SolverProgress.jl` |
| **GPU (Apple Metal)** | `Metal` weakdep extension: device hooks, `sylvesterLM(backend = :metal)`, QuickDer `device = :gpu` (hybrid), mode products without a full permute | `ext/DletoMetal*.jl` |
| **Memory** | Mode-product kernels in one buffer, trivial derivation space factored, `_qdn_restricted_map` scratch allocated once, lean sphere harness | `src/solvers/QuickDerN.jl`, `bench/SphereHarness.jl` |
| **Tests** | 9 new test files wired into `runtests.jl`; equational laws (Z-law, T-law, Galois adjunction); oracles at valence 3/4/5; verdict, seed and precision tests | `test/` |
| **Docs and process** | `docs/CONTEXT.md` running record; design notes; review vs Magma; refactor plan; `bench/jl` budgeted wrapper; beta as an approval gate with tags | `docs/`, `bench/jl` |

---

## 2. Change log by beta tag

Tags mark reviewed merges into `beta`. Each section lists the non-merge commits that
entered with that tag, grouped, with the user-visible effect.

### `v1.1-beta-2026-09-04` — the refactor baseline, QuickDer-n, the GPU skeleton
(`main..c0aea13`, 47 commits; 13,101 tests over 29 testsets on Julia 1.12.3)

**Repairs to what `main` shipped**

- `Project.toml`: `Random` moved from `[weakdeps]` to `[deps]` and to the test target;
  `ProgressMeter` (weakdep with no extension) dropped; `SparseArrays` added. Result: the test
  suite runs to completion for the first time. (`6fa142e`)
- `src/SylverLining/SylverLininig.jl` → `SylverLining.jl`; include path fixed. (`2159e63`)
- `ChiselFramed` renamed to `Chisel`; its dead helpers (undefined `Fch`, `enggaged`)
  repaired; the phantom `export der` removed until `der` existed. (`c86ece9`)
- `SylverLining` rewrote `nd <= 0` to the valency and returned 3 of 6 (or 3 of 38)
  derivations. Fixed; regression oracle is the diagonal tensor's `2n`. (`2f84350`)
- `FastDer3Valent` solved the wrong chisel (third slot must be negated) and its system
  matrix was wrong in all three blocks behind a masking heuristic. Replaced by a
  line-for-line transcription of the reference, with its solution verification restored.
  (`4296005`, `703475e`)
- Symmetry solving, dead exports, and "trivial Z-set is an error" fixed. (`4a55e3d`)
- `TransverseOpsSymmetries(fr, localOp)` referenced an undefined variable. (`2aee31f`)
- `stratify`'s `ivec` default changed from 0 (a fixed column) to −1 (random). (`c5215dd`)
- Operator-layer annotations widened `Vector{<:Number}` → `AbstractVector{<:Number}` so
  block solvers passing views stop dying in `unsafe_embedITensorsSwapped`. (`0a49ba6`)
- Scratch files archived then removed. (`45909b8`, `776fd01`)

**New capability**

- `der` (Z-set) with tiers `der`, `derReduced`, `derTrOps`, `derTrOpsReduced` and a symbol
  form for every partial setting. (`eb825bc`)
- `den` (T-set / densor) implemented as the transpose of the `sylve`/`ester` pair; `denLM`
  is a rectangular `LinearMap` with a genuine adjoint. `den` no longer densifies
  (previously 14 GB at `n = 19`). (`0c5b9bb`, `1f0b400`, `1384850`)
- `QuickSylver` ported: double-restriction solve-and-lift for adjoint chisels. (`c81ce09`)
- **`QuickDer` for any valence**: restrict by sketching every output axis with a random
  orthogonal `W_a`, solve the restricted system with `solve_nullspace`, lift by one
  least-squares per axis, filter for consistency, verify by the Z-law on random slices.
  65× faster than SylverLining at 30³, 152× at 16⁴. (`7f3b77d`)
- `AutoDer` (`:Auto`): QuickDer when the setting allows it, SylverLining otherwise or on
  failure. `stratify` defaults to `:Auto`. (`a7842fc`)
- `sylvesterLM` over plain arrays: zero-allocation apply, sparse branch, `backend =
  :auto/:array/:itensor`; bit-identical to the ITensor path. (`024c15d`)
- `GramSolver` for QuickDer's dense branch (Gram + shifted Cholesky, oversampled subspace
  iteration, Rayleigh–Ritz on the unsquared matrix): 3 s vs 52 s SVD at `d = 100`.
  Dense budget 2.5 GB so the Gram route covers `d ≤ 200` at valence 3. (`e61ecf8`, `c747d15`)
- **Null-solver layer centralised** in `solve_nullspace`: `SOLVER_REGISTRY` /
  `register_solver!` / `available_solvers` (extension solvers register in `__init__`);
  `dense_is_cheap` gates on bytes; `AutoSolver`; adaptive `nv` escalation; `wants_square`
  trait; `ShiftInvertSolver` with a relative shift; relative tolerance; `LSMRSolver`
  (null space by projection, never squaring) — the only solver correct at `n = 19` for
  `den`. Iterative solvers' counts confirmed with a doubled request before being trusted.
  (`1384850`, `d31a532`, `a5a7260`, `b5f43bb`, `e9a2522`, `707cc55`)
- Optional tagged progress reporting (`progress = true / :densify / [:solve]`). (`a5a7260`)
- **Metal extension**: device hooks as `Ref`s set in `__init__` (an extension may only add
  methods); `sylvesterLM(backend = :metal)` 6–17× on dense tensors; QuickDer `device = :gpu`
  hybrid (Gram and `M*X` on device, factorizations on host), 2.6× end to end at `d = 200`.
  (`ef0249a`, `f60ad2b`, `06702ea`)

**Tests, bench, docs**

- `TestSylverLining.jl` ported and re-enabled; `TestDerivationLaws.jl` (Z-law, T-law, Galois
  adjunction, `denLM` adjoint, stratification as change of frame); QuickDer oracle tests at
  valence 3/4/5; `TestAutoDer.jl`. (`9132b97`, `6e881be`, `c9d1534`)
- `bench/jl`: the budgeted Julia wrapper (slot locks, thread and heap caps, RSS watchdog,
  real binary instead of the juliaup shim). (`c9d1534`, `3120277`)
- Benchmarks: `ChiselOperationBench`, `DerivationSolverAudit`, `StratifyScaling`,
  hypersphere harness, `QuickDerScaling`, `QuickDerLargeD`, `Frontier`, CPU frontier CSVs.
- Docs: `CONTEXT.md`, `Dleto-Design.md`, `review/OpenDleto-vs-Magma.md`,
  `review/Refactor-Plan.md`, `design/QuickDer-valence-n.md`, `design/Native-Core-Plan.md`
  (decision: do not port to a native core now).

### `v1.2-beta-2026-09-04` — whitened restriction, precision policy, movie measurements
(`c0aea13..3071ea7`, 20 commits)

- **QuickDer-W, the whitened restriction** (`whiten = true` by default, inherited by
  `:Auto`). Diagnosis: the matrix-free restricted branch was not slow but *not converging*
  (ARPACK hit its cap and returned nullity 0 at every `d ≥ 30` on the scrambled sphere).
  Thin QR of each mode unfolding makes every diagonal Gram block a multiple of the identity
  without touching the null space. Frontier moved from `d = 200` to `d = 500` at valence 3
  and to `d = 100` at valence 4. Also fixes an undercount on rank-deficient modes: trivial
  derivations are written down exactly instead of left as a near-zero cluster.
  (`d132e17`, `b6c59d9`, `047b397`)
- **`Precision.jl`**: one floating-point policy — `compute_eltype` (Float16 → Float32
  arithmetic), `precision_floor`, `data_floor`, `tol_default`, `iter_tol`, `rank_rtol`,
  `qd_tolerance`, `precision_policy` — tuned by the `bench/reports/precision-*` experiments;
  `FLOOR_EPS` raised 5 → 8 for the Float32 frontier. `TestPrecision.jl` parametrised over
  the three types. (`c9185ea`, `0bede3e`, `a10c405`)
- Movie regime measured on `640×480×F×3`; direction set: the movie runs in Float16, memory
  first, then GPU. `labs/MovieRuntime.{jl,ipynb}`. (`6a70742`, `07df754`)
- `d = 500` dense-vs-sparse, CPU-vs-GPU matrix. (`c01375d`, `da24cc4`)
- Docs: `design/Precision-Policy.md`, `design/Prior-Art-Large-d.md`,
  `design/Deployment-Plan.md`; whitened-restriction report.

### `v1.3-beta-2026-09-04` — lean memory, solver honesty, GPU tensor stages
(`3071ea7..cf5ab19`, 17 commits)

- **Memory**: `_qdn_ttm` never permutes its input; `_qdn_ttm!` (two buffers) and
  `_qdn_ttm_square!` (one buffer); trivial derivation space published factored
  (`QDN_TRIVIAL_FACTORED`, capped by `QDN_TRIVIAL_MAX_BYTES`); `_qdn_restricted_map` scratch
  allocated once (a 100k-apply solve had been churning ~700 GB). Frontier: valence 3
  `d = 1000` in 552 s at 13.7 GB; valence 4 `d = 200` in 181 s. Video shape
  `640×480×300×3` Float32: killed at 13.1 GB before, 4.06 GB after. (`b7cf5ae`, `1a56eef`,
  `2dfe948`, `3c5fe08`)
- **Honesty**: a non-converging iterative solve now reports `status` (`:ok` / `:unconverged`
  / `:capped`) instead of an empty null space; `_qdn_solve_and_lift` declines a non-`:ok`
  solve so `:Auto` falls back. (`2d9f28d`)
- Lift consistency filter cuts on a **gap** (floor `sqrt(eps(T))`, ceiling 32×) rather than
  at a hard cutoff. (`6f4d23b`)
- The 15× RSS on video shapes was the *benchmark's* own Z-law check promoting Float32 to
  Float64 through `applyDerivation`; replaced by a blocked accumulation in the tensor's own
  type. (`c96100c`)
- QuickDer device path: cost-model the mode product (`_qdn_slab_is_cheap`), drop the
  full-tensor permute. (`5d03c9d`)
- Float16 vs Float32 on Metal measured: mixed GEMM accepted, ~1× throughput, fp32
  accumulation. `docs/design/Float16-Metal.md`. (`a2cfb5c`, `0beb4a0`)
- **Correction on record**: the movie's frame-linear cost is the restricted eigensolve
  (the restriction grows with frame count), not the tensor stages (1–5 %). The earlier
  attribution in CONTEXT is superseded by the later section. (`e8b161e`, `9c6279c`)
- `bench/WhitenedRestriction.jl` silently ran Float64 when asked for Float16; fixed. (`dbebe62`)

### `v1.4-beta-2026-09-04` — seeded solves, the Float32 cut, status vetoes certification
(`cf5ab19..aa9fd6e`, 5 commits)

- **Seeded null solves**: `solve_nullspace(...; seed)` fixes ARPACK's start vector,
  KrylovKit's block and LOBPCG's `X0`; `opnorm_estimate` takes an `rng` too.
  `QuickDerMethod`'s seed travels to the restricted solve. Determinism exposed that the
  restricted solve loses one copy of a multiple eigenvalue on ~1 case in 8; recorded, not
  yet fixed. (`62cd5bc`)
- **The Float32 undercount** was a hard cutoff at the noise level in
  `_fastder_tall_nullspace`; now a gap cut with `qd_tolerance(T)` as floor and
  `FASTDER_RESTRICT_CEILING = 32×` as ceiling. Float32 is 3/3 at `d = 48..140`. (`26809a3`)
- **A non-ok `status` cannot certify**, and an empty answer from one is a failure; the
  KrylovKit Arnoldi fallback never claims convergence; `_qdn_empty_result` refuses an empty
  space from a non-`:ok` solve. (`bcf0f74`)
- **`Dleto.der_residual`** moves from `bench/` into `src/`: the Z-law check a consumer runs
  on an answer; blocked, bounded by `block_bytes`, no promotion. (`3772347`)

### `v1.5-beta-2026-09-04` — the stored type reaches the verdict, DerivationReport, `:fixed_nd`
(`aa9fd6e..7bc52d9`, 5 commits; 13,770 tests over 45 testsets)

- **Float16 false certificate fixed**: `derTrOpsReduced(::QuickDerMethod, ...)` promotes
  Float16 to Float32 before the kernel, so `data_floor(Float16)` never bound and the verdict
  certified a cut inside the input's rounding. `store` is now an explicit positional
  argument of `_qdn_solve_and_lift`. (`544ce9b`)
- **`DerivationReport`**: `derTrOpsReduced(...; return_diagnostics = true)` returns a
  fourth element carrying the deciding `NullVerdict`, both element types, restriction
  sizes, solver/device/seed/whitened, lift residuals and Z-law residuals. Default `false`;
  the three-tuple is unchanged. Same keyword on QuickDer, SylverLining, AutoDer. (`0c398ff`,
  `647ea21`)
- **`nd > 0` is a policy** (`policy = :fixed_nd`), not a cap: asks for the `nd` smallest by
  singular value, skips the Z-law check, reports residuals per direction; `certified` only
  if the automatic verdict would have cut at exactly `nd`. (`523723d`)

### `v1.5-beta-2026-09-04..beta`

Only the merge commit `2d87dc3`; `beta` HEAD is the tagged state.

---

## 3. Public API delta (exports)

New exports on `beta` that `main` did not have:

```
compute_eltype, precision_floor, data_floor, tol_default, iter_tol,
rank_rtol, qd_tolerance, precision_policy
NullSolver, solve_nullspace, available_solvers, register_solver!,
AutoSolver, SVDSolver, LUSolver, ShiftInvertSolver
PROGRESS_TAGS, progress_spec
der, den, derReduced, derTrOps, derTrOpsReduced, get_derivation_method
DerivationReport
FastDer3ValentMethod, QuickSylverMethod, QuickDerMethod, AutoDerMethod
denLM
gpu_available, to_gpu, to_cpu, gpu_sync
```

Renamed: `ChiselFramed` → `Chisel`. Removed: a phantom `export der` on `main` (now a real
definition). Not exported but public by documentation: `Dleto.der_residual`,
`Dleto.QDN_APPLY_COUNT`, the `QDN_*` tunables, `GramSolver`, `LSMRSolver` (registered by
symbol).

Behaviour changes a `main` user would notice:

- `stratify` defaults to `:Auto` (QuickDer first) instead of SylverLining, and to a random
  `ivec`.
- `nd <= 0` means "a basis" everywhere; it no longer silently truncates to the valency.
- `den` works, and does not densify.
- Every derivation route can return a `DerivationReport`.
- A trivial derivation algebra is a result, not an error.

---

## 4. Dependency delta

| | `main` | `beta` |
|---|---|---|
| `[deps]` added | | `Random`, `SparseArrays` |
| `[weakdeps]` | `Arpack`, `ProgressMeter`, `Random` | `Arpack`, `Metal` |
| `[extensions]` | Arpack, IterativeSolvers, KrylovKit, Plots | + `DletoMetalExt` |
| test target | `Test` | `Random`, `Test` |
| version | 0.1.0 | 0.1.0 (unchanged; tags carry the version) |

Unchanged and worth noting: `IJulia`, `PlotlyJS`, `PlotlyKaleido`, `Plots`, `CSV`,
`DataFrames`, `JSON` remain hard dependencies; `KrylovKit` and `IterativeSolvers` are hard
dependencies that are also extension triggers. See [REVIEW.md](REVIEW.md) §package spine.

---

## 5. Numbers on record (from CONTEXT and bench reports, all Float64 CPU unless stated)

| Case | `main` | `beta` |
|---|---|---|
| Test suite | errors after 2,460 passes | see [REVIEW.md](REVIEW.md) for the run made for this review |
| Scrambled sphere, valence 3, `d = 200` | out of reach | 18.4 s, 2.1 GB (whitened, matrix-free) |
| Scrambled sphere, valence 3, `d = 1000` | — | 552 s, 13.7 GB, nullity 3, residual 7.2e-13 |
| Scrambled sphere, valence 4, `d = 200` | — | 181 s, 17.35 GB, nullity 4 |
| `den` at `n = 19` (map 260642×6859) | 13.3 GB dense, not runnable | 131 s matrix-free, 19 of 19, 1.1e-14 |
| Video `640×480×300×3` Float32 | — | 34.8 s, 5.4 GB peak, nullity 3 certified |
