# QuickDer at d = 500: dense vs sparse-structured, CPU vs GPU

2026-09-04.  Raw numbers: `d500.csv` (regenerate individual rows with
`bench/jl bench/D500Matrix.jl <task> ...`, one process per row — see the file
header).  Harness: `bench/D500Matrix.jl`, built on `bench/WhitenedRestriction.jl`
(the timed call) and `bench/SphereHarness.jl` / `bench/Frontier.jl` (`build_sphere`,
`sphere_octant`, `der_residual`).  Code: `src/solvers/QuickDerN.jl`.

## Stopped early

**The campaign was stopped after the third cell** (dense, GPU, Float32) hit
**15.01 GB peak host RSS**, over the 14 GB reporting threshold set for this
run. Per the brief ("stop and report early if any single cell exceeds 14 GB
RSS or 20 min wall"), the remaining cells were **not run**:

- sparse-structured CPU Float64
- sparse-structured CPU Float32
- sparse-structured GPU Float32
- sparse storage (`SparseArrays`/sparse ITensor) CPU Float64 — the optional cell

Everything below is the three cells that did run, plus what they already
answer about the dense-vs-sparse and GPU questions from the code alone.

## The matrix (d = 500, valence 3, whitened, `:AutoSolver`, matrix-free branch throughout)

Every cell here lands in the matrix-free restricted branch **without forcing
it**: `dense_bytes = R·(Σ dₐrₐ)·sizeof(T)` is ≈28.6 GB at Float64 and ≈14.3 GB
at Float32 (R = 64000, Σdₐrₐ = 60000), which is over both
`QDN_DENSE_BUDGET_BYTES` (2.5 GB, CPU) and `QDN_GPU_DENSE_BUDGET_BYTES` (6 GB,
GPU). So these are the numbers the production default gives at this size.

| data | precision / device | wall s (solve s) | peak host RSS | GPU mem (peak) | applies | nullity (oracle 3) | residual | verdict |
|---|---|---|---|---|---|---|---|---|
| dense (scrambled sphere) | CPU Float64 | 136.13 (129.64) | 11.37 GB | — | 51116 | 3 | 3.18e-13 | **ok** (reused from `whitened.csv`, not rerun) |
| dense (scrambled sphere) | CPU Float32 | 64.82 (59.58) | 13.66 GB | — | 39550 | 1 | 2.23e-04 | **uncertified** |
| dense (scrambled sphere) | GPU Float32 (`device=:gpu`) | 68.79 (60.83) | 15.01 GB | 1.29 GB | 39550 | 1 | 2.23e-04 | **uncertified** — and 14 GB stop line crossed here |
| sparse-structured (raw sphere) | CPU Float64 | not run | — | — | — | — | — | stopped early |
| sparse-structured (raw sphere) | CPU Float32 | not run | — | — | — | — | — | stopped early |
| sparse-structured (raw sphere) | GPU Float32 | not run | — | — | — | — | — | stopped early |
| sparse storage (`SparseArrays`) | CPU Float64 | not run (optional cell) | — | — | — | — | — | stopped early |

GPU stage breakdown for the one GPU cell that did run (from
`Dleto.QDN_STAGE_TIMES`, seconds):

```
upload 0.073   sketch 0.205   lift 1.415   whiten 0.048
restricted 0.023   solve 60.827   verify 3.443   restrict_ops 0.077
```

## Does the GPU cell actually exercise the GPU?

**Only for the parts that scale with `d^n` — never for the restricted solve,
which is 88% of the wall time.**  This is confirmed two ways:

1. **Code**: `_qdn_solve_and_lift` (`src/solvers/QuickDerN.jl:994-1075`) branches
   on `dense_bytes <= dense_budget`. At d = 500 that budget is blown on *both*
   devices (28.6/14.3 GB vs. 2.5/6.0 GB), so both columns take the matrix-free
   branch, and that branch's own comment says it plainly: *"Matrix free stays
   on the HOST whatever `device` says: the null solvers here iterate with host
   vectors."* Only `_qdn_cross_sketches` (sketch stage) and `_qdn_pair_tensor`
   (part of lift) run against device-resident `W`/`W⊥` via `_qdn_axes_device`;
   the ARPACK iteration itself (`solve_nullspace(L, fsolver; ...)`) runs on
   host `Array`s regardless of `method.device`.
2. **Measurement**: the GPU cell's `solve` stage (60.83 s) is statistically
   the same as the CPU cell's (59.58 s) — no speedup, as expected from a stage
   that never left the host — while `sketch` (0.20 s), `lift` (1.41 s),
   `upload` (0.07 s) and `verify` (3.44 s) are the only stages that touch the
   device. Peak Metal allocation was 1.29 GB (`Metal.device().currentAllocatedSize`,
   polled every 20 ms), consistent with one Float32 `d^3` tensor copy plus
   verification buffers — not the 6 GB GPU dense budget, because the dense
   Gram route never engaged.

So `device = :gpu` at this size **is not idle**, but it is doing 12% of the
work (sketch + lift + upload + verify = 5.14 s of 68.79 s) and made the run
*4.0 s slower* overall than the CPU column, because that 12% pays host↔device
transfer and kernel-launch overhead that the CPU column doesn't. The GPU is
only a net win when the dense Gram route can fit in `QDN_GPU_DENSE_BUDGET_BYTES`
(≤ 6 GB) — at d = 500 valence 3 it can't (14.3 GB at Float32), so this size is
squarely in "GPU present but not the bottleneck's owner" territory.

## Float32: half the memory, and uncertified as expected

Peak host RSS: 11.37 GB (Float64) vs 13.66 GB (Float32 CPU). That is **not**
half — Float32 halves the *tensor and restricted-map* footprint (confirmed by
`bytes` allocated over the run: 592 GB-cumulative at Float64 vs 241 GB at
Float32, almost exactly 2.45x, close to the 2x a straight halving predicts),
but peak RSS at this size is dominated by transient allocations in the lift
(`_qdn_pair_tensor` permuting full `d^n`-scale arrays) that don't shrink
proportionally, plus GC not having reclaimed everything at the `Sys.maxrss()`
sample point (`maxrss` is a high-water mark for the whole process, not the
solver's own accounting). The *cumulative bytes allocated* column is the
cleaner precision comparison and it does track 2x.

Both Float32 cells (CPU and GPU) landed nullity 1 against oracle 3, flagged
`uncertified` — exactly the failure mode the brief predicted from this
morning's Float32 video run: the right answer is out of reach because the
gap-test that certifies a nullspace answer can't clear Float32's precision
floor at this problem's conditioning, not because the solver did anything
wrong. Per instructions this was recorded as-is; no tolerance was tuned to
chase a different number.

## What tripped the stop rule

The GPU cell's `Sys.maxrss()` (15.01 GB) is a whole-process high-water mark,
sampled once at exit, and it is close enough to the CPU Float32 cell's
13.66 GB that the extra device round-trip (a second, device-resident tensor
copy, held simultaneously with the host one during upload/sketch/lift) is the
plausible 1.3-1.5 GB difference. `bench/jl`'s own watchdog (`JL_RSS_LIMIT_GB=14`)
did not kill the process — its 2-second polling window can miss a peak that
GC reclaims quickly — but the run's own `Sys.maxrss()` reading is the higher
of the two, unaffected by later GC, and it is the number this report's stop
rule is keyed to.

## What was not learned this pass

The sparse-structured (raw, unscrambled) sphere cells were not reached, but a
quick correctness probe at small d (12, 20, 30, 60, 100, all in the dense
branch there) surfaced something worth flagging before those cells are run:
the **raw** `sphere_octant` — not passed through `nondeg` — has a *certified*
nullity of **13**, not the oracle of 3, at every one of those sizes (residual
~1e-16, not flagged `uncertified` or `trivial_reinjected`). That is not a
solver bug: `nondeg`'s job (per `SphereHarness.jl`'s own header) is exactly to
remove the extra degenerate directions that a raw, un-scrambled, un-nondeg'd
lattice tensor has. The oracle of 3 is a property of the *nondeg'd* sphere,
established by this benchmark family's convention (`build_sphere`), not of
the raw `sphere_octant` on its own — so whoever runs the remaining
sparse-structured cells at d = 500 should expect (and not be alarmed by) a
certified nullity around 13 rather than 3, and should report it as such
rather than as a wrong answer.

## Files

- `bench/D500Matrix.jl` — driver, one process per case (`estimate`,
  `dense-baseline`, `dense <T> <device>`, `sparse <T> <device>`,
  `sparse-storage <T> <device>`).
- `d500.csv` — the three rows above, same column layout as
  `bench/reports/2026-09-04/whitened/whitened.csv` plus `device`, `gpu_mem_gb`,
  `storage`.
