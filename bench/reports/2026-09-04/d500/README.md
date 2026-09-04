# QuickDer at d = 500: dense vs sparse-structured, CPU vs GPU

2026-09-04.  Raw numbers: `d500.csv` (regenerate individual rows with
`bench/jl bench/D500Matrix.jl <task> ...`, one process per row — see the file
header).  Harness: `bench/D500Matrix.jl`, built on `bench/WhitenedRestriction.jl`
(the timed call) and `bench/SphereHarness.jl` / `bench/Frontier.jl` (`build_sphere`,
`sphere_octant`, `der_residual`).  Code: `src/solvers/QuickDerN.jl`.

## Campaign notes

The first pass stopped after the GPU dense cell hit 15.01 GB peak RSS (over
this report's original 14 GB line). The parent raised the stop line to 18 GB
for the resumed run (the 15 GB seen is the same 500³ tensor plus its device
upload copy, inside the parent's reservation) and the three sparse-structured
cells below completed under that line with no further stops. The optional
sparse-storage (`SparseArrays`) cell was dropped from the plan by the parent
and was not attempted.

## The matrix (d = 500, valence 3, whitened, `:AutoSolver`, matrix-free branch throughout)

Every cell here lands in the matrix-free restricted branch **without forcing
it**: `dense_bytes = R·(Σ dₐrₐ)·sizeof(T)` is ≈28.6 GB at Float64 and ≈14.3 GB
at Float32 (R = 64000, Σdₐrₐ = 60000), which is over both
`QDN_DENSE_BUDGET_BYTES` (2.5 GB, CPU) and `QDN_GPU_DENSE_BUDGET_BYTES` (6 GB,
GPU). So these are the numbers the production default gives at this size.

**Oracle note for the sparse-structured rows.** The oracle is **13**, not 3.
A small-d probe (d = 12..100, all landing in the dense branch there) found
that the raw `sphere_octant` — never passed through `nondeg` — has a
*certified* nullity of 13 at every size tried (residual ~1e-16, not
`uncertified`, not `trivial_reinjected`). That is not a solver bug: per
`SphereHarness.jl`'s own header, `nondeg`'s job is exactly to remove the extra
degenerate directions a raw, un-scrambled lattice tensor has. The oracle of 3
is a property of the *nondeg'd* sphere (`build_sphere`), established by this
benchmark family's convention — not of `sphere_octant` on its own. The d = 500
sparse-structured rows below confirm the same 13 at this size.

| data | precision / device | wall s (solve s) | peak host RSS | GPU mem (peak) | applies | nullity (oracle) | residual | verdict |
|---|---|---|---|---|---|---|---|---|
| dense (scrambled sphere) | CPU Float64 | 136.13 (129.64) | 11.37 GB | — | 51116 | 3 / 3 | 3.18e-13 | **ok** (reused from `whitened.csv`, not rerun) |
| dense (scrambled sphere) | CPU Float32 | 64.82 (59.58) | 13.66 GB | — | 39550 | 1 / 3 | 2.23e-04 | **uncertified** |
| dense (scrambled sphere) | GPU Float32 (`device=:gpu`) | 68.79 (60.83) | 15.01 GB | 1.29 GB | 39550 | 1 / 3 | 2.23e-04 | **uncertified** |
| sparse-structured (raw sphere) | CPU Float64 | 261.30 (255.31) | 7.98 GB | — | 85112 | 13 / 13 | 1.71e-15 | **ok** |
| sparse-structured (raw sphere) | CPU Float32 | 192.58 (187.66) | 8.97 GB | — | 98792 | 11 / 13 | 2.95e-06 | **uncertified** |
| sparse-structured (raw sphere) | GPU Float32 (`device=:gpu`) | 113.54 (105.77) | 9.56 GB | 1.65 GB | 68000 | 13 / 13 | 5.60e-08 | **uncertified** (right nullity, precision floor) |
| sparse storage (`SparseArrays`) | CPU Float64 | dropped from plan | — | — | — | — | — | not attempted (parent's call) |

GPU stage breakdown (from `Dleto.QDN_STAGE_TIMES`, seconds):

```
dense, GPU Float32:   upload 0.073  sketch 0.205  lift 1.415  whiten 0.048
                      restricted 0.023  solve 60.827  verify 3.443  restrict_ops 0.077
sparse, GPU Float32:  upload 0.071  sketch 0.050  lift 1.448  whiten 0.051
                      restricted 0.024  solve 105.768  verify 3.420  restrict_ops 0.034
```

## Does the GPU cell actually exercise the GPU?

**Only for the parts that scale with `d^n` — never for the restricted solve,
which is 88-93% of the wall time.** Confirmed two ways:

1. **Code**: `_qdn_solve_and_lift` (`src/solvers/QuickDerN.jl:994-1075`) branches
   on `dense_bytes <= dense_budget`. At d = 500 that budget is blown on *both*
   devices (28.6/14.3 GB vs. 2.5/6.0 GB), so both columns take the matrix-free
   branch, and that branch's own comment says it plainly: *"Matrix free stays
   on the HOST whatever `device` says: the null solvers here iterate with host
   vectors."* Only `_qdn_cross_sketches` (sketch stage) and `_qdn_pair_tensor`
   (part of lift) run against device-resident `W`/`W⊥` via `_qdn_axes_device`;
   the ARPACK iteration itself (`solve_nullspace(L, fsolver; ...)`) runs on
   host `Array`s regardless of `method.device`.
2. **Measurement**: on the dense pair, GPU `solve` (60.83s) ≈ CPU `solve`
   (59.58s) — no speedup, as expected from a stage that never left the host —
   while `sketch`+`lift`+`upload`+`verify` (5.14s of 68.79s) are the only
   stages that touch the device, and that overhead makes the GPU column
   *slower* overall than CPU (68.79s vs 64.82s). On the sparse pair the same
   pattern holds even more starkly: GPU `solve` (105.77s) is *slower* than the
   CPU sparse solve (187.66s / 255.31s is a different apply count, so not
   directly comparable seconds-for-seconds, but the applies-per-second rate on
   GPU's solve, 68000/105.77 ≈ 643/s, is close to CPU Float32's
   98792/187.66 ≈ 526/s and CPU Float64's 85112/255.31 ≈ 333/s — consistent
   with ARPACK iterating host vectors at roughly BLAS-thread speed on both
   devices, not device speed). Peak Metal allocation was 1.29-1.65 GB
   (`Metal.device().currentAllocatedSize`, polled every 20 ms), consistent
   with one Float32 `d^3` tensor copy plus verification buffers — not the 6 GB
   GPU dense budget, because the dense Gram route never engaged.

So `device = :gpu` at this size **is not idle**, but it only owns
sketch+lift+upload+verify — the stages that already cost single-digit seconds
on this problem. The GPU is only a net win when the dense Gram route fits in
`QDN_GPU_DENSE_BUDGET_BYTES` (≤ 6 GB); at d = 500 valence 3 it can't (14.3 GB
at Float32), so this size is squarely "GPU present but not the bottleneck's
owner" territory on both the dense and sparse-structured tensors.

## Dense vs sparse-structured: same code path, different memory and a different answer

Dense (scrambled) and sparse-structured (raw) hit the **same branch decision**
(matrix-free, same `r = [40,40,40]`, same restricted-system shape) because
that decision depends only on `dims`/`T`, never on the tensor's values. But
two things differ:

- **Peak RSS is lower for sparse-structured**, not because the algorithm
  exploits the zeros (it doesn't — both are dense `Array{T}` storage and every
  mode product is a full GEMM/TTM over all `d^n` entries) but because
  `build_sphere` (dense/scrambled path) holds three full tensor copies — the
  sphere, its orthogonal scramble, and the `nondeg`'d result — while the
  sparse-structured path builds `sphere_octant` once and hands it straight to
  the solver. At Float64: 11.37 GB (dense) vs 7.98 GB (sparse-structured); at
  Float32 CPU: 13.66 GB vs 8.97 GB. The gap (≈3.4-4.7 GB) is in the right
  ballpark for two extra 500³ Float64/Float32 arrays.
- **The certified answer is a different tensor property entirely.** Dense
  (nondeg'd) finds oracle 3; sparse-structured (raw) finds oracle 13, and the
  Float64 sparse-structured row (13/13, resid 1.71e-15) is exactly as
  certified as the Float64 dense row (3/3, resid 3.18e-13) — both are correct
  answers to different, legitimately posed problems, not a solver discrepancy.

## Float32: not quite half the memory, and uncertified as expected

Peak host RSS: dense 11.37 GB → 13.66 GB (Float64 → Float32 CPU, i.e. *higher*
at Float32); sparse-structured 7.98 GB → 8.97 GB (same direction). That is
counter to a naive "Float32 halves memory" expectation. The *cumulative bytes
allocated* column tracks precision much more cleanly: dense 592 GB → 241 GB
(2.45x), sparse-structured 967 GB → 571 GB (1.69x) — both real reductions, not
2x, because ARPACK needed more iterations at Float32 in the sparse-structured
case (98792 vs 85112 applies) which adds allocation back. Peak RSS itself is a
whole-process high-water mark dominated by transient allocations in the lift
stage (`_qdn_pair_tensor` permuting full `d^n`-scale arrays) that don't shrink
proportionally with `sizeof(T)`, plus GC not necessarily having reclaimed
everything at the `Sys.maxrss()` sample point — so it is the wrong column to
read precision savings off of at this size.

Three of the four Float32 cells came back `uncertified`:

- dense CPU/GPU Float32: nullity 1 (oracle 3), resid 2.23e-04
- sparse-structured CPU Float32: nullity 11 (oracle 13), resid 2.95e-06
- sparse-structured GPU Float32: nullity 13 (oracle 13, i.e. the RIGHT count),
  resid 5.60e-08, still flagged `uncertified` — the gap test that certifies a
  nullspace answer can't clear Float32's precision floor at this problem's
  conditioning even when the count comes out right, exactly the pattern this
  morning's Float32 video run showed. Per instructions this was recorded
  as-is; no tolerance was tuned to chase a different number.

## Files

- `bench/D500Matrix.jl` — driver, one process per case (`estimate`,
  `dense-baseline`, `dense <T> <device>`, `sparse <T> <device>`;
  `sparse-storage <T> <device>` also exists but was not used this campaign,
  see above).
- `d500.csv` — the six rows above, same column layout as
  `bench/reports/2026-09-04/whitened/whitened.csv` plus `device`, `gpu_mem_gb`,
  `storage`.
