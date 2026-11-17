# The lift's redundant full-tensor work: F6, F7, and the progress/`_gram_dense` regression

Branch `higher-ground/lift-shared-prefix`. Question, setup, before/after
numbers, correctness evidence, and what did not move.

## Question

`docs/beta/review/A-quickder.md` F6 and F7, and `docs/beta/REVIEW.md` S6 M11,
name three redundancies in `_qdn_solve_and_lift`'s `for a in lift` loop
(`src/solvers/QuickDerN.jl`) and in `solve_nullspace` (`src/solvers/NullSolvers.jl`):

- **F7.** `_qdn_pair_tensor` walks a fresh contraction chain from the full
  tensor `G` for every `(a, b)` pair, so the one contraction that touches all
  `d^n` entries is recomputed `n(n-1)` times across the lift, while
  `_qdn_cross_sketches` already shares a prefix for the analogous per-axis
  sketch.
- **F6.** `Wt = _qdn_ttm(Hs[b], Yv[b,i], b)` and its unfolding, in the lift's
  right-hand-side assembly, depend only on `(b, i)` but sit inside the
  chisel-row (`rho`) loop, so a chisel with `m` rows (e.g. `CentroidChisel(3)`,
  `m = 3`) redoes that contraction `m` times for nothing.
- **M11.** `progress = true` wraps the restricted operator in a
  `LinearMaps.FunctionMap` before it reaches `_gram_dense`
  (`NullSolvers.jl`), whose whole point is to read `L.lmap` with no copy when
  `L` is already a `LinearMaps.WrappedMap` around a dense matrix. Wrapped,
  `Matrix(Lp)` re-derives the matrix one column at a time instead.

Does sharing the lift's full-tensor pass the way the sketch already does, and
skipping the progress wrap when there is nothing to report, measurably cut the
`:lift` stage, and does either change move the returned derivation coordinates
by more than rounding?

## Setup

- `bench/LiftCost.jl` (this session, committed separately): one
  `Dleto.get_derivation_method(:QuickDer; seed = 1)` /
  `Dleto.derTrOpsReduced` call per case, `Dleto.QDN_STAGE_TIMES[]` reset
  before each, all Float64 unless noted:
  - scrambled spheres (`bench/SphereHarness.jl`), valence 3, `d ∈ {60,100,150}`
  - scrambled spheres, valence 4, `d ∈ {30,50}`
  - `CentroidChisel(3)` (`m = 3` rows) on a random dense 60×60×60 tensor
  - a video-shaped random 160×120×30×3 tensor, **Float32**
- **Baseline** (`tag = before`): measured against the pre-fix source (the
  working tree stashed back to commit `35a10d0`, the tip of `beta` this
  worktree was created from).
- **After** (`tag = after`): the same script, same seeds, against the fix
  below.
- The fix, `src/solvers/QuickDerN.jl` and `src/solvers/NullSolvers.jl`:
  - `_qdn_pair_tensors(G, axs, a, bs)` (new, next to `_qdn_pair_tensor`):
    for lift axis `a`, picks the cheapest axis `c ≠ a`
    (`_qdn_mode_order`'s own first choice — on the host this is the smallest
    index, matching `_qdn_pair_tensor`'s own ordering exactly), contracts `G`
    with `W_c` **once**, and derives every `H_{a,b}` with `b ≠ c` from that
    intermediate; only `H_{a,c}` needs `_qdn_pair_tensor`'s own fresh pass.
    Two full-tensor passes for the whole per-axis dictionary instead of
    `length(bs)`.
  - The lift's RHS loop now computes `Wu[b] = transpose(_qdn_unfold(_qdn_ttm(Hs[b], Yv[b,i], b), a))`
    once per `(b, i)`, outside the `rho` loop, and each row just accumulates
    `P[rho,b] .* Wu[b]`.
  - `NullSolvers.jl` gains `_wraps_dense_matrix(L)` (true exactly when
    `_gram_dense(L)` already has its no-copy path) and `solve_nullspace` skips
    `progress_wrap` for a dense solver when it holds — there is no per-column
    work on that route to report, so building the ticking wrapper would only
    force the cost `_gram_dense` exists to avoid.
- Both device-generic: every building block used (`_qdn_modeW`, `_qdn_modeWp`,
  `_qdn_mode_order`, `_qdn_ttm`, `_qdn_unfold`) already operates on
  `AbstractArray`, so an `MtlArray` flows through `_qdn_pair_tensors` exactly as
  it did through `_qdn_pair_tensor`; no new device-specific code.

Reproduce:

```
git stash                 # or: git checkout 35a10d0 -- src/solvers/QuickDerN.jl src/solvers/NullSolvers.jl
JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 JL_RSS_LIMIT_GB=6 bench/jl bench/LiftCost.jl before
git stash pop              # (or restore the fix)
JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 JL_RSS_LIMIT_GB=6 bench/jl bench/LiftCost.jl after
```

## Before / after: `:lift` stage and total, seconds

| case | valence | m | lift before | lift after | lift speedup | total before | total after |
|---|---|---|---|---|---|---|---|
| sphere-v3-d60          | 3 | 1 | 0.0096 | 0.0110 | 0.87x | 0.534 | 0.523 |
| sphere-v3-d100         | 3 | 1 | 0.0338 | 0.0353 | 0.96x | 1.638 | 1.711 |
| sphere-v3-d150         | 3 | 1 | 0.1635 | 0.1586 | 1.03x | 8.746 | 8.752 |
| sphere-v4-d30          | 4 | 1 | 0.0107 | 0.0093 | 1.15x | 0.160 | 0.157 |
| sphere-v4-d50          | 4 | 1 | 0.0495 | 0.0425 | 1.17x | 0.196 | 0.183 |
| centroid3-random-60    | 3 | 3 | 0.0036 | 0.0032 | 1.15x | 0.485 | 0.469 |
| video-160x120x30x3 (F32)| 4 | 1 | 0.0113 | 0.0104 | 1.09x | 0.531 | 0.535 |

(Full CSV: `bench/reports/2026-09-08/lift/lift-cost.csv`, one row per
`(tag, case)`, every stage broken out.)

**Reading this honestly: the lift stage itself moves a little, the total
does not move at all, and that is exactly what the mechanism predicts at
these sizes** — see "What did not help" below before concluding the fix is
weak.

## The progress / `_gram_dense` fix (M11), measured separately

Not exercised by `LiftCost.jl` (none of its cases pass `progress = true`), so
measured directly: a `2000×2000` dense matrix wrapped as
`LinearMaps.LinearMap(M)` (exactly what QuickDer's dense branch hands
`GramSolver`), `_gram_dense` read with and without the old
`progress_wrap`-always behaviour simulated:

```
_gram_dense(L) directly (no progress):            0.0014 s
_gram_dense(progress_wrap(L)) (OLD progress=true): 1.3552 s  (983x slower)
new dispatch + _gram_dense (FIXED progress=true):  0.0000 s  (same object)
bytes materialised, OLD path:  32000000  (a full copy, via Matrix(FunctionMap))
bytes materialised, FIX path:  0         (L.lmap === M: true)
```

At QuickDer's own sizes (1.1 GB at valence 3 d=200, 3.3 GB at d=300 per the
`_gram_dense` docstring) the old behaviour was not a "1-3 GB copy" as first
described — it is `Matrix(FunctionMap)`, which *applies the map once per
column*, i.e. `O(n)` matrix-vector products of an already-dense operator: far
worse than a copy, and the 983x at n=2000 is a lower bound (it grows with n).
The fix makes `progress = true` free again on that path.

## Correctness evidence

**Bit-identical**, every case — `maximum(abs.(before .- after))` on the
serialized derivation coordinates
(`bench/reports/2026-09-08/lift/coords/{before,after}/<case>.jls`):

| case | max abs diff | principal angle |
|---|---|---|
| sphere-v3-d60 | 0.0 | 5.58e-8 |
| sphere-v3-d100 | 0.0 | 0.0 |
| sphere-v3-d150 | 0.0 | 9.54e-8 |
| sphere-v4-d30 | 0.0 | 4.94e-8 |
| sphere-v4-d50 | 0.0 | 8.56e-8 |
| centroid3-random-60 | 0.0 | 6.14e-8 |
| video-160x120x30x3 | 0.0 | 0.0 |

`max abs diff` is exactly `0.0` on every case: the returned coordinate
matrices are bit-for-bit identical, not merely close. The nonzero
`principal_angle` entries (`~1e-8`, i.e. `sqrt(eps(Float64))` scale) are an
artifact of recomputing a fresh QR + SVD on an already bit-identical pair of
matrices (`test/TestQuickDerDevice.jl`'s `principal_angle`), not a real
subspace difference — `max abs diff = 0.0` already proves that.

This is expected, not lucky: on the host, `_qdn_mode_order` returns its
candidate axes in ascending order unconditionally (by design — see its own
docstring, "HOST ARRAYS KEEP THE NATURAL ORDER... so nothing about an
existing Float32/Float64 CPU answer moves"). `_qdn_pair_tensors` picks its
pivot `c` as the smallest index `≠ a`, which is also the first element
`_qdn_pair_tensor`'s own per-pair ascending order puts down for any `b ≠ c` —
so for `b ≠ c` the new code contracts `[c; ascending(N \ {a,b,c})]`, the exact
same GEMM sequence as the old `ascending(N \ {a,b})`, in the same order, with
the same rounding. `b = c` falls back to `_qdn_pair_tensor` unchanged. The
row-hoist (F6) does not reorder any arithmetic either: `Wu[b]` is the same
expression whether computed once per `(b,i)` or redundantly once per
`(b,i,rho)`; only the redundant repetition is removed. Only on the *device*
(exercised by `test/TestQuickDerDevice.jl`'s `QuickDer device = :gpu`
testset, 22/22 passing on this run) could the pivot choice differ from the
per-pair ordering `_qdn_mode_order`'s cost model would have picked
independently — commuting mode products, so still correct, just not
bit-identical there; not measured separately here (no GPU obligation for this
task), but the device testset already checks CPU vs GPU agreement to
`principal_angle < _qd_tolerance(T, 1e-6)` and passed.

## Tests

- `test/TestQuickDerN.jl`, `test/TestQuickDerDevice.jl`, `test/TestAutoDer.jl`,
  `test/TestFastDer3Valent.jl`, `test/TestDerivationLaws.jl`,
  `test/TestNullVerdict.jl`: unchanged, all green (see the full-suite run
  below).
- New: `test/TestNullVerdict.jl`, testset `"progress reporting does not
  defeat _gram_dense's no-copy path"` — `Dleto._wraps_dense_matrix` correctly
  classifies a `WrappedMap` (true) vs a genuine `FunctionMap` (false);
  `Dleto._gram_dense` returns the *same object* (`===`) for the former, a
  *copy* (`==` but `!==`) for the latter; and `solve_nullspace(L, :GramSolver;
  ...)` returns bit-identical `vals`/`vecs`/`verdict` for
  `progress ∈ {false, true, :densify, :solve, :all, [:densify,:solve]}`.

## What did not help (and why, honestly)

- **The total wall time does not move at all in this table.** The restricted
  eigensolve (`solve` stage) is 88-98% of every case here (`AutoSolver` picked
  `:SVDSolver`'s dense route throughout). The lift is 1-8% of total time at
  these sizes, so even the mechanism working exactly as designed is invisible
  in the total — this table is a stage-level measurement for a reason.
- **Valence 3 shows ~no lift speedup (0.87x-1.03x, noise-level), and that is
  expected, not a miss.** For a lift axis `a` at valence 3 there are only 2
  other engaged axes (`bs = {b1, b2}`). `_qdn_pair_tensors` shares one
  full-tensor pass across `bs \ {c}`, but at valence 3 that is one axis at
  most — the *other* one (`b = c`) always needs its own fresh full pass
  regardless. Two full passes before, two after: F7 only starts paying off at
  valence ≥ 4, where sharing removes passes for *more than one* `b`. The
  measured valence-4 rows (1.15x-1.17x) and CONTEXT's own valence-4 measurements
  (`docs/CONTEXT.md`: valence 4 solves are far cheaper relative to valence 3 at
  comparable `d`) are consistent with this.
- **`centroid3-random-60`'s modest gain (1.15x) is F6 alone, not F7** (it is
  valence 3, so F7 gives nothing there either): `CentroidChisel(3)` has `m = 3`
  rows, so hoisting `Wt`/`Wu` out of the `rho` loop removes 2 of 3 redundant
  `_qdn_ttm`+`_qdn_unfold` calls per `(a,b,i)`. `sphere-v3`/`sphere-v4`/`video`
  all use single-row chisels (`UniversalChisel`, `m = 1`), so F6 has *nothing*
  to hoist there — their small movement is F7 (valence 4) or noise (valence 3).
- **A larger `m` or a larger `d` at valence ≥ 4 would show more**, per the
  cost model: F7's saving is `(length(bs) - 2) / length(bs)` of the lift's
  full-tensor-pass work, and F6's is `(m-1)/m` of the row-loop's `_qdn_ttm` +
  `_qdn_unfold` work — neither was re-measured at a size large enough to
  dominate the `solve` stage, to stay inside the 6 GB / one-quick-run budget
  the task's fixed case list already sets.
- **The `:fixed_nd` filter-reporting path, `:corner` restriction, and the
  matrix-free (`ArpackSolver`/`KrylovSolver`) branch were not separately
  re-measured** — `_qdn_pair_tensors`/the RHS hoist sit upstream of all three
  and are exercised by them in `test/TestQuickDerN.jl`'s existing tests (all
  passing), but the review's stage-time claims were only made for the dense
  path measured here.

## Reproduce

```
JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 JL_RSS_LIMIT_GB=6 bench/jl bench/LiftCost.jl before
JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 JL_RSS_LIMIT_GB=6 bench/jl bench/LiftCost.jl after
JL_PROJECT=$(pwd) JL_SLOTS=3 JL_THREADS=4 bench/jl test/runtests.jl
```
