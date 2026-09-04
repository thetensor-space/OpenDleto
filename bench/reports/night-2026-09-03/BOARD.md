# Night board — 2026-09-03 — valence-n stratification

Shared coordination file. Every agent APPENDS (never rewrites) a dated entry when it
finds a bug, a surprising number, or finishes. Read this file before starting and
before running any benchmark. Orchestrator relays cross-cutting items.

## Rules of the road
- Run Julia ONLY through `bench/jl` (2 slots, 2 threads each, 4G heap hint, 5 GB RSS kill).
  Timings are therefore 2-thread timings; compare within a run, not against the
  checkpoint's 4-thread numbers.
- Keep any single tensor under ~1 GB (Float64 d^4 with d <= 100; d^3 with d <= 500).
- Do NOT touch the running IJulia kernel (pid 78653) — that is the user's other task.
- Verification of a derivation is the Z-law: `applyDerivation(Γ, D, Chisel)` residual
  relative to `norm(Γ)·norm(D)` (see test/TestDerivationLaws.jl).

## Entries

### contraction-research (2026-09-03)
Full report: `bench/reports/night-2026-09-03/contraction-options.md`.
Dense (B): current ITensor `*` already reduces to permute+GEMM internally,
but has no verified zero-alloc entry point — switch `ester`/`sylve` to
TensorOperations.jl `@tensor` with preallocated output + `ismutating=true`
LinearMaps; stack AppleAccelerate.jl BLAS forwarding on top (vendor-claimed
6–13× GEMM vs OpenBLAS on Apple Silicon, not independently verified). Avoid
Tullio/LoopVectorization on Julia 1.12 (LoopVectorization has active 1.12
deprecation/test failures). Metal GPU is real for ITensor (`mtl()`) but
Float64-only paths can't use it (Apple GPUs have no fp64) — defer until CPU
zero-alloc is measured.
Sparse (A): no general non-QN sparse storage exists in ITensors today.
Recommend mode-n unfold to `SparseMatrixCSC` + stdlib sparse-dense `mul!`
(zero new deps, proven O(nnz·d)) now, with a Finch.jl `@einsum` spike in
parallel as the likely better long-term single-code-path answer across
n=3,4,5 (unverified for this exact TTM shape). No package anywhere covers
the Sylvester/derivation math itself — only the contraction kernel is
reusable.

### tests-quickdern (2026-09-03)
Wrote `test/TestQuickDerN.jl` (section 7 of `docs/design/QuickDer-valence-n.md`)
and wired it into `test/runtests.jl` after `TestFastDer3Valent.jl`. At write
time `Dleto.QuickDerMethod` does not exist yet, so `QUICKDER_AVAILABLE` is
`false` and every `:QuickDer`-specific assertion is `@test_skip`; the
`:SylverLining` half of every oracle runs today and all pass (ran standalone
through `bench/jl` on a driver that defines `der_residual` locally instead of
`include`ing `TestDerivationLaws.jl`: 111 pass, 15 skip, 0 fail/error, ~19s
wall on 2 threads). Once `QuickDerMethod` lands, re-running
`test/runtests.jl` should flip those skips green with no file changes needed
— if any of them instead fail, that is a real QuickDer bug, not a test bug.

Measured oracle table (all via `:SylverLining`, `tol=1e-6`, `UniversalOp()`
unless noted):

| case | dims / d | op space | nullity | max Z-law residual |
|---|---|---|---|---|
| random dense v3 | (4,5,3) | Universal | 2 (=n-1) | 6.0e-16 |
| random dense v4 | (5,4,6,3) | Universal | 3 (=n-1) | 7.1e-16 |
| random dense v5 | (4,3,3,3,2) | Universal | 4 (=n-1) | 9.1e-16 |
| random dense v4, Float32 | (5,4,6,3) | Universal | 3 | ~3e-7 |
| diagonal v4 | d=4 | Universal | 12 (=3d) | — |
| diagonal v4 | d=5 | Universal | 15 (=3d) | — |
| diagonal v4 | d=6 | Universal | 18 (=3d) | — |
| unscrambled sparse sphere, v3 | d=10 | Universal | 13 | — |
| unscrambled sparse sphere, v4 | d=10 (nnz=220/10^4) | Universal | **13** | — |
| scrambled sphere (harness), v3 | d=12 | Symmetric | 3 (lsq_err 1.0e-14) | — |
| scrambled sphere (harness), v4 | d=10 | Symmetric | 4 (lsq_err 1.0e-14) | — |
| disengaged chisel P=[1 1 0], v3 | (4,5,3) | Universal | 1 | — |
| CentroidChisel(3) (3-row), v3 | (4,4,4) | Universal | 1 | — |

Surprises:
- The valence-4 unscrambled sparse sphere (d=10, nnz=220 of 10^4, no scramble,
  `nondeg`'d) has the SAME 13-dimensional universal derivation algebra as the
  valence-3 case reported in `bench/reports/quickder-sphere.md` section 1.
  That report only measured valence 3; nothing here explains *why* the count
  is unchanged at valence 4 (the per-degree block-counting argument in that
  report is written for a ternary form and isn't obviously the same sum for a
  quaternary one) — flagging as worth a look, not asserting it's a general
  fact for all valences/d.
- `bench/SphereHarness.jl` already has the `valence` keyword as of this write
  (another agent's work landed mid-session) — `SPHERE_VALENCE_KW` in the new
  test file detects this at load time rather than assuming either way, so the
  valence-4 sphere test actually runs today instead of being skipped.

No package bug found in `src/` during this work (only `:SylverLining` was
exercised; `src/` was not touched).

### harness-valence-n (2026-09-03)

Generalised `bench/SphereHarness.jl` from hard-coded valence 3 to any valence
`n`. Changed: `sphere_octant(d; valence=3, mode=:densor, cutoff=1.5)` now
builds the hypersphere octant for any `n` (default 3, unchanged); `:lattice`
mode stays valence-3-only (errors otherwise, documented). `build_sphere`
takes `valence` and forwards it to `UniversalChisel(valence)`;
`reconstruction`'s hard-coded 3-axis `reshape(ds[1], :, 1, 1)`-style ALS
scaling is replaced with `_axis_reshape`/`_scaling_array` helpers built on
`ntuple`, generic in the number of axes. `run_stratify`, `warmup!`, `quietly`
had no valence-3 assumptions once `build_sphere` accepted `valence`; only
`warmup!` needed a `valence` kwarg added to pass through.

Construction is a new dense-array builder (`_fill_sum_lattice!`), NOT
`randTensorChisel`/`rand_den` (`src/util/TensorSynthesis.jl` `rand_den`,
`~line 104`, loops every `CartesianIndex` of the full `d^n` array — fine at
`d^3` but `d^4 >= 10^7` is far too slow). The new builder enumerates the
lattice `i_1+...+i_n = d-1` directly (compositions of `d-1` into `n`
nonnegative parts, `~d^(n-1)/(n-1)!` of them) and fills only those entries
with `randn()`. Verified byte-for-byte support agreement with the old
`randSurfaceTensor`-based construction at valence 3, `d=10`: both give nnz=55
and identical nonzero patterns.

New script `bench/HypersphereBaseline.jl`: CLI mirrors
`StratifyLeadersProfile.jl` (`dims... --valence= --seeds= --tol= --only=
--csv=`), runs `:SylverLining`/`{AutoSolver,ArpackSolver,SVDSolver}` and
`:QuickDer`, warms up on `d=6`, writes `d,seed,valence,solver,seconds,bytes,
nullity,lsq_err,status` with a `#`-comment noting these are 2-thread `bench/jl`
timings. `:ArpackSolver` is soft-loaded (`try using Arpack`); in this
environment `Arpack` is a **weakdep only** — it is declared in `[weakdeps]`/
`[extensions]` of `Project.toml` but is absent from `Manifest.toml` (not
`Pkg.add`ed to this project), so `using Arpack` fails and every
`:ArpackSolver` row reports a clean error instead of crashing the sweep. This
is a pre-existing environment gap, unrelated to this change — confirmed by
running `bench/jl bench/StratifyLeadersProfile.jl 12 --seeds=1
--only=Auto/F64` directly, which fails immediately at its `using Arpack` line
before reaching any Dleto code. Verified the "Auto/F64" config equivalently by
calling `build_sphere`/`run_stratify` directly (bypassing the script's
top-level imports): `d=12`, nullity=3, lsq_err=1.0e-14, perm_ok=yes,
support=1.0 — the valence-3 default path is unaffected by this change.

**Baseline table** (2-thread timings, `bench/jl`, `tol=1e-8`, `seeds=1`,
`SymmetricOp()`; full CSVs in `bench/reports/night-2026-09-03/
hypersphere-baseline-v{3,4}.csv`):

valence 4 (`d 8 10 12 16 20 --valence=4`):

| d | solver | seconds | bytes | nullity | lsq_err | status |
|---|---|---|---|---|---|---|
| 8 | AutoSolver | 0.029 | 92 MB | 4 | 2.9e-13 | ok |
| 8 | SVDSolver | 0.026 | 92 MB | 4 | 1.6e-14 | ok |
| 8 | ArpackSolver / QuickDer | — | — | 0 | NaN | error (see above / valence-3-only) |
| 10 | AutoSolver | 0.102 | 307 MB | 4 | 1.0e-14 | ok |
| 10 | SVDSolver | 0.093 | 307 MB | 4 | 2.5e-14 | ok |
| 12 | AutoSolver | 0.303 | 830 MB | 4 | 4.8e-14 | ok |
| 12 | SVDSolver | 0.294 | 830 MB | 4 | 1.6e-14 | ok |
| 16 | AutoSolver | 2.062 | 3.9 GB alloc | 4 | 2.8e-14 | ok |
| 16 | SVDSolver | 2.001 | 3.9 GB alloc | 4 | 6.5e-14 | ok |
| 20 | AutoSolver | — | — | — | — | **killed by bench/jl's 5 GB RSS watchdog before finishing** |

Sweep stopped at `d=16` for valence 4: `d=20` did not merely run slow, it was
**RSS-killed** on `AutoSolver`'s very first call (SVDSolver would fail
identically — it densifies unconditionally). Root cause looks like a real gap
in `dense_is_cheap` (`src/solvers/NullSolvers.jl:95-104`): at
`d=20, valence=4`, the `sylvesterLM` densor map is `(d^n, n·d(d+1)/2) =
(160000, 840)`. `min(m,n)=840 <= DENSE_LIMIT(1000)` so the function takes the
"always safe below `DENSE_LIMIT`, whatever the shape" branch and densifies —
but the docstring's own justification for that branch ("whatever its shape")
assumes the *product* stays modest, which held for the valence-3 examples in
the docstring (`n=19` -> 1083×1083, 9 MB) but not here: the `840`-column
matrix still has `160000` rows (~1 GB just to store), and whatever LAPACK
`gesdd`/copy overhead the dense SVD needs on top of that pushed RSS past
5 GB. At valence 4 (and worse at 5) `op_dim` grows only linearly in `n·d^2`
while `densor_dim = d^n` grows exponentially in `n`, so the `min(m,n) <=
DENSE_LIMIT` branch is the wrong gate whenever the *other* dimension is huge
even if `op_dim` alone looks small — it needs a byte check unconditionally,
not just a fallback below `DENSE_LIMIT`. Flagging for the orchestrator to
route; not fixed here (told not to touch `src/`).

Separately (not fixed, minor papercut): `solve_nullspace(L, solver::Symbol)`
(`src/solvers/NullSolvers.jl:699-700`) does a raw `SOLVER_REGISTRY[solver]`
lookup with no `haskey` guard, unlike `solve(L, sym::Symbol)`
(`NullSolvers.jl:266-273`) which has one and raises a friendly "Unknown or
unavailable solver ... Extension solvers need their package loaded first"
message. Calling `:ArpackSolver` with Arpack unloaded raises a bare
`KeyError: key :ArpackSolver not found` from the `solve_nullspace` path
instead — still caught by `run_stratify`'s try/catch (no crash), just a less
informative message than the sibling function gives.

valence 3 (`d 10 20 30 --valence=3`, sanity check, ran clean, no stop needed):

| d | solver | seconds | bytes | nullity | lsq_err | status |
|---|---|---|---|---|---|---|
| 10 | AutoSolver | 0.013 | 33 MB | 3 | 1.7e-14 | ok |
| 10 | SVDSolver | 0.012 | 33 MB | 3 | 3.2e-14 | ok |
| 10 | QuickDer | 0.046 | 18 MB | 3 | 1.5e-13 | ok |
| 20 | AutoSolver | 0.390 | 539 MB | 3 | 2.0e-13 | ok |
| 20 | SVDSolver | 0.371 | 539 MB | 3 | 4.1e-14 | ok |
| 20 | QuickDer | 0.103 | 85 MB | 3 | 3.9e-14 | ok |
| 30 | AutoSolver | 4.448 | 3.7 GB alloc | 3 | 3.5e-14 | ok |
| 30 | SVDSolver | 4.546 | 3.7 GB alloc | 3 | 4.0e-14 | ok |
| 30 | QuickDer | 0.364 | 292 MB | 3 | 6.4e-14 | ok |

(`ArpackSolver` errors at every row for the reason above.)

**Nullity finding.** With `SymmetricOp()`, valence 4 gives nullity **exactly
4** at every `d` tested (8, 10, 12, 16 via the baseline; also checked 8, 12
directly) with `lsq_err` in the `1e-13..1e-14` range (perfect recovery),
matching the expected 3 scalar derivations + the Euler/sphere derivation —
same story as valence 3's nullity 3, one dimension up. Confirmed with a dense
SVD of `Dleto.sylvesterLM(Ω, ch, Γ)`'s densor map at `d=8`
(size 4096×144) and `d=12` (20736×312): the spectrum has a sharp gap of
~1e13-1e14 between the 4th-smallest singular value (0.54 at d=8, 0.50 at
d=12) and the next one down (~1e-14..1e-15 at both), so nullity is 4 at
every tolerance from `1e-3` to `1e-8` — not a tolerance artefact. (This
matches the independent `tests-quickdern` entry above, which found the same
scrambled-sphere-v4-d=10 nullity of 4 via a different test file.)

**Verification of the valence-3 path.** `bench/jl bench/StratifyLeadersProfile.jl
12 --seeds=1 --only=Auto/F64` could not be run to completion — it fails at
its top-level `using Arpack` before reaching any harness code (see the
pre-existing Arpack gap above; this is not caused by the harness changes).
Verified the equivalent computation directly through the updated harness
instead (see above): default-valence `build_sphere`/`run_stratify` behaves
identically to before the generalisation.

Files changed: `bench/SphereHarness.jl` (generalised to any valence), new
`bench/HypersphereBaseline.jl`, new
`bench/reports/night-2026-09-03/hypersphere-baseline-v{3,4}.csv`.

### sylvester-kernel (2026-09-03)
`sylvesterLM` now has an array kernel: plain `Array`s, preallocated scratch,
`ismutating=true` LinearMaps, and a `SparseMatrixCSC` branch. The ITensor
version is kept verbatim as `sylvesterLM_itensor` and is reachable as
`backend=:itensor`; `:array` is the array kernel with dense unfoldings and
`:auto` (the default) is the array kernel with sparse detection at
`SYLVER_SPARSE_DENSITY = 0.05`. `derTrOpsReduced` gained a matching `backend`
keyword so a bench can pin either kernel without editing source.

Files: `src/SylverLining/SylverLining.jl` (all of the above),
`Project.toml` (+SparseArrays, a stdlib already in the Manifest).

EQUIVALENCE. `:array` and `:auto` are **bit-identical** to `:itensor` (max abs
difference exactly 0.0) on 108 configurations: valence 3/4/5, Float64 and
Float32, dense and sphere-octant Γ, ragged frames, Universal / Symmetric /
Diagonal / AntiSymmetric operator spaces, universal / Tucker / centroid
chisels, an engagement-reduced Ω (so operator `a` is not tensor axis `a`), and
a `TransverseOpsSymmetries` space. `Matrix(densor_map)` agrees entry for entry.
Full `test/runtests.jl` passes (exit 0, 0 fail / 0 error), including the
transpose law and `f'∘f` sets in TestSylverLining.

ONE APPLY, 2 threads, universal chisel + `UniversalOp()`, ms and bytes:

| case | ester :itensor | :auto | sylve :itensor | :auto | sylvester :itensor | :auto | sylvester speedup | sylvester bytes before → after |
|---|---|---|---|---|---|---|---|---|
| dense v3 d=40  | 0.321 | 0.275 | 0.573 | 0.232 | 0.836 | 0.490 | **1.71x** | 6.9 MB → 384 B |
| dense v3 d=80  | 3.105 | 3.103 | 4.118 | 2.682 | 6.782 | 5.905 | 1.15x | 49.4 MB → 384 B |
| dense v4 d=16  | 0.536 | 0.275 | 0.404 | 0.282 | 0.692 | 0.541 | 1.28x | 9.3 MB → 448 B |
| dense v4 d=24  | 1.400 | 1.485 | 1.757 | 1.364 | 3.079 | 2.785 | 1.11x | 53.0 MB → 512 B |
| sphere v3 d=40 | 0.321 | 0.149 | 0.693 | 0.080 | 1.051 | 0.216 | **4.87x** | 6.9 MB → 384 B |
| sphere v3 d=80 | 2.934 | 1.309 | 4.006 | 0.796 | 6.960 | 1.994 | **3.49x** | 49.4 MB → 384 B |
| sphere v4 d=16 | 0.342 | 0.201 | 0.462 | 0.103 | 0.691 | 0.287 | **2.41x** | 9.3 MB → 448 B |
| sphere v4 d=24 | 1.373 | 0.968 | 1.752 | 0.499 | 3.171 | 1.423 | **2.23x** | 63.1 MB → 512 B |

Allocation per warmed-up apply is 384-512 bytes (the `mul!` transpose/reshape
wrappers), against 5-63 MB before: a ~10^5 reduction. Forcing `:array` on the
sphere gives the dense-branch numbers (e.g. v3 d=80 sylvester 5.774 ms), so the
sparse branch itself is worth a further 2.0-2.9x on top of the array rewrite.

END TO END, `der(:SylverLining, Γ; tol=1e-8)`, same two threads, two repeats
(the box is shared, so the `:itensor` column is noisy):

| case | :itensor | :auto | speedup | heap :itensor → :auto |
|---|---|---|---|---|
| dense v3 d=40  | 29.6 / 40.0 s | 28.9 / 26.9 s | 1.0-1.5x | 32.9 GB → 1.24 GB |
| sphere v3 d=40 | 39.2 / 33.8 s | 27.0 / 19.2 s | 1.5-1.8x | 32.9 GB → 1.24 GB |
| dense v4 d=16  | 4.78 / 1.24 s | 1.06 / 0.91 s | 1.4-4.5x | 9.3 GB → 70 MB |
| sphere v4 d=16 | 4.60 / 1.16 s | 0.66 / 0.60 s | 2.0-7.0x | 9.3 GB → 71 MB |

Nullities agree with `:itensor` in every case (2, 13, 3, 13). The valence-3
d=40 rows are held back by the solve, not the kernel: `AutoSolver` densifies
the 4800x4800 Gram operator and the dense SVD of it is ~20 s of the wall time.

WHAT MADE THE DIFFERENCE, for whoever tunes this next. The mode-`p` unfolding
convention is not cosmetic. Writing it as `(N/d_p) x d_p` -- other axes
column-major in the ROWS, `p` in the columns -- makes the mode product
`Γ_(p)·M` and the adjoint `Γ_(p)ᵗ·Z`, which are the two CSC products
SparseArrays has good kernels for. The natural-looking other convention (axis
`p` first) gives `Mᵗ·Γ_(p)`, i.e. dense x sparse, which SparseArrays does NOT
have a good kernel for -- at valence 3, d = 80, sphere density it measured
**0.49 ms against 0.14 ms**, barely better than the 0.82 ms dense GEMM it was
supposed to replace. Dense GEMM is indifferent to the choice (0.82 ms either
way). This chosen convention also makes the permutation the identity on the
LAST axis, so one axis per apply needs no `permutedims` at all; and when the
chisel has one row (universal and adjoint chisels -- every solve `der` and
`stratify` actually run) the residual is its own chisel-collapse up to a
scalar, so that scalar rides in the GEMM's `α`, `sylve` needs no gemv, and
`ester` accumulates straight into the residual. Those two together were worth
more than the sparse branch on the dense inputs.

Also folded in, at the coordinator's request: `derTrOpsReduced` hands
`solve_nullspace` the SQUARE map with `squared=true` again instead of the
rectangular `ester_map`.

BUG FOUND ELSEWHERE (not fixed). `src/ops/TransverseOpsSymmetries.jl:102-103`:
the two-argument constructor
`TransverseOpsSymmetries(fr::Vector{Index}, localOp::Operator)` forwards an
undefined variable `symmetries`, so any call to it is an immediate
`UndefVarError`. Nothing calls it today. (`IndTransverseOps` has the working
twin at TransverseOpsIndependant.jl:75.)

NOT DONE. AppleAccelerate was skipped -- there is no way to try it without
either touching `Project.toml`'s dependency set or standing up a scratch
environment and a `LOAD_PATH` hack, and the dense cases are already within a
few percent of the GEMM floor, so the upside is bounded by the ~30% of a dense
apply that is not GEMM.

HOUSEKEEPING for the orchestrator: this agent's worktree was created off `main`
(40b20d5), not off `aint-no-mountain-high-enough/valence-n-stratify` (e18b65f),
so `src/SylverLining/SylverLining.jl` did not even exist in it. The 25 changed
source files were re-materialised from e18b65f by hand before any work started;
the diff to merge is `src/SylverLining/SylverLining.jl` and `Project.toml`
only.
