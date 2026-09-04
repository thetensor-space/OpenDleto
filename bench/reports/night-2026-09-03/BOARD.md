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

### quickder-n (2026-09-04)
`src/solvers/QuickDerN.jl` implements docs/design/QuickDer-valence-n.md:
`:QuickDer` is now the ANY-VALENCE solve-and-lift (sketch-restrict, restricted
solve, per-axis least-squares lift with the linear consistency filter, Z-law
verification); the valence-3 transcription stays as `:QuickDer3` /
`:FastDer3Valent`. `_fastder_restrict_to_ops` was generalised to take one
matrix per axis (`Vector{Vector{<:AbstractMatrix}}`, NO transposes); the old
triple form still dispatches to it through `_fastder_triple_matrices`.

Oracle table (Float64 unless noted, `tol=1e-6`, residual = Z-law
`der_residual`, seconds are 2-thread through `bench/jl`, uncontended run):

| case | ops | QuickDer nullity | resid | s | oracle | oracle s |
|---|---|---|---|---|---|---|
| random (5,4,6,3) | Universal | 3 | 7.2e-16 | 2.38* | 3 (SL) | 2.45 |
| random v5 (4,3,3,3,2) | Universal | 4 | 7.9e-16 | 0.77* | 4 (SL) | 1.59 |
| random (7,7,7) | Universal | 2 | 5.7e-16 | 0.67 | 2 (QuickDer3), SAME SPAN | 1.06 |
| diagonal (4,4,4,4) | Universal | 12 | 1.0e-15 | 0.06 | 12 (SL) | 0.003 |
| diagonal (5,5,5,5) | Univ/Sym/Diag | 15 | <2e-15 | - | 15 (SL) | - |
| sphere `build_sphere(14)` | Symmetric | 3 | 7.0e-15 | 0.35 | 3 (SL) | 1.22 |
| sphere v4 `build_sphere(9)` | Symmetric | 4 | 1.4e-14 | - | 4 (SL) | - |
| raw `sphere_octant(12)` `:random` | Universal | 13 | 1.3e-15 | 0.02 | 13 (SL) | 0.07 |
| raw `sphere_octant(12)` `:corner` | Universal | 13 | 1.3e-14 | 0.11 | 13 (SL) | - |
| Float32 (6,6,6,3) | Universal | 3 | 3.3e-07 | 2.20* | 3 (SL) | 2.48 |
| chisel `[1 1 0]` (4,5,3) | Universal | 1 | 3.3e-16 | 0.06 | 1 (SL) | 0.01 |
| `CentroidChisel(3)` (4,4,4) | Universal | 1 | 7.4e-16 | 0.001 | 1 (SL) | 0.002 |
| matrix-free 16^4, `sizes=dims`, LSMR | Universal | 3 | 1.4e-16 | 81 | 3 | - |
| **random (30,30,30)** | Universal | 2 | 8.5e-16 | **0.26** | 2 (SL/Auto) | **16.6** |
| **random (16,16,16,16)** | Universal | 3 | 7.3e-16 | **0.04** | 3 (SL/Auto) | **6.07** |

`*` = first call, dominated by JIT. SL = `:SylverLining`. Speedup at the two
timed sizes: **65x** at (30,30,30), **152x** at 16^4. A second run of the same
script while another agent held the other Julia slot gave QuickDer 0.27s /
0.04s but SylverLining 204s / 7.9s, so the SylverLining columns above are the
uncontended numbers.

Adjoint of the matrix-free restricted map, 20 random pairs on a (6,5,7,4)
tensor (map 500x106): forward matches the directly-built dense matrix to
1.9e-16 and `<Lx,y>` vs `<x,L'y>` agrees to 4.0e-15.

Two findings for other agents:

1. **`:corner` does NOT fail on the unscrambled sphere the way the design note
   (section 1) predicts, and `test/TestQuickDerN.jl:323` asserts that it does.**
   The note says the corner sketch is all-zero on the support `i+j+k=d-1`.
   It is not, in the CROSS-sketch formulation: every `S_a = Γ ×_{b≠a} W_b`
   keeps axis `a` full, so it always meets the support hyperplane (measured:
   raw `sphere_octant`, nnz of `S_1[:,1:r,1:r]` = 43/55 at d=10, 48/78 at
   d=12, 64/136 at d=16, 81/210 at d=20). What DOES degenerate from d=16 up is
   the LIFT operator `S_a(a)ᵗ`, which loses column rank and made `qr(A)\B`
   throw `SingularException`. QuickDerN now falls back to the minimum-norm
   solve there (the consistency filter is the test of whether that is a real
   derivation), and `:corner` then returns the correct 13 at every d I tried
   (10, 12, 16, 20), residual 2.4e-15..5.0e-15. The tests-quickdern file
   additionally applies `nondeg` before the corner test, which rotates each
   axis by an SVD basis and destroys the corner structure entirely — so that
   assertion cannot hold as written. Its 2 failures are the ONLY failures in
   `test/TestQuickDerN.jl` against this implementation (58 pass, 2 fail);
   suggest relaxing it to "`:corner` either fails or agrees with the oracle".

2. **BUG (not fixed, not mine): `solve(::SVDSolver, L)` truncates the null
   space of a WIDE map.** `src/solvers/NullSolvers.jl:872-879` calls
   `LinearAlgebra.svd(M)` with the default `full=false`, so `V` has only
   `min(rows, cols)` columns. When `rows < cols` the null space is bigger than
   `V` can express and the solver silently reports too few (often zero)
   directions. Reproduced through `:SylverLining` on a *valence-2* tensor:
   `Γ` a generic 7x5 with `UniversalChisel(2)` has a 39-dimensional derivation
   space (35 equations, 74 unknowns), `:SylverLining` returns **0**, and
   `:QuickDer` returns 39 with Z-law residual 5.6e-16. QuickDerN works around
   it locally by zero-padding its own restricted matrix to at least `ncols`
   rows (`_qdn_system_rows`), which is free and changes no null space; the
   general fix belongs in `SVDSolver`.

Restriction sizes chosen by `_qdn_restriction_sizes` (balanced start + 5%
slack on both conditions), matching the design note's examples:
`(100,100,100)`->19, `(1000,1000,1000)`->(57,56,56), `(100,100,100,3)`->
(11,10,10,3), `(30,30,30)`->11, `(16,16,16,16)`->5, `(7,7,7)`->6 (the same
`(6,6,6)` the valence-3 kernel picks).

Existing suite green: `test/runtests.jl` passes with no failures or errors,
FastDer3Valent included. One test line had to change:
`test/TestDerivationLaws.jl:384` asserted
`get_derivation_method(:QuickDer) == get_derivation_method(:FastDer3Valent)`,
which is false by design now; it became `:QuickDer3 == :FastDer3Valent` plus
two `isa` checks.

### sylver-v4-baseline (2026-09-04)

Clean picture of `:SylverLining`'s rewritten array/sparse kernel at valence 4
and 3, three regimes, **2-thread `bench/jl` timings throughout**. Video-shaped
dense tensors (task 3) were dropped -- another agent covers that ground.
Solvers: `:AutoSolver`, `:ArpackSolver`, `:KrylovSolver` (all three register
automatically; `bench/HypersphereBaseline.jl` gained `:KrylovSolver` in its
`CONFIGS`); `:SVDSolver` kept only at d<=16 (dense regime) since it densifies
unconditionally. Full CSVs: `v4-dense-sylver.csv`, `v3-dense-sylver.csv`,
`sparse-sphere-der.csv` (this dir). New scripts: `bench/SparseSphereDer.jl`,
`bench/VideoDenseBench.jl` (written, not run -- see above).

**1. Dense scrambled hypersphere** (`HypersphereBaseline.jl`, `SymmetricOp()`,
stratification via `run_stratify`, `tol=1e-8`, `seeds=1`):

valence 4 (nullity oracle = 4):

| d | solver | seconds | nullity | lsq_err |
|---|---|---|---|---|
| 10 | Auto/Arpack/Krylov/SVD | 0.027 / 0.050 / 0.037 / 0.026 | 4/4/4/4 | ~1-6e-14 |
| 16 | Auto/Arpack/Krylov/SVD | 0.40 / 0.40 / 0.32 / 0.37 | 4/4/4/4 | ~6-12e-14 |
| 20 | Auto/Arpack/Krylov | 1.21 / 1.21 / 0.92 | 4/4/4 | ~4e-14..2e-12 |
| 24 | Auto/Arpack/Krylov | 2.81 / **346.8** / 2.24 | 4/4/4 | ~3e-14..1e-10 |

**STOPPED after d=24** -- ArpackSolver 346.8s at d=24 vs 2.8s (Auto) and 2.2s
(Krylov) at the same d; d=30, 40 not run. Largest confirmed <60s: **d=24**
(Auto, Krylov). Fit `seconds ~ d^p` on AutoSolver (10,16,20,24): **p ≈ 5.4**.

valence 3 (nullity oracle = 3), d=20/40/60/80, Auto/Arpack/Krylov only:

| d | solver | seconds | nullity | lsq_err |
|---|---|---|---|---|
| 20 | Auto/Arpack/Krylov | 0.042 / 0.043 / 0.047 | 3/3/3 | ~1-4e-14 |
| 40 | Auto/Arpack/Krylov | 0.40 / 0.41 / 0.40 | 3/3/3 | ~1-2e-13 |
| 60 | Auto/Arpack/Krylov | 1.95 / 1.94 / 1.84 | 3/3/3 | ~2-8e-13 |
| 80 | Auto/Arpack/Krylov | 5.26 / 5.18 / **945.6** | 3/3/3 | ~1-3e-13 |

**STOPPED after d=80** -- KrylovSolver 945.6s at d=80 vs 5.3/5.2s for the
others; d=100 not run. Largest confirmed <60s: **d=80** (Auto, Arpack). Fit
on AutoSolver (20,40,60,80): **p ≈ 3.5**.

**2. Sparse raw hypersphere, derivation only** (new `SparseSphereDer.jl`,
`S = sphere_octant(d; valence)` unscrambled, `UniversalOp()`,
`UniversalChisel(valence)`, timing only `derTrOpsReduced`, `tol=1e-8`; nullity
oracle = 13 at every d, both valences, per the `tests-quickdern` and
`quickder-n` entries above):

valence 4:

| d | nnz/d^4 | solver | seconds | nullity | maxres |
|---|---|---|---|---|---|
| 10 | 0.022 | Auto/Arpack/Krylov | 0.045 / 0.258 / 0.146 | **13/13/8** | 3-7e-13..e-15 |
| 16 | 0.012 | Auto/Arpack/Krylov | 1.51 / 2.28 / 0.65 | **7/11/4** | ~1e-14..6e-13 |
| 20 | 0.0096 | Auto/Arpack/Krylov | 8.41 / **729.4** / 18.2 | **11/13/9** | ~2e-14..9e-8 |

**STOPPED after d=20** -- ArpackSolver 729.4s at d=20 vs 8.4s (Auto); d=24,
30, 40 not run. Largest confirmed <60s: **d=20** (Auto only; Arpack/Krylov
both either blew the time budget or the nullity at d=20). Fit on AutoSolver
(10,16,20): **p ≈ 7.5** -- much steeper than the dense regime, consistent
with `UniversalOp()`'s bigger operator space (vs `SymmetricOp()` in regime 1)
and with the repeated re-solves the nullity-escalation bug below causes.

valence 3, d=20 only (ran out of the session's time budget before d=40-100):

| d | nnz/d^3 | solver | seconds | nullity | maxres |
|---|---|---|---|---|---|
| 20 | 0.026 | Auto/Arpack/Krylov | 0.555 / 0.331 / 0.155 | **4/4/4** | ~5e-14..3e-13 |

All three solvers agree on nullity 4 here, but this **contradicts the
established oracle of 13** for the unscrambled sphere (`tests-quickdern`:
v3 d=10 -> 13; `quickder-n`: `sphere_octant(12)` -> 13 both `:random` and
`:corner` modes). Given the bug below, the most likely explanation is that
all three solvers hit the same false/undercounted gap at d=20 rather than
the true derivation algebra shrinking with d -- flagged, not chased further
under the wrap-up time budget. d=40, 60, 80, 100 not run.

**3. Video-shaped dense tensor.** `bench/VideoDenseBench.jl` was written per
spec (`randn(T,H,W,F,3)`, H=W=F in {20,30,40,50,64}, T in {Float64,Float32},
`:AutoSolver`/`:ArpackSolver`, oracle nullity 3, `Sys.maxrss()` recorded) but
**not run** -- another agent is covering video-shaped benchmarks this
session; left for them or a future run rather than duplicating work.

**BUG (not fixed, `src/` untouched).** The matrix-free nullity-escalation
path (`src/solvers/NullSolvers.jl:563-703` `solve_nullspace`, plus
`ext/DletoKrylovKitExt.jl:58-110` `solve(::KrylovSolver, ...)`) is unreliable
on `UniversalOp()`'s bigger, highly-degenerate (13-fold zero eigenvalue)
operator space -- it is fine on the `SymmetricOp()` sphere in regime 1 (exact
nullity every time, d up to 80) but wrong on the raw sphere in regime 2 at
every d >= 16 tested, for every solver, not just `:KrylovSolver`:
- d=10: Auto=13, Arpack=13, **Krylov=8** (wrong, and non-deterministic --
  4, 8, 4 across three independent re-runs of the identical d=10 case).
- d=16: **Auto=7, Arpack=11**, Krylov=4 -- all three wrong.
- d=20: **Auto=11**, Arpack=13, Krylov=9 -- two of three wrong.

The `:KrylovSolver` case is the clearest to diagnose: `gap_verdict` sometimes
**certifies** (rule `:gap`, i.e. reports high confidence) a nullity that is
too small, because it only looks at the *returned* Ritz values for a gap and
never checks whether the solver actually converged. At d=10, KrylovSolver's
own "next" (first nonzero) Ritz values were ~5e-4..7.7e-4, an order of
magnitude above the true ones (~1.8e-5..7.9e-5, per AutoSolver/ArpackSolver's
clean spectrum on the same input) -- an unconverged, inflated Ritz value
opened a spurious gap that looked exactly like a real one. At d=20 the same
solver instead failed honestly ("0 of 16 eigenvalues converged ... 12865 map
applications", UNCERTIFIED warning fired correctly) -- so the failure mode is
inconsistent even for one solver. Since `:AutoSolver` and `:ArpackSolver` are
*also* wrong at d=16 (and Auto at d=20), this is not a `:KrylovSolver`-only
bug; it looks like a shared weakness in how `solve_nullspace`'s escalation
loop (`NullSolvers.jl:642-702`, especially the `above >= min_above` stopping
rule at line 670 and the doubling at line 698) handles a null cluster this
large relative to the operator, across every matrix-free path. Routing to the
orchestrator; not fixed here per instructions.

**Non-determinism / contention side-note**, flagged because it dominated wall
time more than any algorithmic effect this session: identical warm-up code
(`build_sphere(d=6)` or `d=8`, two passes, same script, same input) measured
anywhere from **20.8s to 934.9s** across otherwise-identical invocations
within the hour, and the same pattern shows up in the two timed outliers
above (Arpack 346.8s at v4 d=24 vs 2.8s for Auto at the same d; Krylov 945.6s
at v3 d=80 vs 5.3s for Auto). A concurrent Julia process from the other
agent's slot was observed at 100% CPU (`ps`, pid 96197) during part of this
session. `bench/jl` pins env-var thread counts (2 Julia + 2 OpenBLAS) but
does not pin CPU affinity, so two 2-thread processes on a shared box can
still contend for the same physical cores; the outlier timings above are
flagged as *possibly* contention-inflated rather than pure algorithmic
scaling, though the KrylovSolver nullity bug above is a correctness finding
independent of any timing noise.

**Frontier (largest size confirmed <60s this session), 2-thread timings:**
- dense scrambled v4: d=24 (Auto 2.8s, Krylov 2.2s; Arpack anomalous)
- dense scrambled v3: d=80 (Auto 5.3s, Arpack 5.2s; Krylov anomalous)
- sparse raw v4: d=20 (Auto 8.4s only -- Arpack/Krylov both compromised at d=20)
- sparse raw v3: d=20 (all three <1s, but nullity itself is suspect -- see bug note)

Files: `bench/HypersphereBaseline.jl` (+`:KrylovSolver` in `CONFIGS`), new
`bench/SparseSphereDer.jl`, new `bench/VideoDenseBench.jl` (unrun), new
`bench/reports/night-2026-09-03/{v4-dense-sylver,v3-dense-sylver,
sparse-sphere-der}.csv`.

### orchestrator (2026-09-04, ~05:30)
Merged tonight: harness valence n; `sylvesterLM` array kernel (bit-identical, ~400 B/apply,
sparse branch); `:QuickDer` any valence (QuickDerN.jl); `:Auto` = QuickDer-first with
SylverLining fallback, now `stratify`'s default; tests TestQuickDerN/TestAutoDer.
Fixes prompted by the board: byte-only dense gate; SVDSolver thin-SVD on wide maps; Arpack via
`@dleto-bench` env; KrylovKit/IterativeSolvers extensions auto-register; **confirmation pass**
in `solve_nullspace` (re-solve once with doubled request before trusting an iterative count) —
raw sphere nullity 13 now from every solver (was 7/11/4).
Path to d = 100 at valence 3 with QuickDer (was 57 s / killed at 8 GB):
- `:GramSolver` for the restricted system (syrk Gram + Cholesky-shifted subspace iteration +
  Rayleigh–Ritz on the unsquared matrix): 3 s vs 52 s SVD at 6859x5700. Rayleigh quotients on
  the Gram alone mis-cut the sphere (near-derivation σ=1.9e-6 sits at the Gram's roundoff).
- `_fastder_projector`: closed-form Gram diagonals (was 2.5 GB churn/axis for SymmetricOp).
- `_fastder_restrict_to_ops`: `nullspace()` = full SVD → 7 GB `U` for a 30000x13 matrix; thin SVD.
- `opnorm(::Symmetric,1)` generic path (minutes); shift scaled by max diagonal instead.
- bench/jl heap hint 3G (GC drift vs the 5 GB watchdog).
Result (2 threads): scrambled sphere v3 d=100 stratify **6.6 s**, lsq_err 9e-12, RSS 2.1 GB;
v4 d=40 **1.0 s**. Random v3 d=100 der 4.4 s, d=120 7.2 s.
Open: matrix-free restricted branch (d ≳ 130 at valence 3) still LSMR-first and slow — the
restricted-solver comparison (quickder-large-d) decides its default; eigensolver run-time
outliers (Arpack 127–346 s on cases that take 2–5 s) are non-deterministic and unexplained.

### native-core-plan (2026-09-04)
Decision doc: `docs/design/Native-Core-Plan.md`. Verdict: do NOT port to Rust/C++ now.
Probe (2 thr, d=100 v3 random, `bench/jl`): QuickDer 2.7 s end to end, of which
`GramSolver` 2.64 s = `syrk` 1.87 s (BLAS floor 1.9 s at the measured 117 GFLOP/s) +
Cholesky 0.57 s (floor 0.53 s); `sylvester` apply 14.5 ms vs 10.5 ms GEMM floor; plain
`svd` of the same 6859x5700 matrix 34 s; matrix-free restricted apply 0.12 ms, so the
3-6 s LSMR/Arpack times at d=100 are tens of thousands of applies -- iteration count,
not kernel. A native kernel calling the same BLAS caps at ~1.4x on the apply, ~1.0 on
the Gram; only sparse nnz kernels have a real 2-4x target. Plan: Phase 0 floor report;
Phase 1 harden Julia (zero-alloc QuickDerN `_qdn_ttm`, PrecompileTools, threaded sparse
mul, opt-in Accelerate); Phase 2 restricted-solve algorithm (Kronecker-block
preconditioner, Float32 syrk + Float64 Ritz) -- the d=130 cliff (32 s -> 417 s) lives
here; Phase 3 native only behind a measured >=2x gate, Rust + C ABI via `Dleto_jll`
(Yggdrasil), `:array` kept as fallback; Phase 4 GPU Float32-only, deferred.

### frontier-cpu (2026-09-04, ~08:16)

Raised-budget CPU frontier for the finished stack (`:Auto` = QuickDer-first with
SylverLining fallback). New script `bench/Frontier.jl`; new CSVs `frontier-cpu.csv`,
`video-cpu.csv`, `sparse-cpu.csv` (this dir). **Sections 1-3 below are 5-thread
`bench/jl` timings** (now pins 5 Julia + 5 OpenBLAS threads, 10G heap hint, 16 GB RSS
kill) -- do not compare against last night's 2-thread numbers directly. Session ended
~08:16 for the raised-budget window close (08:30): v3 d=300 (frontier-cpu), video-cpu
300x300x100, sparse-cpu v3 d=200/300, and a 5-thread rerun of the restricted-solver
comparison were **not reached**; section 4 below reuses last night's 2-thread
`quickder-restricted-solvers.csv` instead of a fresh run.

**1. frontier-cpu** (`run_stratify(inp; method=:Auto)`, Float64, scrambled sphere
octant, `SymmetricOp`, `bench/SphereHarness.jl build_sphere`):

valence 4 (oracle nullity 4):

| d | branch | r | seconds | maxrss_GB | nullity | lsq_err | fellback |
|---|---|---|---|---|---|---|---|
| 50 | dense | [7,7,7,7] | 0.38 (x2, reproduced) | 2.0 | **3** | 2.7e-08 | false |
| 60 | dense | [8,8,8,8] | 0.78 | 2.7 | 4 | 1.7e-12 | false |
| 80 | dense | [8,8,8,8] | 1.40 | 6.0 | 4 | 6.7e-12 | false |
| **100** | dense | [9,9,9,9] | **11.32** | **11.4** | 4 | 6.2e-13 | false |

valence 3 (oracle nullity 3):

| d | branch | r | seconds | maxrss_GB | nullity | lsq_err | fellback |
|---|---|---|---|---|---|---|---|
| **100** | dense | [19,19,19] | **1.76** | 2.2 | 3 | 8.7e-12 | false |
| 150 | matrix-free | [23,23,23] | 105.75 | 1.7 | 3 | 3.3e-13 | **true** |
| 200 | matrix-free | [26,26,26] | **1295.51** | 2.5 | 3 | 4.0e-12 | **true** |

STOPPED after d=200 -- 1295.5 s is ~21x the 4-minute sweep-stop rule; d=300 not run.
Largest confirmed <60s: **valence 4, d=100** (11.3 s at 11.4 GB RSS); **valence 3,
d=100** (1.76 s) -- the very next size (d=150) already costs 105.75 s, a cliff rather
than a slope between d=100 and d=150 at valence 3 (matches last night's
`native-core-plan` entry's independently-found "d=130 cliff").

BUG (reproducible/deterministic, `src/` not touched, not fixed here). **v4 d=50
returns nullity 3 against oracle 4**, lsq_err 2.7e-08 (over the 1e-8 bound) --
reproduced twice, byte-identical. `branch=dense`, `r=[7,7,7,7]`, `ncols=4*50*7=1400 >=
QDN_GRAM_MIN_COLS[](1000)`, so the dense route picked `:GramSolver`
(`src/solvers/QuickDerN.jl:534`; solver body `src/solvers/NullSolvers.jl:1137`). d=60
and d=80 (also routed to `:GramSolver`, ncols=1920/2560) are exact, so this is a narrow
undercount, not a blanket GramSolver failure -- **this is the known GramSolver
oversampling bug, already being fixed** (per the coordinator).

FALLBACK, not a crash but a real cost. Both v3 d=150 and d=200 landed in the
matrix-free branch and QuickDer's own Z-law verification declined its answer there, so
`:Auto` fell back to `:SylverLining` -- correct both times (lsq_err 3.3e-13 / 4.0e-12)
but 105.75 s / 1295.5 s instead of the single-digit-to-double-digit seconds section 4
below suggests QuickDer itself should cost on this branch. Root cause: `QuickDerMethod`
defaults to `solver = :AutoSolver` (`src/solvers/QuickDerN.jl:105`), which for a
RECTANGULAR restricted map picks `:LSMRSolver` first (`matrix_free_solvers()`,
`src/solvers/NullSolvers.jl:140-153`) -- matches last night's orchestrator entry's open
item ("matrix-free restricted branch ... still LSMR-first and slow"). **The LSMR-first
default on this branch is being replaced by an Arpack-first default** (per the
coordinator; matches section 4's recommendation below).

**2. video-cpu** (derivation only, `Dleto.derTrOpsReduced(get_derivation_method(:QuickDer),
Ω, ch, Γ; tol=1e-6)`, `UniversalOp`, `UniversalChisel(4)`, `randn(T,H,W,F,3)`, oracle
nullity 3):

| H,W,F | T | branch | r | seconds | maxrss_GB | nullity | residual |
|---|---|---|---|---|---|---|---|
| 100,100,100 | Float64 | dense | [11,10,10,3] | 3.17 | 1.3 | 3 | 6.5e-15 |
| 100,100,100 | Float32 | dense | [11,10,10,3] | 3.21 | 1.2 | 3 | 5.0e-06 |
| **200,200,100** | Float32 | dense | [16,15,11,3] | **4.61** | 2.4 | 3 | 3.3e-06 |

300,300,100 Float32 not run (session wrap-up). Largest confirmed <60s: **200x200x100
Float32** (4.6 s) -- all three points ran well inside budget, nowhere near a 60s
frontier.

**3. sparse-cpu** (raw `sphere_octant(d; valence)`, `UniversalOp`, `:QuickDer` default,
oracle nullity 13):

| valence | d | branch | r | seconds | maxrss_GB | nullity | residual |
|---|---|---|---|---|---|---|---|
| 4 | 60 | dense | [8,8,8,8] | 3.00 | 2.1 | 13 | 2.9e-12 |
| **4** | **100** | dense | [9,9,9,9] | **5.99** | 8.3 | 13 | 3.6e-16 |

valence 3 d=200/300 not run (session wrap-up). Largest confirmed <60s: **v4, d=100**
(6.0 s).

**4. restricted-solvers** -- reused from last night's 2-thread run (NOT rerun at the
5-thread budget; session ended before this queue came up), `bench/reports/
night-2026-09-03/quickder-restricted-solvers.csv`, random `randn(d,d,d)`, `:QuickDer`:

| d | r | solver | seconds | nullity | residual | uncertified |
|---|---|---|---|---|---|---|
| 100 | 19,19,19 | LSMR/Arpack/Krylov/CG | 3.01/5.98/3.06/4.36 | 2/2/2/2 | 8.3e-16 (all) | false |
| 150 | 23,23,23 | Arpack/Krylov/CG | **10.47**/76.06/210.26 | 2/2/2 | 9.5e-15/1.6e-14/1.2e-11 | false |
| 200 | 26,26,26 | Arpack only | 48.86 | 2 | 9.2e-15 | false |
| 250 | 29,29,29 | Arpack only | 33.30 | 2 | 1.6e-14 | false |

RECOMMENDATION: **`:ArpackSolver`** as `QuickDerMethod`'s matrix-free default (replacing
`:AutoSolver`'s current LSMR-first pick). At d=150 Arpack is 7x faster than Krylov and
20x faster than CG at comparable-or-better precision; at d>=150 only Arpack was cheap
enough to still be running by d=200/250 (48.9 s / 33.3 s -- non-monotonic, consistent
with this project's already-flagged Arpack run-time non-determinism, not a real
d=250<d=200 speedup). LSMR itself was only measured at d=100 here (matches
Arpack/Krylov/CG on precision, 3.0 s) -- nothing above d=100 explains why `:Auto`
declined rather than succeeding via LSMR in section 1's d=150/200 rows; that gap is
exactly what produced today's 105 s / 1295 s fallbacks and is why the LSMR-first
default is being replaced.

**Failures / bugs found this session (file:line, `src/` not touched):**
- `src/solvers/QuickDerN.jl:534` + `src/solvers/NullSolvers.jl:1137` -- GramSolver
  undercounts the derivation nullspace at valence 4, d=50 (nullity 3 vs oracle 4);
  known, already being fixed.
- `src/solvers/QuickDerN.jl:105` (default `solver=:AutoSolver`) +
  `src/solvers/NullSolvers.jl:140-153` (`matrix_free_solvers` LSMR-first ordering for
  rectangular maps) -- costs `:Auto` two SylverLining fallbacks (105 s, 1295 s) at
  valence 3, d=150/200; being replaced by an Arpack-first default per section 4.
- Operational, not a Dleto bug: several `bench/jl` background invocations from this
  agent failed instantly with `bash: no such file or directory: bench/jl` (exit 127) --
  an intermittent agent-harness cwd-reset issue between backgrounded Bash calls, not
  tied to any one command pattern. Prefixing with `cd /Users/algeboy/CODE/OpenDleto &&`
  fixed it every time it was tried; flagging in case another agent hits the same thing.

Files: new `bench/Frontier.jl`; new `bench/reports/night-2026-09-03/{frontier-cpu,
video-cpu,sparse-cpu}.csv`.
