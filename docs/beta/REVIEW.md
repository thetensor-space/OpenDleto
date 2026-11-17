# Code review of OpenDleto `beta`

Reviewed 2026-09-08 against `beta` at `2d87dc3` (tag `v1.5-beta-2026-09-04`), 113 commits ahead
of `main` at `40b20d5`. The source tree is 15,147 lines across `src/`, `ext/`, `test/`; beta
added 11,713 of them. Five slice reviews were made in parallel and are attached in full under
[review/](review/); this document is the consolidated result. Companions:
[CHANGELOG.md](CHANGELOG.md), [DESIGN-CHOICES.md](DESIGN-CHOICES.md),
[NEW-FEATURES.md](NEW-FEATURES.md), [COMING-FEATURES.md](COMING-FEATURES.md).

Method: reading only. No file on `beta` was modified. The test suite was run once on the beta
worktree under the budgeted wrapper (`JL_PROJECT=…/OpenDleto-beta bench/jl test/runtests.jl`).

---

## 1. Verdict

**Beta is a large, real improvement over `main` and is in a mergeable state for a beta
channel.** The suite is green (51 testsets, 14,279 passes, zero failures, zero errors; `main`
errored after 2,460 passes). The new solver reaches sizes `main` could not approach, the densor
exists, and the answers now come with a verdict. The commit history and the design record are
unusually good.

The review found **no defect that produces a wrong certified answer on the default path**
(`stratify` → `:Auto` → QuickDer, Float64). It found:

- **4 bugs** that give a wrong result or crash on a non-default but supported path;
- **9 correctness risks** where the machinery built to keep answers honest is bypassed on one
  route;
- a **consistent pattern**: a good idea (the precision policy, the verdict, seeding, the
  `derTrOpsReduced` seam, the registry) was implemented in one place and not carried through
  the other places that needed it;
- **repository hygiene** problems that would block a public release: 13 MB of notebook output
  and a corrupted notebook, orphan logs with machine paths, a wrapper that tests the wrong
  worktree by default, and a `Project.toml` a registry would reject.

Recommended reading order for the author: §3 (bugs) and §4 (correctness risks) first; they are
short and each has a one-line fix. §7 is the ordered work list.

### Status, 2026-09-08 (branch `fix-you/beta-bugs-b1-b4`, off the review branch)

All four bugs and all twelve correctness risks below were fixed the same day, at the
author's direction, with a regression test for each. Test-suite result on the fix branch:
52 testsets, 14,374 passes, 0 failures, 0 errors (Julia 1.12 via `bench/jl`, Arpack present); the review's baseline was 51 testsets and 14,279 passes.

| Item | Fix applied |
|---|---|
| B1 | `coordinates(::EmptyOp)` tests every entry; negative membership tests added. |
| B2 | QuickSylver's consistency and verification checks are relative to the data's scale (as FastDer's already were); `atol` floored by the precision policy; scale-invariance test at 1e-6…1e6. |
| B3 | `GramSolver` and `LSMRSolver` take `seed`, declare `wants_seed = true`; reproducibility tests for both. |
| B4 | The Transpose testset calls `testTranspose`. |
| C1 | `QuickDerDeclined <: Exception` thrown at the four deliberate decline sites; `:Auto` catches only that, rethrows everything else, logs the fallback at `@warn`. Asserting on that warning exposed that the existing "`:Auto` recovers" test never reached QuickDer (the 12³ sphere is under `AUTODER_MIN_ENTRIES`); it now sets `min_entries = 0`. |
| C2 | `backend = :metal` sets the compute type to Float32 whatever the tensor's type; verdict and report follow. |
| C3 | `den` raises on an empty answer from a non-`:ok` solve; `nd` defaults to `-1` (a basis), as for `der`. |
| C4 | `store_eltype` defaults to the compute type only where no narrower type promotes to it; a Float32 map without it is refused (`promotes_to`, exported). |
| C5 | `:CGSolver` is dropped below Float64 on both map shapes. |
| C6 | `kwargs...` removed from the QuickDer, FastDer3Valent and QuickSylver methods; the contract's two per-call keywords (`progress`, `return_diagnostics`) are named on every method, and the two oracles refuse `return_diagnostics = true` with a message instead of returning a 3-tuple; `:Auto` routes `backend` to SylverLining only and has no sink. |
| C7 | `NullVerdict(v; field = value, …)` keyword copy-constructor in `NullSolvers.jl`; both positional rebuilds use it. |
| C8 | `QDN_LAST_SOLVE_STATUS` deleted; `_qdn_empty_result` takes the status from the kernel's returned `info`. |
| C9 | `denLM` is generic in the element type (follows the derivations, chisel converted to match); `den` passes `store_eltype`. |
| C10 | `ArpackDenseSolver` deleted; small dense maps go to `SVDSolver`. |
| C11 | The pirated `Base.:*` / `Base.:+` methods are replaced by the named verb `act(Γ, Xs)` (exported); `stratify`, `randomize_tensor`, tensor synthesis and the SphereLab notebook updated; the swapped forms, the scalar chain and the `+` methods had no users and are gone. |
| C12 | `realCanonicalForm` recognises conjugate pairs by `imag(λ)`, checks LAPACK's pairing, rejects non-real input; regression tests for the nearly defective case, a genuine pair, a mixed spectrum and the symmetric path. |

The remaining sections describe the code as reviewed, before these fixes.

---

## 2. The test run

| | |
|---|---|
| Julia | 1.12 (real binary via `bench/jl`) |
| Testsets | 51 |
| Passes | 14,279 |
| Failures / errors / broken | 0 / 0 / 0 |
| Wall time | dominated by `Testing Tensor Synthesis` (1 m 15 s of ~4 min); every other testset under 20 s |
| Solver extensions loaded | KrylovKit, IterativeSolvers (always), **Arpack** (because `bench/jl` stacks the `@dleto-bench` environment) |
| Library output to stdout during the run | 490 lines of `Using XSolver...` / `Loading ... Extension` |
| Warnings | 5 distinct (all intentional demonstrations in tests) |

Two things the numbers do not show. First, the suite is not the same in every environment: with
Arpack present `AutoSolver` prefers it; without, it falls to KrylovKit. CONTEXT.md already
records a case where the two paths gave different answers ("two code paths, one testset"). A
CI run without Arpack is the missing control. Second, three `@test_broken` in
`TestDerivationLaws.jl` and sixteen `@test_skip` arms in `TestQuickDerN.jl` are stale
scaffolding from before QuickDer landed and should be deleted rather than left inflating the
skipped count.

---

## 3. Bugs (wrong result or crash on a supported path)

| # | Where | What | Fix |
|---|---|---|---|
| B1 | `src/ops/OperatorImpls.jl:120` | `coordinates(::EmptyOp, M)` inspects only the first column: `M[1:dim]` is linear indexing, repeated `dim` times. A matrix with a zero first column and nonzero elsewhere is accepted as the empty operator. The test only feeds `zeros`. | `all(__isapproxzero, M)`; add a negative test. |
| B2 | `src/solvers/QuickSylver.jl:126` | Verification is absolute (`isapprox(…, 0; atol)`), the scale bug FastDer already fixed at `FastDer3Valent.jl:199-205`. A tensor scaled by 1e6 throws "did not find a correct affine frame"; scaled by 1e-6 anything passes. | Port `_qd_isrelzero`/`triple_scale`; floor `atol` with `qd_tolerance`. |
| B3 | `src/solvers/NullSolvers.jl:1592`, `ext/DletoIterativeSolversExt.jl:126` | `GramSolver` and `LSMRSolver` draw their start from the global RNG yet declare `wants_seed = false`. LSMR leads the rectangular matrix-free order, so seeded `den` is not reproducible; `GramSolver` is QuickDer's dense route above `QDN_GRAM_MIN_COLS`. `TestSolverSeed.jl:71` asserts the false premise. | Add `seed`, `wants_seed = true`, `MersenneTwister(seed)`. |
| B4 | `test/TestOperators.jl:110-114` | The "Transpose" testset calls `testInverse`; `testTranspose` never runs. | One word. |

## 4. Correctness risks (honesty machinery bypassed on one route)

| # | Where | What | Fix |
|---|---|---|---|
| C1 | `src/solvers/AutoDer.jl:90-97` | `:Auto` catches **every** exception except `InterruptException`, logs at `@info`, and falls back to SylverLining. A `MethodError`, `BoundsError` or `OutOfMemoryError` inside QuickDer is invisible, and an OOM on `d^n` triggers a fallback whose operator is `n·d^{n+1}`. | `struct QuickDerDeclined <: Exception`; throw at the four deliberate decline sites; catch only that; `rethrow()` the rest; log the decline at `@warn`. |
| C2 | `ext/DletoMetalSylver.jl:79-82` + `SylverLining.jl:100-101` | `backend = :metal` accepts Float64 and computes in fp32, but `Tc = compute_eltype(Float64) = Float64`, so the null solver gets a Float64 floor and `tol²` nine decades below fp32 noise. The Float16-honesty machinery is bypassed for the one backend that lies about its arithmetic. | Refuse Float64 on `:metal`, or set `Tc = Float32` when `backend === :metal`. |
| C3 | `src/Densors.jl:148-150` | `den` destructures `(vals, vecs)`, drops the verdict, and returns `ITensor[]` on an empty basis whatever `status` is. This is the "failed solve read as no solutions" the `status` field was added to prevent. Docstring says `nd = -1`, signature says `nd = 10`. | Check `verdict.status`; default `nd = -1`. |
| C4 | `src/solvers/NullSolvers.jl:795` | `store_eltype` defaults to the compute type. That default is the one that yields a false certificate on promoted Float16; the fix commit patched callers instead of the default. | Require it when `eltype(L) !== compute_eltype(store)`, or carry it on the map. |
| C5 | `src/solvers/NullSolvers.jl:141-153` | `:CGSolver` is excluded below Float64 for square maps but not rectangular ones; Float32 LOBPCG's block-Gram Cholesky is known to collapse regardless of shape. | Apply the eltype filter in both branches. |
| C6 | `src/solvers/QuickDerN.jl:2274`, `FastDer3Valent.jl:620`, `QuickSylver.jl:245` | `kwargs...` accepted and never read. `derTrOpsReduced(m, …; whiten = false)` or `sizes = …` is a silent no-op; `AutoDer` forwards the same kwargs to both routes, so `backend = :metal` meant for SylverLining is dropped when QuickDer answers. `return_diagnostics = true` on FastDer/QuickSylver returns a 3-tuple and a 4-way destructure fails. | Remove `kwargs...` or error on unknown keys; AutoDer splits by destination. |
| C7 | `src/solvers/QuickDerN.jl:1574-1577` | `NullVerdict` rebuilt positionally with 17 arguments (five `Float64`s and four `Int`s in a row). Any same-typed insertion mis-assigns silently. | Keyword copy-constructor in `NullSolvers.jl`. |
| C8 | `src/solvers/QuickDerN.jl:254-259, 1733, 2181, 2305` | `QDN_LAST_SOLVE_STATUS` is stale global state; its justification was invalidated when `_qdn_solve_and_lift` started returning `info`, which is in scope at both call sites. A concurrent call or an error between set and read makes it wrong. | Pass `info.verdict.status`; delete the `Ref`. |
| C9 | `src/Densors.jl:81, 103-105` | `denLM` hard-codes `Vector{Float64}` and untyped `LinearMap`s; the Float32/Float16 direction stops at the densor. | `LinearMap{T}` with `T` from the tensor. |
| C10 | `ext/DletoArpackExt.jl:111-117` | `ArpackDenseSolver` calls `eigs(M; sigma = 0.0)`, factoring `M − 0·I = M`, singular by construction on every input this package produces; returns no `converged` field. | Delete, or implement shift-invert with a nonzero relative shift. |
| C11 | `src/DletoBase.jl:85-179` | `Base.:+` and `Base.:*` defined on `ITensor × AbstractArray` / `Vector{ITensor}`. None of the types are owned by Dleto: type piracy that can change other packages' behaviour. | Named verb (`act`, `contract_frame`) or an owned wrapper type. |
| C12 | `src/DletoBase.jl:219` | `realCanonicalForm` detects conjugate pairs by proximity of the real parts of eigenvectors, not by `imag(λ)`; repeated real eigenvalues with near-equal eigenvectors are misclassified. Flagged in the September review (RISK3) and still open. | Test `imag(λ)`. |

---

## 5. API and interface findings

**A1. Four dialects of one interface.** All five methods implement `derTrOpsReduced`, which is
right, but the table below is what a caller actually meets. Fix by writing the contract once on
the abstract type (a proposed form is in [review/C-sylver-fastder.md](review/C-sylver-fastder.md) §5)
and by a generic test that loops over every registered method.

| aspect | SylverLining | FastDer3Valent | QuickSylver | QuickDer |
|---|---|---|---|---|
| `tol` type | `Real` | `Real` | `Float64` | `Real` |
| `nd > 0` | `:fixed_nd` policy | silent truncation | silent truncation | policy; warns and refuses after restriction |
| `return_diagnostics` | yes | swallowed | swallowed | yes |
| eltype of `coords` | storage `T` | `eltype(Γ)` | always `Float64` | `T` |
| `rΩ` | engagement-reduced | `Ω` | `Ω` | `Ω` |

**A2. `tol` is `Float64` at the top.** `der(Γ; tol::Real)` forwards into
`der(method, …; tol::Float64)` (`Derivations.jl:225 → 128`), so `der(Γ; tol = 1f-6)` is a
`MethodError`; `stratify` and `QuickSylver` also demand `Float64`; `den` papers over it with
`Float64(tol)`. Make it `Real` everywhere.

**A3. `nd` and default-method drift.** The abstract `derTrOpsReduced` overloads default
`nd = 10` (`Derivations.jl:185, 193`) while every wrapper and concrete method default `−1`;
`den` defaults `nd = 10`. The truncation class fixed in commit `2f84350` survives on one path.
`der(Γ)` and `den(Γ)` default to `:SylverLining`, `stratify` to `:Auto`. Unify on `−1` and `:Auto`.

**A4. Closed factories.** `get_derivation_method` is an `if/elseif` (`Derivations.jl:77-92`), so
a user-added method cannot be reached by symbol; `AutoDerMethod` hard-codes exactly two fields;
`matrix_free_solvers` is a literal list, so a registered third-party null solver is never chosen
by `AutoSolver`. Copy the `register_solver!` pattern to methods; give `AutoDer` a
`Vector{DerivationMethod}` with an `applicable` trait.

**A5. `solver` means different things per branch.** QuickDer's dense branch ignores
`method.solver` and picks SVD/Gram by size (`QuickDerN.jl:1688-1689`); the matrix-free branch
honours it. `FastDer3ValentMethod.solver` is never read at all. Two default solvers exist:
`solve(L, sym = :SVDSolver)` vs `solve_nullspace(L, solver = :AutoSolver)`.

**A6. Inner-solver options cannot pass through `AutoSolver`.** It forwards `kwargs...` to
`SVDSolver`, which accepts none, so `maxiter`/`krylovdim` throw on the dense branch and work on
the matrix-free one; `tol` is consumed by `solve_nullspace` and never forwarded, so the
extensions' un-floored `1e-10`/`1e-12` defaults are what always run.

**A7. `rule`/`status` are `Symbol`s.** A typo in `verdict.status == :not_converged` is silently
`false`. `@enum`, or validate in an inner constructor.

**A8. Export list.** Internals exported: `unsafe_*` (whose ITensor variants differ in
*semantics*, not just checks), `embedITensorsSwapped`, `sylvesterLM`, `denLM`, `PROGRESS_TAGS`,
`progress_spec`, `save`, `compare`, `⊕` (still carrying the author's "what is this?" comment).
Not exported: `der_residual`, which its own docstring calls "the first thing a consumer runs".
`DletoExports.jl:81-83` still says `den` is an abstract placeholder; every file-name comment in
the list is wrong.

**A9. `stratify` surface.** `ivec` default changed from `0` to `−1` (random, unseeded) on beta,
documented nowhere; `stratify(Γ::AbstractArray)` does not accept `ivec`; `reduced` is accepted
and ignored; the result is an anonymous `NamedTuple` that discards `δ`; a local `der = …`
shadows the exported function. Decision 2 (a `Stratification` type) is pending.

**A10. Signatures too narrow.** `engaged(P::Matrix)`, `normalize_chisel(P::Matrix)`,
`randTensorChisel(…, ch::Matrix)`; AutoDer already writes `engaged(Matrix(P))` to get past it. The
file's own TODO at `Chisels.jl:28` says so.

---

## 6. Maintainability findings

**M1. Policy in global `Ref`s.** Eleven in `QuickDerN.jl` alone (`QDN_DENSE_BUDGET_BYTES`,
`QDN_GRAM_MIN_COLS`, `QDN_LIFT_CEILING`, `QDN_TRIVIAL_MAX_BYTES`, …) plus
`FASTDER_RESTRICT_CEILING`, `GRAM_GPU_FACTOR`. Per-call policy that is process-global and not
thread-safe; tests save/restore in `try/finally`. Move policy to `QuickDerMethod` fields and
outputs (`QDN_APPLY_COUNT`, stage times, trivial space) into `DerivationReport`, which already
has `stage_times`.

**M2. Include-order cycle.** `QuickDerN.jl` names `NullVerdict`, `gap_verdict`, `GramSolver` in
bodies only and is included before `NullSolvers.jl`; `NullSolvers.jl` calls `_qdn_stage!`. Four
comments work around it by dropping annotations and duplicating constants, one on a false
premise (`GAP_RATIO` *is* in scope; `Precision.jl` is included first). Move the stage/budget
knobs to a small file included after `Precision.jl`, then restore the annotations. More
generally: one types file first, then methods.

**M3. Shared numerics in a retired method's file.** `_fastder_restrict_to_ops`,
`_fastder_tall_nullspace`, `_qd_tolerance`, `_qd_linear_equals_affine` live in
`FastDer3Valent.jl` and are used in production by QuickDer and QuickSylver. Move to their own
file; un-export `FastDer3ValentMethod`; keep `:QuickDer3` as the oracle.

**M4. Restrict/solve/lift written three times.** `_qd_solve_and_lift`, `_qs_solve_and_lift`,
`_qdn_solve_and_lift`, each with its own size selection, check, unpack. Tolerance fixes made to
one copy were not made to another (B2).

**M5. Magic constants outside `Precision.jl`.** Extensions: `tol = 1e-10`/`1e-12`,
`lsmr_tol = 1e-12`, `rank_tol = 1e-8`. Core: `cg_solve tol = 1e-10`, `shift_rel = 1e-10`,
`cgtol = 1e-4` (Float32 CG spins to `maxiter`), `GRAM_SHIFT_REL`, `engaged` cutoff `1e-6`,
`realCanonicalForm tol = 1e-10`, `nondeg tol = 1e-10`, `+ 1e-15` in TensorSynthesis,
hand-written `sqrt(eps(RT))` at four sites in QuickDerN. `LUSolver` uses `max(m,n)·eps` directly
though `Precision.jl:147` says it uses `rank_rtol`. `QDN_LIFT_CEILING` and
`FASTDER_RESTRICT_CEILING` are the same constant (32) with the same reasoning.

**M6. Stale documentation of tuned constants.** `Precision.jl:42, 158, 285, 396-398, 433-436`
still describe `FLOOR_EPS` as 100 or 5; `tol_default(Float64; squared = true)` is documented as
1.8e-15 in the code and in `Precision-Policy.md` but evaluates to 1e-12. Comments claim tests
that do not exist (`SylverLining.jl:183, 312-313` point to a kernel cross-check that lives in
`bench/`, not `test/`); `TestDerivationLaws.jl:232-236` calls live tests "marked broken".

**M7. `println` in library code.** Nine sites in `NullSolvers.jl` and the three solver
extensions; three `__init__` banners fire on every `using Dleto`. 490 lines in one test run.
`@debug`, or the progress channel that already has an `io`.

**M8. Long functions.** `_qdn_solve_and_lift` 400 lines (with five commented seams already),
`solve_nullspace` 272, `_sylverlm_metal` 232, `derTrOpsReduced(::QuickDerMethod)` 144,
`_sylvesterLM_array` 135, `solve(::KrylovSolver)` 125, `solve(::GramSolver)` 115.

**M9. Bench-only kernel code in the solver file.** `_qdn_ttm_square!` has no `src/` caller;
with `_qdn_ttm`, `_qdn_ttm!`, `_qdn_unfold(!)`, `_qdn_fold`, `_qdn_slice` it is the contraction
seam the header promises. A `src/util/ModeProducts.jl` shrinks `QuickDerN.jl` by ~350 lines and
lets SylverLining's array path share it.

**M10. Older core anti-patterns (inherited, not created by beta).** Abstract-typed struct
fields (`Chisel.ch::AbstractMatrix`, `idx::Dict{Index,Integer}`, `IndTransverseOps.val::Integer`,
`localOps::Vector{<:Operator}`); `Int8` block indices (valence capped at 127) and `Int16` maps;
`@assert` for user-facing validation (compiled out under some flags, no message);
element-by-element ITensor copies in `__asMatrix`; a non-`const` global alias
`side_by_side = compare`; `__init__` mutating `ENV["WEBIO_WARN"]`; blanket `using ITensors` in a
util file included last being what makes `hasind`/`replaceind` resolve in core files; dead code
calling undefined `randomOrthogonalMatrix`/`ArrayToITensor` in `TensorSynthesis.jl`; two temp-index
helpers, two random-basis-change implementations, two `__ITensor`/`__asITensor` with different
tag schemes.

**M11. Performance items with a named fix.** The lift walks a fresh chain from `G` per `(a,b)`
pair where the cross sketches share a prefix (`_qdn_pair_tensor`), so it pays `n(n−1)` full
passes where 2 would do, and CONTEXT already names the lift as the new wall on the device. The
lift RHS recomputes `_qdn_ttm(Hs[b], Yv[b,i], b)` once per chisel row (3× for `CentroidChisel(3)`).
`progress = true` wraps the map in a `FunctionMap`, which defeats `_gram_dense`'s no-copy path
(1–3 GB copied because progress was requested). The CPU Sylver kernel keeps `engsize` separate
scratch buffers where the Metal twin shares one (972 MB vs 324 MB at 300³×3).

**M12. Naming.** The two-coder seam: camelCase (`derTrOpsReduced`, `embedITensors`,
`globalDim`, `sylvesterLM`, `randSurfaceTensor`) against snake_case (`solve_nullspace`,
`der_residual`, `get_derivation_method`, `rand_den`); `IndTransverseOps` prefix vs
`TransverseOpsSymmetries` suffix; `Independant` (file) and `Incompatable` (~30 messages in 9
files); `_qd_tolerance` and `qd_tolerance` both exist; dev-diary comments ("I am gettign stupid
error…", `SylverLining.jl:187-190`); stale headers naming other files on `DletoExports.jl`,
`DletoBase.jl`, `Random.jl`, `Nondegenerate.jl` ("?????"), `ChiselImpls.jl`, two bench scripts.

---

## 7. Test coverage

**Strong.** Oracle-anchored numerics for QuickDer (valence 3/4/5, sphere, degenerate modes,
whitening identities, `:fixed_nd`, diagnostics with independently recomputed residuals);
equational laws (Z-law, T-law, Galois adjunction, `denLM` adjoint, stratify as change of frame);
the precision policy per type incl. an `@allocated` no-promotion check; verdict rules on
hand-built spectra; seed reproducibility for Arpack/KrylovKit/LOBPCG; SylverLining transpose and
composition laws at valence 3–5.

**Gaps, in priority order.**

1. One shared case list (random, near-degenerate, scaled, Float32, Float16, non-universal `Ω`,
   random one-row chisel) run through **every** registered method. Today the Z-law loop covers
   only `[:SylverLining, :FastDer3Valent]`; near-degenerate and scale tests run SylverLining
   only (which is why B2 is untested); QuickSylver gets one 6³ tensor.
2. Float16 on the CPU through `:QuickDer`: no test pins `certified == false` /
   `undecidable > 0` on a near-degenerate Float16 tensor, nor `eltype(ders) === Float16`; the
   only Float16 QuickDer case is behind `if DEVICE_TESTS`.
3. Direct tests for `LUSolver`, `GramSolver`, `LSMRSolver`, `LanczosSolver`, `ShiftInvertSolver`,
   `ArpackDenseSolver`; an "every registered solver agrees on the same matrix" sweep; a full null
   space (`L = 0`) and a legitimate empty derivation algebra (`P = I(3)` on a random 3-tensor).
4. Error paths: `_qdn_check_sizes`, the 1.5× retry and final failure, `device = :gpu` without a
   backend and with Float64, `solve_nullspace(L, :Unknown)`, `ShiftInvertSolver(:Missing)`.
5. Valence 5 with an actual restriction (the one case has `r == d` on every axis, so no sketch,
   lift or whitening runs); `:corner` positive case (today `corner_ok`'s `catch → true` passes on
   any exception); `_qdn_ttm!` with `α/β`, `_qdn_ttm_square!`, wide-system padding.
6. Core: `EmptyOp` on a non-zero matrix (B1), `testTranspose` (B4), negative membership,
   `normalize_chisel` (no caller anywhere), `Chisel`/`reduceByEngaged(::Chisel)` (no caller),
   `embedITensorsSwapped`, `nondeg`, `⊕`, `stratify` with `ivec` variants, `den` with
   `TransverseOpsSymmetries`. `test/old-tests/TestDleto.jl` holds the only multiplication and
   nondegeneracy tests; port them and delete the directory.
7. Suite mechanics: `JULIA_TEST_MODE`/`TEST_MODE` is dead; core tests are `@assert`-in-helper
   (a failure is an Error, not a Fail, and only the first is reported); three copies of
   `der_residual` and other helpers across test files; tests `include` `bench/SphereHarness.jl`;
   `TestPrecision.jl` uses `Logging` which is not a test dependency; a tautology at
   `TestPrecision.jl:87`; `SolverProgress.jl` has no tests; the "TestSolverSeed is the only file
   that loads extensions" premise is false since `Dleto.jl` imports the triggers.

---

## 8. Repository hygiene (blocks a public release)

Sizes from `git ls-tree -l` on `beta`; details and a proposed layout in
[review/E-hygiene.md](review/E-hygiene.md).

| | |
|---|---|
| Tracked tree | `main` 6.35 MB / 54 files → `beta` 26.6 MB / 214 files |
| Of the 20 MB added, not code or data | ~13.3 MB |
| `labs/FastDerSphereComparison.ipynb` | 13.0 MB: 7 KB of source, 8.2 MB of embedded plotly/HTML output |
| `labs/WWEIA2.ipynb` | 5.0 MB: **3,413 byte-identical copies** of one markdown cell. A generation loop, not a result. |
| `bench/**/*.log` | 16 files; 15 referenced nowhere; contain `/Users/algeboy/…` and `.claude/worktrees/agent-…` paths |
| Genuinely valuable results | the dated `bench/reports/2026-09-04/*/README.md` and small CSVs, ~0.65 MB |

Must fix before release:

1. Dedupe or delete `WWEIA2.ipynb`; strip outputs from `FastDerSphereComparison.ipynb`; add an
   `nbstripout` filter for `labs/*.ipynb`.
2. `bench/jl:32` defaults `PROJECT` to the primary worktree, so `bench/jl test/runtests.jl` run
   inside the beta checkout **tests `main`** unless `JL_PROJECT` is set. Default to the repo
   containing the script. (`bench/SylvesterKernelBench.jl:12` has the same absolute path in an
   `include`.)
3. `git rm` the orphan `.log` files; add `*.log`, `.ipynb_checkpoints/`, `.claude/worktrees/`,
   `.claude/settings.local.json` to `.gitignore`; remove the orphan `!` line.
4. `README.md` vs `docs/Installing-Dleto.md` contradict each other ("now a Julia package, Julia
   1.12" vs "not a formal package, Julia 1.7"); README lists the weakdep `Arpack` as
   auto-installed; the WhatWeEat Binder URL lacks `labs/`; `labs/geometry/SphereLab.{html,pdf}`
   do not exist.
5. `Project.toml`: version still 0.1.0 after 113 commits and API changes; no `julia` compat;
   no compat for `KrylovKit`, `LinearMaps`, `PlotlyBase`, `PlotlyKaleido`, `Random`,
   `SparseArrays`, `Arpack`, `Metal`; `Logging` used by a test but not a test dep; `IJulia`,
   `CSV`, `DataFrames`, `JSON`, `Plotly*`, `Plots` are hard deps of a numerics library;
   `IterativeSolvers`, `KrylovKit`, `Plots` are hard deps *and* extension triggers.
6. `.claude/settings.json` (new on beta) whitelists bare `julia --project=.`, the thing
   `bench/jl` exists to prevent. Move to `settings.local.json`.
7. No CI, no root `CHANGELOG.md`, no `CITATION.cff`.

Should fix: `docs/CONTEXT.md` has no TOC, the 2026-09-04 material is in three non-adjacent
sections, and the superseded movie-cost numbers (lines 887–924) are live text above their
correction; no docs index; `Refactor-Plan.md` still says nothing has been implemented; bench
output locations are inconsistent and several committed CSV names no longer match what the
scripts write; `night-2026-09-03/BOARD.md` (1,206 lines) is an agent journal to mine and archive.

Nothing private was found: no tokens, no personal emails, the coordination directory and the
downstream project are not named in tracked files.

---

## 9. What is good, and should not be touched in the clean-up

- The sketch-restrict / whiten / solve / lift / verify pipeline, with verify as a veto.
- The stored type threaded positionally into the restricted verdict; `data_floor` vs
  `precision_floor` as two independent floors; `status` vs `certified` as two independent vetoes.
- `DerivationReport` as one struct for three routes with `nothing` meaning "no such number".
- Composition-not-matrix squaring; the byte-based dense gate; the LSMR "return one column
  above threshold so the caller can bracket" rule; CPQR replacing LU with a written post-mortem;
  Gram shift escalation upward from the floor; `check = false` on Cholesky rather than try/catch.
- `_sylver_plan` and the zero-allocation array kernel shared with the GPU; the `Val{:metal}`
  hook; "an explicit backend request never silently degrades".
- `_qdn_ttm!` never permuting on the host; `_qdn_ttm_square!` in one buffer; the trivial space
  published factored with a byte budget; `der_residual` blocked in the tensor's own type.
- `den` as a rectangular `LinearMap` reusing `applyDerivation`, so Z-law and T-law agree by
  construction.
- Registering solvers inside `__init__` with the precompile reason stated.
- Commit messages, the dated bench READMEs, and the design notes that mark decision vs proposal.

---

## 10. Ordered work list

Small, high-value, no design decisions needed:

1. B1, B4 (one line each); B3 (`seed` on Gram/LSMR); C3 (`den` checks `status`); C8 (delete the
   stale `Ref`); C10 (delete `ArpackDenseSolver`); M7 (`println` → `@debug`).
2. `tol::Real` everywhere (A2); `nd = -1` everywhere (A3); `kwargs...` either consumed or
   rejected (C6); `NullVerdict` keyword copy-constructor (C7); QuickSylver relative check (B2).
3. C1: `QuickDerDeclined` so `:Auto` stops swallowing bugs and OOMs.
4. C2: refuse Float64 on `:metal` or compute in Float32 knowingly.
5. Hygiene items 1–3 and 6 of §8 (notebooks, `bench/jl` default, logs, settings).

Medium, one session each:

6. Constants sweep into `Precision.jl` (M5, M6); stale docstrings.
7. Include-order fix via a knobs file; restore annotations (M2).
8. Move shared numerics out of `FastDer3Valent.jl`; un-export the method (M3).
9. Policy `Ref`s → `QuickDerMethod` fields; outputs → `DerivationReport` (M1).
10. Method registry + `applicable` trait for AutoDer (A4); one contract on the abstract type (A1)
    and a generic per-method law test (§7.1).
11. `Project.toml`: decide deps vs weakdeps, add compat, bump version (§8.5); CI.

Larger, already decided in the design record:

12. `Chisel` as `(𝕋, Ω, P)` keyed by `Index` (Refactor-Plan Phase 1).
13. `Stratification` return type and the `σ_{e+1}` verdict.
14. Extract the contraction kernel to `util/ModeProducts.jl` and the one chisel-weighted kernel
    with a choice of unknown slot (Phase 2).

---

## Appendix: the five slice reviews

- [review/A-quickder.md](review/A-quickder.md): `QuickDerN.jl`, `AutoDer.jl`, `DerivationReport.jl`, the Metal QuickDer extension.
- [review/B-nullsolvers-precision.md](review/B-nullsolvers-precision.md): `NullSolvers.jl`, `Precision.jl`, `SolverProgress.jl`, the three solver extensions.
- [review/C-sylver-fastder.md](review/C-sylver-fastder.md): `SylverLining.jl`, `FastDer3Valent.jl`, `QuickSylver.jl`, the Metal Sylvester kernel.
- [review/D-core-spine.md](review/D-core-spine.md): the type system, `Derivations.jl`, `Densors.jl`, exports, `Project.toml`, core tests.
- [review/E-hygiene.md](review/E-hygiene.md): repository, `bench/`, `labs/`, `docs/`.
