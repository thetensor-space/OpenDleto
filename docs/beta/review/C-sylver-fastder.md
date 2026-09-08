# Review C: SylverLining, FastDer3Valent, QuickSylver, and the Metal Sylvester kernel

Branch `beta` at `2d87dc3`, read from `the beta worktree`. All paths below are relative to that root. Nothing was run; every claim is from reading the source.

## 1. What it is

### SylverLining (`src/SylverLining/SylverLining.jl`)
**Math.** A `P`-derivation is an `X = (X_a)` in the operator space `Ω` with `Σ_a P[c,a]·(Γ ×_a X_aᵗ) = 0`. SylverLining builds that linear map `ester: Ω-coords → chisel-rows × tensor` and its adjoint `sylve`, hands the square Gram operator `sylvester = sylve∘ester` to a null solver, and reads the derivations off its null space (`:58-173`).
**Surface.** `SylverLiningMethod(; solver=:AutoSolver, backend=:auto)` (`:38-55`); `derTrOpsReduced(::SylverLiningMethod, Ω, P, Γ; tol, nd, progress, backend, return_diagnostics)` (`:58-79`); `sylvesterLM(Ω, P, Γ; backend)` returning `(derdensor_map, densor_map)` (`:751-759`), exported. Three kernels: `:itensor` reference (`:186-239`), the plain-array zero-alloc kernel with a sparse branch (`:587-721`), `:metal`.
**When chosen.** It is the `:Auto` fallback whenever QuickDer is not applicable or declines (`src/solvers/AutoDer.jl:62-66, 85-101`), the default for every `der(Γ)` / `der(ch, Γ)` convenience overload (`src/Derivations.jl:225-253`), and the default for `den` (`src/Densors.jl:175`).

### FastDer3Valent (`src/solvers/FastDer3Valent.jl`)
**Math.** Liu's quick-der: write the one-row, fully engaged valence-3 condition as `X·R + S·Y − T·Z = 0` with `R = c₁Γ, S = c₂Γ, T = −c₃Γ`, solve the dense system on a corner restriction `(a',b',c')`, lift each restricted null vector to the full axes with three least-squares solves, then verify slice by slice (`:146-349`). The kernel solves over all matrices; `_fastder_restrict_to_ops` intersects the result with `Ω` afterwards (`:460-506`).
**Surface.** `FastDer3ValentMethod(; triple_restriction_size_override, solver, faster_randomized_check)` (`:53-60`); `derTrOpsReduced` at `:613-652`; symbols `:QuickDer3` and `:FastDer3Valent` (`src/Derivations.jl:84-85`). Exported (`src/DletoExports.jl:92`).
**When chosen.** Never by `:Auto`; only by explicit `method=:FastDer3Valent`. It also hosts helpers QuickDerN calls in production: `_fastder_restrict_to_ops` (`src/solvers/QuickDerN.jl:2378`), `_qd_tolerance` (`:2314`).

### QuickSylver (`src/solvers/QuickSylver.jl`)
**Math.** Liu's quicksylver: `X·R + S·Y = T` with a double restriction `(a',b')`, returning an affine frame (particular solution plus offsets) (`:96-192`). With `T = 0`, `R = c₁Γ`, `S = c₂Γ` this is the two-engaged-axis (adjoint / nucleus) derivation condition; the frame's offsets from its first point are the basis (`:273-292`).
**Surface.** `QuickSylverMethod(; double_restriction_size_override, faster_randomized_check)` (`:42-48`); `derTrOpsReduced` at `:238-301`; symbol `:QuickSylver`. Exported.
**When chosen.** Never by `:Auto` (which only knows QuickDer and SylverLining); only explicitly. Refuses anything but a one-row chisel with exactly two engaged axes on `IndTransverseOps`/`UniversalOp` (`:223-236`).

### Metal backend (`ext/DletoMetalSylver.jl`, `ext/DletoMetalExt.jl`)
Accelerates only `sylvesterLM`'s array kernel: the mode-`p` unfoldings and the residual live on the GPU, the operator coordinates (`Σ d_a²`) cross the bus per apply of the square map (`ext/DletoMetalSylver.jl:9-30`). Two axes dodge the permute ("head" and "flat", `:88-114`). Reached by `backend=:metal` through `_sylvesterLM_device(::Val{:metal}, ...)` (`src/SylverLining/SylverLining.jl:784-795`, `ext/DletoMetalSylver.jl:311-312`). It does **not** fall back: `:metal` without the extension errors with a diagnostic (`SylverLining.jl:786-791`), `:auto` never picks the GPU (`:738-740`). Float32 arithmetic always; Float64 tensors are accepted and computed in fp32 (`ext/DletoMetalSylver.jl:79-82`).

## 2. Design assessment

### One interface, four dialects
All four implement `derTrOpsReduced(method, Ω, P, Γ; tol, nd, kwargs...) -> (rΩ, expand_map, coords)`, and `der`/`derReduced`/`derTrOps` are written once on top (`src/Derivations.jl:124-174`). That is the right shape. The drift is in everything around it:

| aspect | SylverLining | FastDer3Valent | QuickSylver | QuickDer |
|---|---|---|---|---|
| `tol` type | `Real` (`SylverLining.jl:62`) | `Real` (`:618`) | `Float64` (`QuickSylver.jl:243`) | `Real` (`QuickDerN.jl:2270`) |
| tolerance meaning | relative, squared for the Gram map, floored by `Precision.jl` (`:141-153`) | relative, floored by `qd_tolerance` (`:631`) | raw, mixed abs/rel, no floor (`:60-72, 122-133`) | relative, floored |
| `return_diagnostics` | yes, 4-tuple (`:77, 172`) | swallowed by `kwargs...` (`:620`) | swallowed (`:245`) | yes |
| `nd > 0` | `:fixed_nd` policy passed to the solver (`:151-153, 167`) | silent column truncation (`:646-648`) | silent truncation (`:295-297`) | warns, refuses after restriction (`QuickDerN.jl:2382-`) |
| eltype of `coords` | storage type `T`, Float16 round-tripped (`:100-103, 155`) | `eltype(Γ)` | always `Float64` (`:61, 261, 273`) | `T` (`QuickDerN.jl:2379`) |
| eltype of `expand_map` | `Tc` | untyped `LinearMap` → Float64 (`:650`) | Float64 (`:299`) | Float64 (`QuickDerN.jl:2190, 2405, 2427`) |
| `rΩ` | engagement-reduced `Ω_reduced` (`:106`) | `Ω` itself | `Ω` itself | `Ω` itself |

So `derReduced` means "coordinates in the engaged sub-space" for SylverLining and "coordinates in `Ω`" for the other three; a Float32 tensor comes back Float32 from SylverLining/QuickDer, Float64 from QuickSylver, and Float32 columns through a Float64 identity map from FastDer3Valent. None of this is stated in the abstract type's docstring, which still says a method "should inherit DerivationMethod and implement der and den" (`src/Derivations.jl:33-39`) -- neither is what a method implements.

### Is FastDer3Valent still needed?
As an *algorithm*, no: QuickDer at valence 3 answers the same cases and SylverLining is already the exact dense oracle `:Auto` falls back to. As a *file*, it cannot move: the production path depends on it. `_fastder_restrict_to_ops`, `_fastder_projector`, `_fastder_gram_diag`, `_fastder_tall_nullspace`, `FASTDER_RESTRICT_CEILING`, `_qd_tolerance`, `_qd_linear_equals_affine`, `_qd_nullspace` are all defined here (`:80-132, 384-598`) and used by QuickDerN (`QuickDerN.jl:2314, 2378`) and QuickSylver (`QuickSylver.jl:181-182`). A file named after a retired method holds the most delicate shared numerics in the package (the Float32 restriction cut). Recommendation: move the shared helpers into `src/solvers/RestrictToOps.jl` (and `_qd_*` into `Precision.jl`/a `SolveLift.jl`), un-export `FastDer3ValentMethod`, and keep the transcription as an oracle reachable through `get_derivation_method(:QuickDer3)` for the comparison tests. Its header (`:10-23`) is a good argument for keeping a line-for-line transcription *somewhere*; that somewhere need not be the public surface.

### Restrict / solve / lift, three times
`_qd_solve_and_lift` (`FastDer3Valent.jl:246-324`), `_qs_solve_and_lift` (`QuickSylver.jl:158-192`), `_qdn_solve_and_lift` (`QuickDerN.jl:1629`). FD and QS share `_qd_linear_equals_affine` but each carry their own `select_restriction_sizes`, `check_solution`, `unpack`, system-matrix builder and `*_solver` wrapper. For FD this is defensible (faithful transcription of a separate reference file); for QS, a production method, it means the tolerance fixes made to FD's copies (`:83-90, 196-221`) were never applied to QS's (`:60-72, 122-133`). See F2.

### Metal gating
Good: an explicit request never silently degrades (`SylverLining.jl:780-782`), `:auto` never picks the GPU, `Metal.functional()` is checked at construction (`ext/DletoMetalSylver.jl:73-75`), the eltype check is explicit (`:79-82`), and the `Val{:metal}` hook is a correct and well-explained answer to the Julia 1.12 method-overwriting problem (`SylverLining.jl:772-778`). Weak: a Float64 `Γ` is computed in fp32 inside a Float64-typed map, and the caller's precision policy does not know (see F1). Also `ext/DletoMetalExt.jl:28-30` includes the kernel files behind `isfile(...) &&` -- if a file is missing, the extension loads fine and `:metal` fails at first use with "no device kernel is registered", which is a build error disguised as a runtime one.

### The zero-alloc apply
This is the best-engineered code in the slice. `_sylver_plan` (`:532-577`) states the layout once and is shared with the device kernel; the unfolding orientation is chosen on measurements and the reasoning is written down (`:554-564`); the flat-axis reshape and the `m == 1` `α`-fold (`:634-645, 654-661`) remove a permute and a gemv from every derivation solve that actually runs; the sparse branch is gated on a constant with its motivation (`:291-304`); every `Operator`/`TransverseOps` has an allocating fallback so new types are correct by default (`:317-321, 480-488, 509-513`); copying through `R` rather than reshaping a view is justified by `StridedMatrix` (`:704-708`). Type stability looks sound: every ternary in the closures picks between values of one concrete type (`:605, 640-645, 663, 693`), no captured variable is reassigned. Two weaknesses: the shared scratch makes the two maps non-reentrant and this is a comment (`:286-289`), not a check; and the CPU kernel keeps `engsize` separate `Wkd` buffers where the Metal twin shares one (`ext/DletoMetalSylver.jl:141-150` says 972 MB vs 324 MB at 300³×3) -- the optimisation was made on the GPU side and not ported back.

### Good decisions worth naming
Store-vs-compute type with Float16 promoted and rounded back (`SylverLining.jl:89-103`); `store_eltype` reaching the verdict (`:146-153`); an empty basis as a mathematical fact rather than an error (`:125-133`); `der_residual` in blocks in the tensor's own type with a fallback when operators cannot be matched to axes (`src/Derivations.jl:343-442`); relative tolerances and scale-aware feasibility in FD (`:83-131, 196-221`); validation by `error` with a message, not `@assert`, in FD/QS (`:600-611`, `QuickSylver.jl:223-236`); the closed-form Gram diagonals that removed 2.5 GB of churn (`:393-430`).

## 3. Findings

**F1 (correctness risk) -- `:metal` computes Float64 tensors in fp32 but the precision policy sees Float64.** `ext/DletoMetalSylver.jl:79-82, 28-30` accept `Float64` and run fp32; `SylverLining.jl:100-101` sets `Tc = compute_eltype(T) = Float64`, so `solve_nullspace` gets `store_eltype = Float64`, a Float64 floor (`Precision.jl:258`), and `tol²` (`:141-144`) -- nine decades below fp32 noise. The verdict machinery built to keep Float16 honest is bypassed for the one backend that lies about its arithmetic. Fix: either refuse `Float64` in `_sylverlm_metal` (consistent with QuickDer's device path, which promotes the *tensor* to Float32 before solving), or have `derTrOpsReduced` choose `Tc = Float32` when `backend === :metal` and convert `Γ` first.

**F2 (correctness risk) -- QuickSylver's verification is absolute; the scale bug FastDer fixed still lives here.** `QuickSylver.jl:126` is `isapprox(X*R_k + S_k*Y, T_k; atol, rtol)` with `T = 0`, i.e. `norm(...) ≤ atol`. A tensor scaled by 1e6 fails and `derTrOpsReduced` throws "did not find a correct affine frame" (`:211-214`); scaled by 1e-6 anything passes. `_qs_lin_solve` (`:66-69`) mixes `atol`/`rtol` the same way. Compare `FastDer3Valent.jl:199-205`, whose comment explains exactly this. The near-degenerate and scale-invariance tests (`test/TestDerivationLaws.jl:186-228`) run only SylverLining, so it is untested. Fix: port `_qd_isrelzero` and `triple_scale` into `_qs_check_solution`, and floor `atol` with `_qd_tolerance`.

**F3 (API) -- `FastDer3ValentMethod.solver` is accepted and never read.** Defined `:55`, defaulted `:59`; the only field reads are `:639-640`. `FastDer3ValentMethod(solver=:KrylovSolver)` silently does nothing. Fix: delete the field, or use it in `_qd_solve_dense`.

**F4 (API) -- `return_diagnostics`, `progress`, `backend` are swallowed by FD and QS.** `FastDer3Valent.jl:620`, `QuickSylver.jl:245`. A caller doing `(rΩ, m, c, rep) = derTrOpsReduced(FastDer3ValentMethod(), ...; return_diagnostics=true)` gets a `BoundsError` on the 3-tuple. `AutoDer` forwards `return_diagnostics` verbatim (`AutoDer.jl:85-101`), so any future route through these would break. Fix: every method either honours `return_diagnostics` (a `DerivationReport` with the lift fields filled and `verdict = nothing`) or errors on unknown kwargs.

**F5 (API) -- `nd` means four things.** Table above. In FD/QS `ders[:, 1:floor(Int, nd)]` (`:646-648`, `:295-297`) truncates a null-space basis that has no preferred order, after a restriction the CONTEXT says a fixed count cannot survive (`docs/CONTEXT.md:270-275`). QuickDer warns; FD/QS do not. Fix: share one `_apply_nd_policy(ders, nd)` that warns, and document `:fixed_nd` as "first `nd` columns of an arbitrary basis" if that is what is meant.

**F6 (API) -- inconsistent `nd` defaults on the abstract path.** `src/Derivations.jl:185` and `:193-194` default `nd=10`; every concrete method and `der` default `nd=-1`. `derTrOpsReduced(Ω, P, Γ)` returns at most 10 vectors while `der(Γ)` returns a basis -- the class of bug already fixed once ("Stop truncating the derivation basis to the tensor valency", commit `2f84350`). Fix: `nd=-1` everywhere; delete the method-less overload the comment at `:192` already says should move.

**F7 (API) -- `der` rejects a Float32 tolerance that `derTrOpsReduced` accepts.** `src/Derivations.jl:128` is `tol::Float64`; `SylverLining.jl:62` is `tol::Real` and `test/TestSylverLining.jl:44` passes `tol=1f-5` directly to `derTrOpsReduced`. `der(m, Ω, P, Γ; tol=1f-5)` is a `MethodError`. `derTrOps` (`:164-174`) also drops `kwargs...`, so `progress`/`backend` cannot reach it. Fix: `tol::Real` and `kwargs...` on all four generic entry points.

**F8 (maintainability) -- the Float32 cut is principled in form, calibrated on one family.** `_fastder_tall_nullspace` (`FastDer3Valent.jl:569-587`) reuses `gap_verdict` with `floor = atol`, `ceiling = 32·atol`, and falls back to the old count when no gap clears -- the same rule as the null solvers and the lift filter, so it is not a one-off patch. But: the ceiling constant is a `Ref` tuned on the scrambled sphere (`:516-527`); a genuine direction above `32·atol` is still dropped; the `m < n` branch (`:573`) has no gap logic at all; and the fallback makes the answer discontinuous in the spectrum (a 99x ratio gives the old count, 101x the new). `test/TestFastDer3Valent.jl:41-93` pins the rule on synthetic spectra, which is good, but no test runs FD or QuickDer end-to-end on a Float32 tensor with non-universal `Ω`. Fix: an end-to-end Float32 `SymmetricOp` case in the tests; consider whether `FASTDER_RESTRICT_CEILING` and `QDN_LIFT_CEILING` should be one constant in `Precision.jl`.

**F9 (correctness risk, minor) -- `engaged(P)` requires a `Matrix`.** `src/Chisels.jl:50` is `engaged(P::Matrix, ...)`; `SylverLining.jl:105`, `FastDer3Valent.jl:608`, `QuickSylver.jl:232` call it on `P::AbstractMatrix`. AutoDer already works around it with `engaged(Matrix(P))` (`AutoDer.jl:64`). A chisel passed as a view or `Adjoint` fails three of four methods. Fix: `engaged(P::AbstractMatrix)`.

**F10 (maintainability) -- sign convention is documented in one comment and tested on one chisel.** `FastDer3Valent.jl:628-635` (`T = -coeffs[3]*G`) and the transpose of `X` at `:361-369` are the two orientation facts. The Z-law with `UniversalChisel(3)` (`test/TestDerivationLaws.jl:59-97`) would catch a wrong sign on slot 3, so the convention *is* tested -- but only for the all-ones chisel; a chisel like `[1, 2, -3]` never reaches FD or QS in any test, and QS's `R = c₁G, S = c₂G` (`QuickSylver.jl:259-260`) is only exercised through `AdjointChisel`. Fix: one Z-law case per method with a random one-row chisel.

**F11 (maintainability) -- misleading claims in comments.** `SylverLining.jl:183, 312-313` say the `:itensor`/array cross-check is in `test/TestSylverLining.jl`; that file never mentions `backend` -- the check is `bench/SylvesterKernelEquivalence.jl:23-29`, outside the test suite. `test/TestDerivationLaws.jl:232-236` says `den` "is still an abstract placeholder" and the T-law tests are "marked broken"; they are live and pass. `src/Derivations.jl:36-38` (see above).

**F12 (style) -- leftovers and length.** Dev-diary comments at `SylverLining.jl:187-190` ("I am gettign stupid error..."); docstring typo "Sylver Lininig" at `:34`; user-facing validation via `@assert` at `:86-87, 535, 550` (compiled out with `--check-bounds=no`-style flags and giving no message to the user) where FD/QS use `error`. `_sylverlm_metal` is 232 lines (`ext/DletoMetalSylver.jl:72-303`), `_sylvesterLM_array` 135 (`:587-721`), `derTrOpsReduced(::SylverLiningMethod)` 116 (`:58-173`, mostly commentary). Naming: `sylvesterLM`/`derTrOpsReduced`/`globalDim` (camelCase, the original coder) against `_fastder_restrict_to_ops`/`der_residual`/`solve_nullspace` (snake_case, the second) -- the two-coder seam runs exactly along the SylverLining / solvers boundary. `_qd_nullspace(M; atol)` passes its `atol` as `rtol` (`FastDer3Valent.jl:87`). `_qd_tolerance(::Type, tol)` (`:81`) is a fallback for non-`Number` types that nothing can reach. The identity `LinearMap` is built five times (`FastDer3Valent.jl:650`, `QuickSylver.jl:299`, `QuickDerN.jl:2190, 2405, 2427`), always untyped (Float64). QuickSylver computes an affine frame and subtracts a zero particular solution (`:105-119, 273-278`) where a nullspace would do.

## 4. Test coverage

What `test/TestDerivationLaws.jl` proves:
- **Z-law on random tensors**: only `SOLVERS = [:SylverLining, :FastDer3Valent]` (`:53`), one `(4,5,3)` Float64 tensor, `UniversalChisel(3)`, generators and random combinations (`:56-97`). QuickDer is **not** in this loop; its Z-law lives in `test/TestQuickDerN.jl` (`:160-330`, including a Float32 case at `:178-199`). QuickSylver: one `6³` tensor, three adjoint pairs, plus the refusal of three engaged axes (`:102-127`).
- **Near-degenerate and scale invariance** (`:186-228`): SylverLining only. This is the test that would expose F2.
- **T-law and Galois adjunction** (`:288-314, 427-454`): the derivations come from SylverLining on the diagonal `3³` tensor; `den` is method-independent, so this is a test of `den`, not of the methods.
- **stratify laws** (`:331-388`): `:SylverLining` and `:QuickDer`; the residual is checked on `der(:SylverLining, ..., res.Σ)`, so QuickDer's own output is only indirectly tested here.
- `der_residual` library vs test copy (`:472-538`), Float32 no-promotion -- strong.

`test/TestFastDer3Valent.jl`: `:15-27` asserts `size(ders,2) >= 1` and `rΩ === Ω` on one `4³` tensor with `UniversalOp` -- so `_fastder_projector`, `_fastder_gram_diag` and the non-trivial branch of `_fastder_restrict_to_ops` (`:480-505`) have no test in this file; `:41-93` pins the gap rule on synthetic spectra (good); `:95-103` checks only `ndims` and `length`.

`test/TestSylverLining.jl`: transpose and composition laws on random tensors, valence 3-5, random local ops, trivial and non-trivial symmetries (`:119-161`) -- the strongest test in the slice for the map pair, and the Float32 eltype test (`:27-48`). But: default backend only; no `:itensor` vs `:array` equivalence; no sparse `Γ` (random tensors are dense, so the `SYLVER_SPARSE_DENSITY` branch is untested in `test/`); no `:metal` anywhere in `test/` (`test/TestQuickDerDevice.jl` covers QuickDer's device path only). Failures inside `testTranspose`/`testComposition` are `@assert`s (`:68, 80`), so one failure aborts the testset instead of being counted.

Missing, in priority order: (1) a shared case list (random, near-degenerate, scaled, Float32, non-universal `Ω`, random one-row chisel) run through **every** registered method; (2) the kernel-equivalence check moved from `bench/` into `test/`; (3) a Float32 end-to-end FD/QuickDer case with `SymmetricOp`; (4) a sparse `Γ` case for the array kernel; (5) a `:metal` test gated on `Metal.functional()`.

## 5. Extensibility

Adding a fifth method today means: a struct `<: DerivationMethod`; a `derTrOpsReduced` method whose kwargs and return shape you infer from four differing examples; a new branch in the hard-coded `if/elseif` in `get_derivation_method` (`src/Derivations.jl:77-92`) and its docstring list (`:46-75`) and error string (`:89-91`); an `export`; and remembering to add the symbol to `SOLVERS` in the tests. `AutoDer` cannot use it: applicability is a hand-written predicate for QuickDer (`AutoDer.jl:62-66`) and failure is `try/catch` (`:86-97`).

A minimal contract, written once on the abstract type:

```julia
# every method implements exactly this
derTrOpsReduced(m::M, Ω::TransverseOps, P::AbstractMatrix, Γ::ITensor;
                tol::Real, nd, progress=false, return_diagnostics=false)
    -> (rΩ::TransverseOps, expand::LinearMap{T}, coords::Matrix{T})
       or the same plus a DerivationReport when return_diagnostics
# invariants a generic test can check for any M:
#   T == eltype(Γ) (storage type; Float16 allowed to compute in Float32)
#   size(coords,1) == globalDim(rΩ); size(expand) == (globalDim(Ω), globalDim(rΩ))
#   der_residual(Γ, embedITensors(Ω, expand(coords[:,i])), P) <= tol for every i
#   nd <= 0: a basis; nd > 0: :fixed_nd with the ONE documented meaning
#   unknown kwargs error
# optional trait for AutoDer, replacing try/catch:
supports(m::M, Ω, P, Γ)::Bool
# registration, replacing the if-chain:
method_symbol(::Type{M})::Symbol  (or a Dict populated at include time)
```

With that, `TestDerivationLaws` becomes a loop over `subtypes(DerivationMethod)` rather than a hand-maintained list, and the table in section 2 collapses to one row.
