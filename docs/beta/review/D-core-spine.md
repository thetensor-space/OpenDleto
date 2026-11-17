# Review D — core type system and package spine (`beta` vs `main`)

Scope: `src/Dleto.jl`, `DletoExports.jl`, `DletoBase.jl`, `Project.toml`, `Chisels.jl` + `chisels/ChiselImpls.jl`,
`Operators.jl` + `ops/OperatorImpls.jl`, `TransverseOperators.jl` + `ops/TransverseOps*.jl`, `Derivations.jl`,
`Densors.jl`, `util/*.jl`, the core tests and `test/runtests.jl`. Read-only; no Julia was run.
All paths are relative to `the beta worktree`.

Diff footprint in this slice (`git diff --stat main beta`): Densors +253, Derivations +417, Dleto +41, Exports +29,
Operators/OperatorImpls (Vector -> AbstractVector widening only), TransverseOps* (+swapped embedding for
symmetries, eltype-parametrised `reduceByEngaged`), ChiselImpls (rename + two dead-function repairs), Project.toml
(Random/SparseArrays to deps, Metal weakdep+ext, ProgressMeter dropped).

## 1. What it is

**Type hierarchy (as of beta).**

| Layer | Types | Where |
|---|---|---|
| Chisel `P` | *untyped*: `UniversalChisel`, `TuckerChisel`, `AdjointChisel`, `CentroidChisel` all return a bare `Matrix` | `src/Chisels.jl:86-172` |
| Framed chisel | `struct Chisel{ch::AbstractMatrix, frames::Vector{Index}, idx::Dict{Index,Integer}, ch_axis::Index}` (renamed from `ChiselFramed`) | `src/chisels/ChiselImpls.jl:39-55` |
| Local operators | `abstract type Operator`; `UniversalOp, DiagonalOp, SymmetricOp, AntiSymmetricOp, ScalarOp, EmptyOp` (all singleton structs) | `src/Operators.jl:41`, `src/ops/OperatorImpls.jl:38-64` |
| Transverse operators | `abstract type TransverseOps`; `IndTransverseOps` (independent per-axis), `TransverseOpsSymmetries` (fused/dual axes via `syms`/`duals`) | `src/TransverseOperators.jl:55`, `ops/TransverseOpsIndependant.jl:43`, `ops/TransverseOpsSymmetries.jl:46` |
| Derivation methods | `abstract type DerivationMethod`; `SylverLiningMethod`, `FastDer3ValentMethod`, `QuickDerMethod`, `QuickSylverMethod`, `AutoDerMethod` | `src/Derivations.jl:39`; structs in `SylverLining.jl:38`, `solvers/FastDer3Valent.jl:53`, `QuickDerN.jl:437`, `QuickSylver.jl:42`, `AutoDer.jl:37` |
| Densor | no type; `denLM` returns `(A::LinearMap, AtA::LinearMap, fr, dims)`; `den` returns `Vector{ITensor}` | `src/Densors.jl:68-152` |
| Null solvers | `abstract type NullSolver` + registry (`register_solver!`, `solve_nullspace`) | `src/solvers/NullSolvers.jl:20,379,781` (other reviewer's slice) |

**Operator interface.** `Operator` exposes `coordinates`/`unsafe_coordinates` (matrix -> native vector or `nothing`),
`embed`/`unsafe_embed` (vector -> matrix), `transposeEmbed`/`unsafe_transposeEmbed` (adjoint of embed),
`dualize`, `localDim`, `containScalars` (`src/Operators.jl:48-93`). `TransverseOps` lifts the same six verbs to a
list of per-axis matrices or two-index `ITensor`s, adds `embedITensorsSwapped` (temp index first), `globalDim`,
`axisDims`, `valency`, `frames`, `framesTemporary`, and `reduceByEngaged -> (rΩ, expand::LinearMap)`
(`src/TransverseOperators.jl:61-173`). The safe verbs are generated from the unsafe ones by adding size checks
(`embed` -> `unsafe_embed` at `Operators.jl:69-72`).

**DerivationMethod interface.** One required method: `derTrOpsReduced(method, Ω, P, Γ; tol, nd, ...) ->
(rΩ, expand_map, reduced_der_coords)` (`src/Derivations.jl:180-189`). Everything else — `der`, `derReduced`,
`derTrOps` — is derived from it in `Derivations.jl:124-174`. `get_derivation_method(::Symbol; kwargs...)` is the
factory (`Derivations.jl:77-92`).

**Public API as exported** (`src/DletoExports.jl`): chisel builders + `Chisel`, `applyDerivation`; the six
`Operator` verbs with `unsafe_*` twins; `TransverseOps` verbs incl. `embedITensorsSwapped`; precision policy
(`compute_eltype`, `tol_default`, ...); solver registry (`solve_nullspace`, `register_solver!`, `AutoSolver`, ...);
`DerivationMethod`, `get_derivation_method`, `der`, `derReduced`, `derTrOps`, `derTrOpsReduced`, `den`,
`DerivationReport`; method types; `stratify`, `denLM`; IO stubs (`compare`, `save`, `plot_tensor`, ...);
synthesis (`rand_den`, `randSurfaceTensor`, ...); `gpu_*`; and `⊕`.

**Tensor -> stratification path.** `stratify(Γ)` (`Densors.jl:278`) builds `UniversalChisel(n)` and
`IndTransverseOps(frame, UniversalOp())`, then `stratify(Ω, ch, Γ; method=:Auto)` (`Densors.jl:228`) ->
`get_derivation_method(:Auto)` -> `AutoDerMethod` -> `derTrOpsReduced(::AutoDerMethod)` (`AutoDer.jl:68`), which
tries `QuickDerMethod` when `Ω isa IndTransverseOps`, some axis is engaged and the tensor has >= `min_entries`
entries, and otherwise (or on any non-interrupt exception) falls back to `SylverLiningMethod`
(`AutoDer.jl:84-100`). SylverLining builds `sylvesterLM` and calls `solve_nullspace(L, :AutoSolver)`. Back in
`stratify`, one derivation is chosen (`ivec`: random combination / column), expanded via `expand_map`,
embedded with `embedITensors(Ω, ...)`, put in real canonical form (`realCanonicalForm`, `DletoBase.jl:201`) and
contracted against `Γ`; indices are retagged back to the original frame (`Densors.jl:217-222`).

## 2. Design assessment

**Good decisions (keep).**
- *`derTrOpsReduced` as the single required method.* All five methods implement exactly it (`SylverLining.jl:58`,
  `FastDer3Valent.jl:613`, `QuickSylver.jl:238`, `QuickDerN.jl:2265`, `AutoDer.jl:68`); `der`/`derReduced`/`derTrOps`
  are free. This is a real seam: adding a method is one struct + one method + one `elseif` in the factory.
- *`den` implemented, and as a rectangular `LinearMap` with a genuine adjoint* (`Densors.jl:68-107`). Reusing
  `applyDerivation` for the forward map makes Z-law and T-law consistent by construction (`Densors.jl:1-20`), and
  dropping the `__needsSquare` policy in favour of letting the solver square the map (`Densors.jl:109-113`) is
  the right division of labour. The Galois adjunction is testable and is tested (`test/TestDerivationLaws.jl:282-304`).
- *`AutoDerMethod` as composition* (`quick`, `fallback`, `min_entries`; `AutoDer.jl:37-41`) rather than a flag
  on one method; exceptions from QuickDer are treated as "not applicable", `InterruptException` is rethrown.
- *Widening `Vector{<:Number}` -> `AbstractVector{<:Number}` across Operators/TransverseOps* (whole diff of
  `Operators.jl`, `OperatorImpls.jl`, `TransverseOperators.jl`) lets views and LinearMap outputs flow without copies.
- *`reduceByEngaged(..., ::Type{T}=Float64)`* so the expand map has a declared eltype
  (`TransverseOpsIndependant.jl:127-161`) — needed for Float32/Float16 work.
- *`der_residual` blocked in the tensor's own eltype* (`Derivations.jl:343-442`): the Z-law check is library
  code, memory-bounded, and does not promote.
- *`stratify` retag* (`Densors.jl:217-222`) so a stratified tensor can be re-chiseled — a real bug fixed.
- *Dead-on-arrival code repaired rather than deleted* with the reason recorded (`ChiselImpls.jl:61-68`,
  `TransverseOpsSymmetries.jl:102-105,153-158`).

**Weak spots.**
- *`der`/`den` are not yet a symmetric pair.* `der` defaults `nd=-1` (full basis) while `den` defaults
  `nd=10` (`Densors.jl:136,174`) — the T-set is silently truncated by default. `der(Γ)` defaults to
  `:SylverLining` (`Derivations.jl:225-228`) while `stratify` defaults to `:Auto` (`Densors.jl:235`), and
  `den(Γ; method=:SylverLining)` (`Densors.jl:175`). Three entry points, two default methods. `den` ignores `Ω`'s
  operator restriction entirely ("supplying the frame", `Densors.jl:129`), so the 𝕋 slot of the design's
  `(𝕋, Ω, P)` chisel is still absent — as `ChiselImpls.jl:34-37` admits.
- *The `Chisel` type is still `P` plus a frame*; the four builders return bare `Matrix`, so nothing dispatches on
  chisel kind and `AdjointChisel(valence, left, right)` remains positional (the exact hazard
  `docs/review/Refactor-Plan.md` §1.2 names). `normalize_chisel` still has no caller (`grep` over src/test/bench/labs).
- *Package spine.* `Project.toml` makes `IJulia`, `PlotlyJS`, `PlotlyBase`, `PlotlyKaleido`, `Plots`, `CSV`,
  `DataFrames`, `JSON` hard deps (`Project.toml:7-19`) of a numerics library; `Plots` is at once a hard dep and an
  extension trigger (`Project.toml:19,32`), as are `KrylovKit`/`IterativeSolvers` (`Project.toml:11,13,29-30`). An
  extension whose trigger is unconditionally imported (`src/Dleto.jl:41-42`) is just a module with an extra
  indirection and a load-order hazard; it also relies on Julia >= 1.11 semantics for triggers that live in
  `[deps]` rather than `[weakdeps]`, and there is **no `julia` compat entry at all**. `[compat]` is missing for
  `LinearMaps`, `KrylovKit`, `PlotlyBase`, `PlotlyKaleido`, `Random`, `SparseArrays`, `Arpack`, `Metal`
  (`Project.toml:34-43`). `__init__` mutates the user's `ENV["WEBIO_WARN"]` for a plotting stack that is not even
  loaded (`Dleto.jl:122-124`), next to commented-out `Plots.plotlyjs()` code (`Dleto.jl:44-47,126-133`).
- *Include order is load-bearing and documented as such* (`Dleto.jl:69-72,100-105`). The comments are honest but
  the cure is structural: put all `struct`/`abstract type` definitions in one types file included first, then
  methods. Today adding a `::DerivationReport` annotation to any signature in `QuickDerN.jl` breaks the load, and
  the three `derTrOpsReduced` methods had to *drop* their return-type annotations to cope (`QuickDerN.jl:2276-2282`).
- *Namespace imports are scattered and order-dependent.* `hasind`/`replaceind` used in `Densors.jl:45,218-221` and
  `Derivations.jl:354` resolve only because `util/TensorSynthesis.jl:27` does a blanket `using ITensors` — a util
  file included *after* the core files (`Dleto.jl:115`). `DletoBase.jl:27` imports a curated list; `Dleto.jl:34`
  does `import ITensors`. Pick one (curated `using ITensors: ...` in `Dleto.jl`) and delete the others.
- *Export list coherence.* `unsafe_*` variants exported (`DletoExports.jl:49-62`) double the surface for
  functions whose only difference is a size assert; the "unsafe" ITensor variants also differ in *semantics*
  (they ignore index orientation, `TransverseOperators.jl:81-82,112-113`), which is not what a user infers from
  the name. `embedITensorsSwapped` is an implementation detail of the transpose convention (Refactor-Plan §1.3)
  and should not be public. Very generic names are exported: `save`, `compare`, `frames`, `valency`, `engaged`,
  `⊕` — the last still carrying the author's own "what is this?" comment (`DletoExports.jl:37-38`).

## 3. Findings (ranked)

**Bugs**
1. `coordinates(::EmptyOp, M)` checks only the first column. `src/ops/OperatorImpls.jl:120`:
   `all(__isapproxzero, vcat([M[1:dim] for i=1:sizes[1]]...))` — `M[1:dim]` is linear indexing of the first `dim`
   entries, repeated `dim` times. A matrix with a zero first column and nonzero elsewhere is accepted as the
   empty operator. Fix: `all(__isapproxzero, M) || return nothing`. The test only feeds `zeros(dim,dim)`
   (`test/TestOperators.jl:97-98`), so it cannot catch this.
2. `test/TestOperators.jl:110-114` — the "Transpose" testset calls `testInverse`, not `testTranspose`;
   `testTranspose` never runs. One-word fix.
3. Type piracy on `Base.:+` and `Base.:*` for foreign types: `Base.:+(::ITensor, ::AbstractArray)`,
   `Base.:+(::AbstractArray, ::ITensor)` (`src/DletoBase.jl:171-179`), `Base.:*(::ITensor, ::Vector{ITensor})`,
   `Base.:*(::AbstractMatrix, ::Vector{ITensor})`, `Base.:*(::ITensor, ::Vector{<:Number})` etc.
   (`DletoBase.jl:85-144`). None of the argument types are owned by Dleto, so these can change behaviour of
   other packages and are invalidation magnets. Fix: a named verb (`act(Γ, Xs)` / `contract_frame`) and keep `*`
   only for a Dleto-owned wrapper type, or make `Stratification`/`Frame` types that own the operator.
4. `tol` type mismatch across entry points: `der(Γ::ITensor; tol::Real)` (`Derivations.jl:225`) forwards to
   `der(method, ...; tol::Float64)` (`Derivations.jl:128`), so `der(Γ; tol=1f-6)` or `tol=1//10^6` is a
   `MethodError`; `den(Γ)` papers over it with `Float64(tol)` (`Densors.jl:180`); `stratify` demands `Float64`
   (`Densors.jl:232,280,305`); `QuickSylver` demands `Float64` (`QuickSylver.jl:243`) while the other four methods
   take `Real`. Fix: `tol::Real` everywhere, convert once inside the solver.

**Correctness risks**
5. `denLM` hard-codes `Float64`: `out = Vector{Float64}(undef, ...)` (`Densors.jl:81`) and the two `LinearMap`
   constructors carry no eltype (`Densors.jl:103-105`, default `Float64`). A Float32/complex tensor is promoted or
   errors on assignment. Fix: `T = promote_type(eltype(Γ-storage), eltype(P))`, `LinearMap{T}(...)`.
6. `stratify` default changed from `ivec=0` (deterministic column) on main to `ivec=-1` (random `randn`
   combination) on beta (`Densors.jl:236,285` vs `main:Densors.jl:36,73`), unseeded, and `ivec` is documented
   nowhere (docstring at `Densors.jl:184-197` predates it). `ivec==0` means "column `min(valence, n_ders)`"
   (`Densors.jl:262-264`), which is not what "fixed column 0" suggests. Fix: document; take an `rng`; make the
   deterministic choice the default or return the coefficients in the result.
7. `stratify(Γ::AbstractArray; ...)` (`Densors.jl:303-312`) does not accept `ivec`; passing it falls into
   `method_kwargs` and reaches the method constructor, which rejects it. `reduced=false` on
   `stratify(Γ::ITensor)` (`Densors.jl:284`) is accepted and ignored. Fix: unify the keyword set with a single
   `stratify(Ω, ch, Γ; ...)` and thin wrappers that forward `kwargs...` untouched.
8. `abstract derTrOpsReduced` default is `nd=10` (`Derivations.jl:185,193`) while every wrapper and every
   concrete method defaults `nd=-1`; the `der` docstring still says "default: 10" (`Derivations.jl:117`).
9. `Chisel` inner constructor promises "test that all elements of frame are different" (`ChiselImpls.jl:48`) and
   does not; a repeated `Index` silently overwrites `idx`. `Γ_frame_ch` is computed and unused
   (`ChiselImpls.jl:84`).
10. `realCanonicalForm` (`DletoBase.jl:201-231`) detects conjugate pairs by proximity of the *real parts of
    eigenvectors* (`DletoBase.jl:219`), not by `imag(λ)`; repeated real eigenvalues with near-equal eigenvectors
    are misclassified as complex. Also flagged in `docs/review/OpenDleto-vs-Magma.md` (RISK3) and still open.

**API design**
11. `DletoExports.jl:81-83` still says `den` "is still an abstract placeholder (it asserts false for every
    method)". Stale: `den` is implemented at `Densors.jl:135`. Every file-name comment in the export list is also
    stale: "DletoUtil.jl" (`:36`, it is `DletoBase.jl`), "OperatorsImpls.jl", "Transverse.jl",
    "GlobalOperatorsIndependant.jl", "GlobalOperatorsSymmetries.jl", "DerivationMethodSylverLininig.jl",
    "NonDegenerate.jl" (`:33,55,58,65,68,90`). Header says "Strata Dleto: Dleto.jl / Main module"
    (`DletoExports.jl:2-3`).
12. Naming: camelCase `derTrOpsReduced`, `embedITensors`, `randTensorChisel`, `randSurfaceTensor` vs snake_case
    `solve_nullspace`, `get_derivation_method`, `der_residual`, `rand_den`, `randomize_tensor`,
    `normalize_chisel`; prefix `IndTransverseOps` vs suffix `TransverseOpsSymmetries`; file `Independant`
    (misspelt) and `Incompatable` in ~30 assert messages (`grep -ril incompatable src` hits 9 files). Pick one
    convention now, before the public surface grows; deprecate with `Base.depwarn` aliases.
13. `sylvesterLM`, `denLM`, `embedITensorsSwapped`, `unsafe_*`, `PROGRESS_TAGS`, `progress_spec` are exported
    internals (`DletoExports.jl:49-62,78,91,95`). Suggested public core: chisels, operator/transverse types,
    `der`/`den`/`stratify`, method types + `get_derivation_method`, `solve_nullspace` + registry,
    `DerivationReport`, `der_residual`. `der_residual` is *not* exported although the docstring calls it "the
    first thing a consumer runs" (`Derivations.jl:302-303`).
14. Signatures too narrow: `engaged(P::Matrix)`, `normalize_chisel(P::Matrix)`, `__dist(P::Matrix, ::Vector)`
    (`Chisels.jl:50,60,75`), `randTensorChisel(..., ch::Matrix)` (`TensorSynthesis.jl:125,128`) — the file's own
    `#TODO replace Matrix with AbstractMatrix` at `Chisels.jl:28`. `AutoDer.jl:72` already has to write
    `engaged(Matrix(P))` to get past it.
15. `stratify` returns an anonymous `NamedTuple` and discards `δ` (the derivation actually used) and the
    coefficients (`Densors.jl:273-274`); docstring says `Σ::AbstractArray` (`Densors.jl:186`) while the annotation
    says `ITensor` (`:201`). Refactor-Plan decision 2 (a `Stratification` type) is still pending.

**Maintainability / Julia anti-patterns**
16. Abstract-typed struct fields everywhere in hot types: `Chisel.ch::AbstractMatrix{<:Number}`,
    `frames::Vector{Index{K}} where K`, `idx::Dict{Index,Integer}` (`ChiselImpls.jl:40-43`; the constructor
    actually builds `Dict{Index,Int16}`, `:49`); `IndTransverseOps.val::Integer`, `axisDims::Vector{<:Integer}`,
    `localOps::Vector{<:Operator}` (`TransverseOpsIndependant.jl:44-51`); same in `TransverseOpsSymmetries`
    (`:47-57`) plus `blocks::Vector{Vector{Int8}}` (valence capped at 127) and `Int16` index maps
    (`TransverseOpsIndependant.jl:132-133`). Fix: parametrise (`struct Chisel{M<:AbstractMatrix}`), use `Int`.
17. `side_by_side = compare` is a non-const global (`util/TensorIO.jl:40`) and is exported. `const`.
18. `__asMatrix`/`__asMatrixTranspose` copy element-by-element through scalar ITensor indexing
    (`TransverseOperators.jl:182-202`); `Array(T, i, j)` / `permutedims` does it in one call.
19. `@info "Found ... derivations"` unconditionally inside a library function (`Densors.jl:254`); local variable
    `der = ders*coefs` shadows the exported function `der` (`Densors.jl:261,264,267`), as does the positional
    argument `der::Vector{ITensor}` (`Densors.jl:200`).
20. Stale/misnamed headers: `DletoBase.jl:2` "Utils"; `Random.jl:2-3` "Utils / ?????"; `Nondegenerate.jl:3`
    "?????"; `ChiselImpls.jl:2` "Framed Chisels" (after the rename); `TransverseOpsIndependant.jl:2` "Transverse
    Operators"; `Densors.jl` has no licence header at all; `DletoExports.jl:2`. Floating module-level docstrings
    attached to nothing (`Chisels.jl:30-44`, `Operators.jl:27-40`, `TransverseOperators.jl:29-54`) because the
    `# module` lines are commented out (`Chisels.jl:179`, `TensorIO.jl:26`).
21. Duplicated utilities: two random-basis-change implementations (`util/Random.jl:39-52` `randomize_tensor` and
    `util/TensorSynthesis.jl:36-76` `__randomize_tensor`/`randomizeITensor*`, the latter calling undefined
    `randomOrthogonalMatrix`, `randomInvertibleMatrix`, `ArrayToITensor` — dead code); two "make a temp index"
    helpers (`__globalOpsMakeTempIndex`, `__new_index_for_change_of_basis`, `__new_index_for_randomization`);
    `__ITensor` (`DletoBase.jl:36`) vs `__asITensor` (`Derivations.jl:214`) with different tag schemes
    (`"a1"` vs `"a_1"`).

## 4. Test coverage

`test/runtests.jl` includes, in order: `TestDletoBase`, `TestChisels`, `TestOperators`, `TestTransverseOps`,
`TestTensorSynthesis`, `TestTensorDensity`, `TestFastDer3Valent`, `TestQuickDerN`, `TestQuickDerDevice`,
`TestAutoDer`, `TestDerivationLaws`, `TestNullVerdict`, `TestPrecision`, `TestSylverLining`, and last
`TestSolverSeed` (which loads the KrylovKit/IterativeSolvers/Arpack extensions and so changes the solver registry
for anything after it — hence "LAST, on purpose", `runtests.jl:34-37`). Beta wired in nine new files and
re-enabled `TestSylverLining` (diff of `runtests.jl`).

`JULIA_TEST_MODE` is dead: `TEST_MODE` is defined at `runtests.jl:14` and never read; the assert/test switch is a
commented-out block (`:43-58`). The header instructions (`:3-6`) are therefore wrong. Remove both.

Core tests are `@assert`-inside-helper-returning-`true` wrapped in `@test` (`TestChisels.jl:9-28`,
`TestOperators.jl:9-29`, `TestTransverseOps.jl:13-32`). A failure surfaces as an *Error* not a *Fail*, only the
first failing case is reported, and `@assert` is not guaranteed to run at all optimisation levels. Convert to
`@test` per property.

Gaps in this slice: `normalize_chisel` (noted at `TestChisels.jl:7`), `Chisel` construction/`reduceByEngaged(::Chisel)`
(never called anywhere), `embedITensorsSwapped` for either transverse type, `coordinates` returning `nothing` on
non-members (`TestOperators.jl:3` TODO), `EmptyOp` on a non-zero matrix (finding 1), `testTranspose` (finding 2),
`randomize_tensor`, `nondeg`, `⊕`, `match_idx`, `__ITensor` (the old `testMultiplication`/`testRandomization` sit
commented out at `TestDletoBase.jl:22-54`); `stratify` with `ivec` variants; `den` with `TransverseOpsSymmetries`.
`TestDletoBase.jl`/`TestOperators.jl` use `LinearAlgebra.*` without a `using LinearAlgebra` anywhere before them
in `runtests.jl` (`TestDletoBase.jl:14`, `TestOperators.jl:40`) — resolves only through ITensors' exports.

`test/old-tests/` (4 files) is abandoned and stale against the API: `UniversalOps`, `member`, `transverse`
(`old-tests/TestTransverseOps.jl:33-35`), `randomize` (`old-tests/TestTensorSynthesis.jl:9`),
`TensorIO.loadTensor`/`derden`/`NullSolvers.solve` (`old-tests/TestDerivations.jl:12-23`), plus a
`Pkg.activate` inside a test file (`:6`). `old-tests/TestDleto.jl` holds the only multiplication / nondegeneracy
tests (`:10-64`) and those *would* still run — port `testMultiplication`, `testRandomization`, `testDegeneracy`
into `TestDletoBase.jl`, then delete the directory.

## 5. Extensibility

- **New derivation method: yes, almost.** Subtype `DerivationMethod`, implement `derTrOpsReduced` — no core edit
  needed, *except* that `get_derivation_method` is a closed `if/elseif` (`Derivations.jl:77-92`), so
  `der(:MyMethod, Γ)` and `stratify(...; method=:MyMethod)` cannot reach it. Fix: a `Dict{Symbol,Type}` registry
  mirroring `register_solver!`.
- **New local operator kind: yes.** Subtype `Operator`, implement the six verbs (`Operators.jl:48-93`);
  `IndTransverseOps` composes them generically. Only `unsafe_dualize` has a fallback; the rest assert.
- **New transverse-operator kind: yes, with a caveat.** The abstract verbs are all there, but `AutoDer` only
  fires for `Ω isa IndTransverseOps` (`AutoDer.jl:71`) and QuickDer validates the same, so a new `TransverseOps`
  silently gets SylverLining only.
- **New chisel kind: no.** Chisels are bare matrices; there is no type to subtype and nothing dispatches on kind.
  A user can pass any `AbstractMatrix` (which is the design win over Magma), but cannot attach behaviour
  (representative policy, transpose convention, `show`). This is Refactor-Plan §1/§2 and remains the largest
  open design item in the core.
- **New null solver: yes** via `register_solver!` — the pattern the method factory should copy.

## Summary of recommended fixes, in order
1. `coordinates(::EmptyOp)` (bug) and the `testTranspose` wiring (bug); add negative-membership tests.
2. `tol::Real` everywhere; `nd` default unified; `den` default `nd=-1`; one default method for `der`/`den`/`stratify`.
3. `denLM` eltype-generic; `Chisel`/`*TransverseOps` fields concrete; `Int` not `Int8`/`Int16`.
4. Move the `Base.:+`/`Base.:*` piracy behind a named verb or an owned wrapper type.
5. Prune exports (drop `unsafe_*`, `embedITensorsSwapped`, `sylvesterLM`, `⊕`-with-comment, `save`); export `der_residual`; fix stale comments.
6. `Project.toml`: plotting/IJulia/CSV/DataFrames/JSON to weakdeps or a companion package; `[compat]` for every dep plus `julia`; drop `ENV["WEBIO_WARN"]` and the commented Plots code.
7. One types file first in `Dleto.jl`; curated `using ITensors: ...` at module top; delete blanket `using ITensors` in util files.
8. Symbol registry for derivation methods; document `ivec` (or replace with `rng`/`select`); `Stratification` type.
9. Delete `test/old-tests/` after porting `TestDleto.jl`; remove `JULIA_TEST_MODE`; convert `@assert` helpers to `@test`.
