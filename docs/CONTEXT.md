# OpenDleto — working context for AI chats

Purpose: hand this file to a new chat so it starts with the context of prior sessions.
Keep it updated as work progresses. Last updated 2026-09-02 (session 2: Phase 0 + densor +
solver repair).

## What this package is

OpenDleto is the Julia implementation of the *chiseling* algorithms for detecting sparsity
patterns in tensor data. It is the Julia sibling of the Magma packages at
<https://github.com/thetensor-space>.

Governing papers (in the authors' hands, not all public):

| Paper | Role |
|---|---|
| `null_patterns.pdf` | Defines what this software does: chisels `C`, sparsity patterns `Delta(C, delta)`, `Der(C, Gamma)`, Algorithms 1 and 2 |
| `Densor.pdf` (*Tame Traits of Tensors*) | The mathematical model underneath: the ternary Galois connection on T-sets / I-sets / Z-sets, derivations, densors, Theorem A |
| `Rihana.pdf` (*Exact sequences of inner automorphisms of tensors*) | Extends valence; situates derivations among nuclei and centroids via Rosenberg–Zelinsky-style exact sequences |
| `der-densor.pdf` (*Tensor isomorphism by conjugacy of Lie algebras*) | An application of densor spaces: isomorphism testing |

Key correspondence to remember: a **chisel is the linear-polynomial special case of the
polynomial ideal `P`** in the Densor T/I/Z framework. `Der(C, Gamma) = Z(Gamma, P)` for the
linear ideal `P` spanned by the rows of `C`.

## Sister Magma packages

`TensorSpace` (core types), `Sylver` (`src/Invariants.m` — `DerivationAlgebra(t,A,k)`,
`Centroid`, `Nucleus`, `AdjointAlgebra`), `Densor` (`DerivationClosure`, `NucleusClosure`,
`UniversalDensorSubspace`), `Stratacastor`, `Homotopism`, `MatrixAlgebras`.

Magma parametrizes invariants by `(A, k)` — a subset of coordinates plus an integer. OpenDleto
parametrizes by an explicit `m x l` chisel matrix, which is **strictly more expressive** (it can
express the inequivalent chisel families of `null_patterns` §8.3 that `(A,k)` cannot). Preserve
that advantage in any refactor.

## How the code reached its current state

Two developers with opposing strategies worked on it and overwrote each other — one an
incremental "add as you go" problem-solver, one an over-designer. History is on GitHub. The
result is a package that has both bare untyped helpers and deep abstract-placeholder hierarchies
(`@assert false` stubs), plus dead code and exports of names that were never defined. Treat
inconsistency as collision fallout, not intent.

**Current goal:** plan a refactor that strikes the balance between those two styles and leaves
room to extend the package with new applications.

## Stated aims so far

- OpenDleto should be a state-of-the-art Julia implementation.
- T-sets, I-sets and Z-sets from `Densor.pdf` must be easy to define in the model — especially
  the linear-polynomial case, i.e. chisels.
- Multiple selectable derivation solvers (the `FastDer3Valent` work is a first instance).
- Extensibility for new applications.

**The authoritative statement of intent is [`docs/Dleto-Design.md`](Dleto-Design.md)** — read it
first. It defines chisel / derivation / densor, fixes the substrate vocabulary (axes, base, frame,
tensor space, interpretation, tensor), states that T and Z are the same system with a different
unknown slot, asks for a solver factory with defaults, and sketches the intended file layout. Its
own open TBDs: a dual/covariant axis marker, and `U → hom(V,W)`-style interpretations.

The refactor plan derived from it is [`docs/review/Refactor-Plan.md`](review/Refactor-Plan.md).

## Findings

Full review with mermaid diagrams: [`docs/review/OpenDleto-vs-Magma.md`](review/OpenDleto-vs-Magma.md).
Findings marked **[FIXED]** were repaired in session 2; see *Work completed* below.

Headlines:
- **[FIXED] T-sets were absent.** `Densors.jl` contained only `stratify`; `den` was an
  `@assert false` stub. The package named for the densor could not compute a densor.
- **I-sets are still absent.**
- **Z-sets exist** and are more general than Magma's, but chisels are untyped bare matrices and
  no solver applies `normalize_chisel`.
- **Numerics diverge from the paper**: Algorithm 2 says take the SVD of `N`; the code
  eigen-decomposes `N'N`, squaring the condition number. The `sigma_(e+1)` test that decides
  "this tensor admits no pattern for this chisel" is never performed. *(Still open. Session 2
  measured the consequence: see "Conditioning is the binding constraint" below.)*
- **`stratify` throws away `delta`**, i.e. the sparsity pattern itself, and applies no canonical
  eigenvalue ordering. *(Still open.)*
- **[FIXED]** Dead code: `ChiselImpls.jl:57,61` (undefined `Fch`, `enggaged`),
  `Derivations.jl:88`, `export der` with no definition.
- No sanity-check harness *(now partly addressed: equational-law tests, see below)*, no invariant
  caching, no `Homotopism` / `TensorOverCentroid` analogue.
- **[FIXED] `FastDer3Valent` solved the wrong chisel** — the reference solves
  `X·R + S·Y − T·Z = 0`, so the third slot enters negated, and the port turned chisel
  `[c₁,c₂,c₃]` into `[c₁,c₂,−c₃]`. Fixing the sign was not enough: the system-matrix
  construction was wrong in all three blocks and a fabrication heuristic masked it. Replaced by a
  line-for-line transcription of the reference.
- **[FIXED] `FastDer3Valent` dropped the reference's solution verification**, which exists
  precisely because solve-and-lift is only generically correct at a given `(a',b',c')`.

## Sibling repo: `../fast-der-solver`

Chris Liu's repo (grad-show poster 2025) holding **two** solve-and-lift algorithms. Both restrict
to a small corner, solve densely, then lift by small least-squares solves; both are valence-3
only, which is acceptable. **Both are now ported.**

| Liu's file | What it solves | Restriction | OpenDleto name |
|---|---|---|---|
| `quick-der-lib.jl` | derivations `XR + SY − TZ = 0` | all three axes | `:QuickDer` (alias `:FastDer3Valent`) |
| `quicksylver-lib.jl` | Sylvester with RHS `XR + SY = T` | two axes, affine frame | `:QuickSylver` |

The naming was a live source of confusion: `:FastDer3Valent` made the derivation lift sound like a
different family from `:QuickSylver` when it is the *flagship derivation member of the same
family*. `:QuickDer` is now the primary name, matching Liu's own `quick-der` / `quicksylver`
split; `:FastDer3Valent` is kept as an alias.

`quicksylver-lib.jl` also has an ITensor variant, `quick-der-itensor-lib.jl` (the newest file in
that repo), which differs only in using `pinv` for the lifts instead of `linear_equals_affine`.
Author's own benchmark `quicksylver-vs-dleto-results.csv`: at ~6 minutes wall clock,
solve-and-lift reached `n = 315` vs the then-current OpenDleto path at `n ≈ 68` and a dense
baseline at `n ≈ 33`.

## Work completed (session 2)

Branches, in order. Each is a task branch off the previous.

**`i-can-see-clearly-now/phase-0-green-baseline`** — get the suite green.
- `Project.toml`: `Random` moved from `[weakdeps]` to `[extras]` with a `[targets]` test entry;
  `IJulia`/`ProgressMeter` weakdeps with no `[extensions]` entry dropped.
- `test/TestSylverLining.jl` ported from the pre-rename API (`Local*Ops`→`*Op`,
  `TransverseOpsIndependant`→`IndTransverseOps`) and re-enabled in `runtests.jl`. It checks the
  transpose law `⟨a, f'(B)⟩ = ⟨f(a), B⟩` and `f'∘f == derdensor_map`.
- `ChiselImpls.jl` dead code fixed; `ChiselFramed` renamed to **`Chisel`**.
- `SylverLininig.jl` → `SylverLining.jl`; scratch files removed.

**`signed-sealed-delivered/der-and-derivation-law`** — the T-set, the laws, the solvers.
- **`der`** (Z-set) replaces `derITensor`; three tiers (`der`, `derReduced`, `derTrOps`,
  `derTrOpsReduced`) plus convenience overloads that default `Ω` and the chisel.
- **`den`** (T-set) implemented, as the design predicted: it is the transpose of the
  `sylve`/`ester` pair. `denLM` returns a genuine rectangular `LinearMap` with a real adjoint
  (verified to 3.6e-15), so every null solver applies; `den`'s `nd` parameter is a batch size,
  default 10, negative meaning "a basis".
- **Equational-law tests** (`test/TestDerivationLaws.jl`): Z-law, T-law, Galois adjunction,
  `denLM` adjoint, and stratification-as-change-of-frame. The verifier `applyDerivation` already
  existed in the package with **no caller** — it was written and never wired up.
- **`:QuickDer`** — the FastDer transcription; dim 38 at 6e-16 on the `n = 19` circulant family
  where it previously returned dimension 0.
- **`:QuickSylver`** ported (double restriction, affine frame → linear basis via offsets from the
  first point); validated against the closed-form oracle `dim scl(C_adj) = 1`
  (`null_patterns` §7.2 eq. 18).
- **`nd` truncation bug**: `SylverLining` rewrote `nd <= 0` to the *valency*, silently returning
  3 vectors of a 6-dimensional space (diagonal 3×3×3) and 3 of 38 at `n = 19`. Oracle for the
  regression test: the diagonal tensor's derivation space is `2n`.
- **Trivial Z-sets are no longer errors.** A Tucker chisel on a generic tensor legitimately has
  `Z = {0}`; the code asserted "Not enough eigenvalues computed; increase `tol`", reporting a
  mathematical fact as a solver failure.
- **Solver registry.** `solve(L, sym)` was a hard-coded `if/elseif` over seven symbols, but the
  extension solver *types* live inside the extension modules, so five of the seven raised
  `UndefVarError: KrylovSolver not defined in Dleto`. Replaced by
  `SOLVER_REGISTRY` / `register_solver!` / `available_solvers()`, with registration inside each
  extension's `__init__()` — **not** at module level, where precompilation discards it.
- **All five null solvers now run.** `Vector{<:Number}` → `AbstractVector{<:Number}` across the
  operator layer (block methods pass views, so CG and Lanczos died in
  `unsafe_embedITensorsSwapped` before reaching any linear algebra); KrylovKit returned `vecs` as
  a vector-of-vectors instead of a matrix; `svdl` returns `(SVD, history)` not `(S, V)`; LOBPCG's
  block size ignored `nv` and needed adaptive shrinking.
- **`der(::Symbol, Ω, ch, Γ)`** was missing — every *partial* setting had a symbol form, so the
  natural call for comparing methods on a fixed operator space was a `MethodError`.
- **`stratify` retag** — the `# retag the indexes` TODO. Each `X_a` carries
  `(a-th index, temporary partner)`, so contracting left the *temporary* index behind and `Σ` came
  back on the temporary frame: a stratified tensor could not be re-chiseled or even verified
  (`AssertionError: Incompatable Indexes`).

### Conditioning is the binding constraint (measured)

Decision 4 predicted that `κ(C)` matters because the composed operator carries `CᵗC`. Session 2
measured the stronger version of the same effect: `sylvester = ester ∘ sylve` **is** `AᵗA` for the
densor map `A`, so its condition number is `κ(A)²`. On the `n = 19` circulant family the spectrum
spans ~25 orders of magnitude, and the consequences are visible per solver:

| solver | dim found (true: 38) | residual | note |
|---|---|---|---|
| `SVDSolver` | 38 | 4.1e-14 | correct; densifies the map |
| `LUSolver` | 32 | 3.1e-14 | `lu` pivots rows only, so it is **not rank revealing** |
| `KrylovSolver` | 9 | 3.5e-15 | accurate but partial: Arnoldi stops on an invariant subspace |
| `LanczosSolver` | 0 | — | `svdl` converges to the **largest** singular values; the null space is at the other end |
| `CGSolver` | 19 | 7.1e-07 | LOBPCG unpreconditioned on `AᵗA`; block must be shrunk to factorize |
| `:QuickDer` | 38 | 5.3e-15 | 3× faster than `SylverLining/SVD`; solves densely on a restriction |

Reading: the iterative solvers are the wrong tool for "give me the whole null space" of a squared
operator. This is direct evidence for the Phase 5 numerics work — the fix is to stop forming `AᵗA`
and take the SVD of `A` (or LSQR on `A`), exactly as Algorithm 2 says.

## Decisions on record

1. **`Chisel` is the full setting `(𝕋, Ω, P)`** (2026-09-02). A wrapper around `P` alone would
   just wrap a Julia array and earn no new name.
   - Follow-up (2026-09-02): the convenience builders keep their names and return a full
     `Chisel`, defaulting `𝕋` to the full tensor space with no symmetry and `Ω` to full square
     matrices on every axis. Since those defaults need `dim(U_a)`, the builders take a **frame**
     (or a tensor to read it from) rather than a bare valence.
   - Follow-up (2026-09-02): builders are **curried** — they return a `ChiselTemplate` that
     completes itself when applied to a tensor or frame. Templates are reusable across tensors;
     `stratify`/`der`/`den` accept either a template or a full `Chisel`.
   - Follow-up (2026-09-02), load-bearing: **chisels are keyed by `Index`, not by axis
     position.** A chisel names the indices it engages and applies to any later tensor carrying
     them, at any valence and any axis order. So `UniversalChisel()` takes nothing and reads the
     valence off the data; `UniversalChisel(i,j,k)` engages exactly those; `AdjointChisel(i,j)`
     and `TuckerChisel(i)` take `Index` terms directly instead of valence-plus-positions.
     `UniversalChisel(3)` no longer needs the `3` — recommendation is to keep the integer form
     as a *valence assertion* rather than deprecate it. This kills the silent-wrong-answer class
     where a positional chisel targets the wrong axes after a reorder or relabel, turns
     "engagement" from a position mask into a set of indices, and makes
     `chisels/ChiselImpls.jl`'s `ChiselFramed` (which already carries `Dict{Index,Integer}`) the
     direct ancestor of `Chisel` — so its dead functions get superseded rather than deleted.
   - Follow-up (2026-09-02), **transpose convention**: the engaged indices are an *ordered
     tuple*, never a set. `AdjointChisel(i,j)` vs `(j,i)` cuts out the same solution set but
     decides **which coordinate is transposed**, and that is what makes the result an algebra
     rather than just a subspace: from `AM = MB` and `XM = MY` one gets `(XA)M = M(YB)`, so the
     two coordinates compose in opposite orders and pointwise product needs one coordinate in the
     opposite ring (equivalently, stored transposed, since `(XA)ᵗ = AᵗXᵗ`). **Convention: the
     transpose lives on the first coordinate**, matching linear algebra and matching Magma's
     `End(U_a)^op × End(U_b)` (`Rihana.pdf` §1.2; `Sylver/src/Invariants.m` transpose bookkeeping
     and `LeftNucleus(t : op := false)`). **Implementation: track it passively** — solve for the
     same matrices, apply the transpose only at the reporting boundary and inside products.
     Note this already exists in the code unnamed: `sylvesterLM` solves via
     `unsafe_embedITensorsSwapped` while `stratify` reports via plain `embedITensors`. Name it,
     document it, pin it with tests — after verifying which embedding is which.
2. **`stratify` returns a `Stratification` type**, not a `NamedTuple` (2026-09-02), so fields can
   be added later without breaking callers. Fields: `Σ`, `Xs`, `δ`, `pattern`, `verdict`,
   `chisel` (provenance, mirroring Magma's `DerivedFrom` breadcrumbs). Field access is unchanged,
   so labs keep working; only the positional destructure at `Densors.jl:58` needs attention.
3. **Densor first** (2026-09-02): fill the missing T-set gap before the numerics work. Confirms
   the existing phase order. Caveat recorded in the plan — the new T-path must extract nullspaces
   via SVD/LSQR from the start rather than copying the `NᵗN` pattern, or Phase 5 fixes the same
   flaw twice.
4. **No single canonical `P`; keep any representative of the row span** (2026-09-02). Because
   `ester`/`sylve` contract with chisel columns on the way in and out, the composed operator
   carries the column Gram matrix `CᵗC`, which iterative solvers hit repeatedly — so `κ(C)` can
   degrade convergence multiplicatively. RREF + integers is the form for reasoning and for how a
   user starts; conditioning-aware selection is future work. This vindicates the existing
   `normalize_chisel` (SVD row basis, `κ = 1`) as the conditioning-optimal policy — it is simply
   never called and not selectable.

### Test-suite state

**As found (session 1):** `ERROR: Package Dleto errored during testing` after 2460 passes.
`Random` sat in `[weakdeps]` while `[targets]` declared only `test = ["Test"]`, so
`TestFastDer3Valent.jl` never executed and the new solver was effectively untested. Both
`TestSylverLining.jl` and `TestDerivations.jl` were commented out of `runtests.jl`, so the
primary numerical solver had no tests running at all — the single largest coverage gap.

**Now (`Pkg.test()`, Julia 1.12.3, end of session 2): green, 7853 passing, 0 failing, 0 broken.**

| testset | tests | time |
|---|---|---|
| realCanonicalForm | 4500 | 1.4s |
| Chisel | 1280 | 0.5s |
| Operators | 190 | 2.2s |
| TransverseOpsIndependant | 900 | 11.1s |
| TransverseOpsSymmetries | 1340 | 15.9s |
| Tensor Synthesis | 32 | **3m50s** — still pathologically slow, unfixed |
| QuickDer / FastDer3Valent | 8 | 5.3s |
| Z-law | 32 | 1.7s |
| `denLM` adjoint | 29 | 1.3s |
| T-law | 23 | 0.0s |
| stratify as change of frame | 17 | 0.9s |
| Galois adjunction | 19 | 0.1s |
| SylverLining (independent / trivial-symmetry / symmetry) | 1650 / 1650 / 208 | 43.7s / 42.8s / 13.7s |

## Benchmarks

`bench/ChiselOperationBench.jl` compares **all three operations** — `der` (Z-set), `den` (T-set),
`stratify` (pattern) — across `(method, chisel)` categories, plus the null-solver axis, writing
`bench/chisel-operation-results.csv` and a six-panel log-log figure (time and accuracy per
operation). Run `julia --project=. bench/ChiselOperationBench.jl [short|long]`.

Two things the benchmark had to get right to mean anything:
- The tensor family must be **structured**. Liu's `K_M_field_tensor` is the multiplication tensor
  of `K[x]/(x^n − c)` (`null_patterns` Ex. 5.2), scrambled by random basis changes, so the
  derivation space is `2n` but invisible in the given basis. A *generic* tensor has only the
  scalar derivations, and its densor is then the **whole** tensor space (§5.4, "scalar derivations
  reveal nothing") — measuring `den` there measures nothing.
- The size grid must cross **`n = 19`**, because `SylverLining` only dispatches to the null
  solver when `globalDim(Ω) = 3n² ≥ 1000`. Below that it uses a dense `eigen` and every
  `/solver` variant is the same code path.

An intended result worth not mistaking for a bug: **`stratify`'s frame conditioning `κ` comes out
identical across all four categories** (~10 digits) while varying trial to trial. `K[x]/(x^n − c)`
is commutative, so its elements share an eigenbasis; the frame putting a random derivation into
real canonical form is the same frame whichever derivation was drawn and whichever chisel or
method found it. The pattern is a property of the *tensor*. So `κ` cross-checks agreement between
methods rather than discriminating them, and the discriminating axis for `stratify` is time.

## Conventions agreed with the user

- **Branches:** every edit goes on its own task branch, named to hint at popular song lyrics.
  Claude may use all git features of this kind freely and pauses only to confirm a push to `main`.
  Session 1: `we-can-work-it-out/opendleto-review-refactor`. Session 2:
  `i-can-see-clearly-now/phase-0-green-baseline`, then
  `signed-sealed-delivered/der-and-derivation-law` (current).
- **Context:** keep this file updated for future chats.
- Diagrams: mermaid fenced in markdown.
- Running the Julia test suite is pre-approved, as are informational bash commands.

## State of the working tree

- `signed-sealed-delivered/der-and-derivation-law` is the live branch and carries everything in
  *Work completed* above. It has **not** been merged to `main`.
- The review docs (`docs/CONTEXT.md`, `docs/Dleto-Design.md`, `docs/review/*`) were authored on
  `we-can-work-it-out/opendleto-review-refactor` and did not descend to the later branches; they
  were checked across explicitly. If a future branch is cut from `main`, bring them along.
- Cruft still to clean: `src/I want you to read OpenDelto and fast-de.md` (a mis-saved prompt),
  root-level `test_deriv*.jl` scratch files, `test/old-tests/`, a Python `.venv/` in the package
  root.
- Note on history: the Phase 0 archive commit accidentally swept in a `git mv`
  (`SylverLininig.jl` → `SylverLining.jl`), because `git commit` takes the whole index. Not
  rewritten, since six uncommitted edits were at risk at the time.

## Next up

Ordered by what unblocks what.

1. **Numerics (Phase 5), now evidence-backed.** Stop forming `AᵗA`; take the SVD of `A` or run
   LSQR on it, as Algorithm 2 specifies. The measured table above is the argument.
2. **The `σ_{e+1}` verdict** (Algorithm 2) — the test that decides "this tensor admits no pattern
   for this chisel". Never implemented, and it is what would give `stratify` an accuracy oracle;
   the benchmark currently reports conditioning and Z-set-dimension preservation instead.
3. **`stratify` should return `δ` and the pattern**, and a `Stratification` type (decision 2).
4. **Phase 1 `Chisel`**: the full `(𝕋, Ω, P)` keyed by `Index`, with curried `ChiselTemplate`
   builders (decision 1 and its follow-ups, including the transpose convention).
5. **I-sets** — still entirely absent.
6. Remaining law families not yet written: the scalar lower bound `dim Der ≥ e = dim null(C)`,
   chisel row-span / torus equivalence, and product closure.
7. `LUSolver` returns an incomplete basis (32 of 38) because `lu` is not rank revealing. It
   reports honest residuals, so a caller filtering on `vals` is safe, but it should probably not
   be offered as a general null solver.
8. Arpack paths are unexercised — it is a weakdep and is not installed in the manifest.

## Open questions

1. Should the refactor keep ITensors as the tensor substrate, or abstract over it?
2. Which applications are next — isomorphism testing, SphereLab-style continuous patterns,
   hypergraphs?
3. Target field support: floats only, or exact arithmetic (rationals / finite fields) too?
4. Liu's thesis was offered and not yet read. It would settle: the sufficiency theorems for the
   restriction sizes `(a',b',c')`, whether valence 3 with a one-row chisel is essential or
   incidental, and the expected complexity — all of which currently rest on the reference
   implementation's own choices rather than on a stated theorem.
