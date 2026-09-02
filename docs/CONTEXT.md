# OpenDleto — working context for AI chats

Purpose: hand this file to a new chat so it starts with the context of prior sessions.
Keep it updated as work progresses. Last updated 2026-09-02.

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

Headlines:
- **T-sets are absent.** `Densors.jl` contains only `stratify`; `den` is an `@assert false` stub.
  The package named for the densor cannot compute a densor.
- **I-sets are absent.**
- **Z-sets exist** and are more general than Magma's, but chisels are untyped bare matrices and
  no solver applies `normalize_chisel`.
- **Numerics diverge from the paper**: Algorithm 2 says take the SVD of `N`; the code
  eigen-decomposes `N'N`, squaring the condition number. The `sigma_(e+1)` test that decides
  "this tensor admits no pattern for this chisel" is never performed.
- **`stratify` throws away `delta`**, i.e. the sparsity pattern itself, and applies no canonical
  eigenvalue ordering.
- Dead code: `ChiselImpls.jl:57,61` (undefined `Fch`, `enggaged`), `Derivations.jl:88`,
  `export der` with no definition.
- No sanity-check harness, no invariant caching, no `Homotopism` / `TensorOverCentroid` analogue.
- **`FastDer3Valent` solves the wrong chisel.** The reference solves `X·R + S·Y − T·Z = 0`, so the
  third slot enters negated; the port sets `R,S,T = c₁Γ, c₂Γ, c₃Γ`, turning chisel `[c₁,c₂,c₃]`
  into `[c₁,c₂,−c₃]`. With the default `[1,1,1]` it computes the `[1,1,−1]` derivations.
- **`FastDer3Valent` dropped the reference's solution verification**, which exists precisely
  because solve-and-lift is only generically correct at a given `(a',b',c')`.

## Sibling repo: `../fast-der-solver`

Chris Liu's repo (grad-show poster 2025) holding **two** algorithms — `quick-der` (triple
restriction) and `quicksylver` (double restriction, affine frame). Only the first was ported, as
`FastDer3Valent`; `quicksylver` corresponds to the `[COMING SOON] QuickSylver` stub in
`src/solvers/NullSolvers.jl`. Both restrict to a small corner, solve densely, then lift by small
affine solves. Author's own benchmark `quicksylver-vs-dleto-results.csv`: at ~6 minutes wall
clock, solve-and-lift reached `n = 315` vs the current OpenDleto path at `n ≈ 68` and a dense
baseline at `n ≈ 33`. Both are valence-3 only, which is acceptable.

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

### Verified test-suite state (`Pkg.test()`, Julia 1.12.3, 2026-09-02)

`ERROR: Package Dleto errored during testing`. 2460 tests pass, then the suite dies:

- pass: Operators 190, TransverseOpsIndependant 900, TransverseOpsSymmetires 1340,
  Tensor Synthesis 30 (the last takes 4m12s for 30 tests — pathologically slow)
- **error** at `test/runtests.jl:22` → `test/TestFastDer3Valent.jl:1`:
  `ArgumentError: Package Random not found in current path`
- Root cause: `Random` sits in `Project.toml` `[weakdeps]` (line 24) while `[targets]`
  declares only `test = ["Test"]`. Fix: add `Random` to `[extras]` and to the `test` target.
  Separately, `Random`, `IJulia` and `ProgressMeter` are weakdeps with **no** matching entry in
  `[extensions]`, which is a Project.toml defect in its own right.
- Consequence: `TestFastDer3Valent.jl` never executes, so the new solver is effectively untested.
- **`TestSylverLining.jl` (150 lines) is commented out of `runtests.jl` (line 25)**, as is
  `TestDerivations.jl`. The primary numerical solver therefore has no tests running at all.
  This is the single largest coverage gap.

## Conventions agreed with the user

- **Branches:** every edit goes on its own task branch, named to hint at popular song lyrics.
  Current: `we-can-work-it-out/opendleto-review-refactor`.
- **Context:** keep this file updated for future chats.
- Diagrams: mermaid fenced in markdown.

## State of the working tree

- Branch `feature/fast-der-3valent-integration` carries the `FastDer3Valent` solver
  (`src/solvers/FastDer3Valent.jl`, `test/TestFastDer3Valent.jl`,
  `labs/FastDerSphereComparison.ipynb`), wired in via `method=:FastDer3Valent`. **The test
  suite does not pass on it** — see below.
- Cruft to clean: `src/I want you to read OpenDelto and fast-de.md` (a mis-saved prompt),
  root-level `test_deriv*.jl` scratch files, `test/old-tests/`, a Python `.venv/` in the
  package root, and the misspelled `src/SylverLining/SylverLininig.jl`.

## Open questions

1. What are the main aims in full? (user offered a walkthrough)
2. Should the refactor keep ITensors as the tensor substrate, or abstract over it?
3. Which applications are next — isomorphism testing, SphereLab-style continuous patterns,
   hypergraphs?
4. Target field support: floats only, or exact arithmetic (rationals / finite fields) too?
