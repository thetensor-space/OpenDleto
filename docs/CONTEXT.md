# OpenDleto — working context for AI chats

Purpose: hand this file to a new chat so it starts with the context of prior sessions.
Keep it updated as work progresses. Last updated 2026-09-03 (session 3: valence-n stratification,
in progress -- see the section of that name).

## Session 3 (2026-09-03, overnight): stratification for valence >= 4, fast

Branch `aint-no-mountain-high-enough/valence-n-stratify`.  Coordination board with every
agent's findings: `bench/reports/night-2026-09-03/BOARD.md`.  Goals set by the user: 4-D
sphere as the test input; two regimes (sparse tensors where contraction is cheap; dense
video-like tensors H x W x T x 3); dims 10..100 under a minute now, 500..1000 within an hour
later; find the right auto-selection of solvers / QuickDer options; verify every derivation by
the Z-law.  Compute budget: Julia only through `bench/jl` (2 procs x 2 threads, 4G heap hint,
5 GB RSS kill) because another job shares the machine.

Work items and where they live:

| item | files | status |
|---|---|---|
| Sphere harness for any valence (`build_sphere(d; valence)`, `reconstruction` for n axes, `bench/HypersphereBaseline.jl`) | `bench/SphereHarness.jl` | done; valence-4 nullity is 4 at every d, recovery 1e-13 |
| QuickDer for any valence: sketch-restrict / solve / lift / verify | design `docs/design/QuickDer-valence-n.md`; code `src/solvers/QuickDerN.jl`; `:QuickDer` -> new, `:QuickDer3`/`:FastDer3Valent` -> old | done; 65x faster than SylverLining at 30^3, 152x at 16^4 |
| `sylvesterLM` over plain arrays, zero-alloc apply, sparse branch | `src/SylverLining/SylverLining.jl` (`backend=:auto/:array/:itensor`) | done; bit-identical, 400 B/apply, 2-5x on sparse |
| Oracle tests at valence 3/4/5 | `test/TestQuickDerN.jl`, `test/TestAutoDer.jl` | done |
| Contraction backend research | `bench/reports/night-2026-09-03/contraction-options.md` | done |
| Auto-selection: `:Auto` = QuickDer when applicable, SylverLining otherwise/on failure; `stratify` default | `src/solvers/AutoDer.jl` | done; thresholds pending the scaling sweeps |
| Scaling sweeps (valence 4 dense/sparse/video; valence 3 to d = 250) and restricted-map solver choice | `bench/QuickDerScaling.jl`, `bench/QuickDerLargeD.jl`, `bench/Frontier.jl`, CSVs in `bench/reports/night-2026-09-03/` | done; Arpack-first on the matrix-free branch |
| `:GramSolver` for the dense restricted solve (Gram + Cholesky-shifted, oversampled subspace iteration + Rayleigh-Ritz on the unsquared matrix) | `src/solvers/NullSolvers.jl` | done; 3 s vs 52 s SVD at d = 100 |
| Apple-GPU backend (Metal.jl weakdep): `backend=:metal` for `sylvesterLM`, `device=:gpu` for QuickDer | `ext/DletoMetalExt.jl`, `ext/DletoMetalSylver.jl`, `ext/DletoMetalQuickDer.jl` | done (2026-09-04 morning) |
| Native-core (Rust/C++) study | `docs/design/Native-Core-Plan.md` | done: do not port now |

Frontier reached (5 CPU threads, Float64, `:Auto`): valence-4 scrambled sphere stratified at
d = 100 in 11 s (d = 50 in 0.4 s, d = 80 in 1.4 s); valence-3 sphere d = 100 in 1.8 s, d = 150 in
11 s, d = 200 in 22 s (8 GB peak; the dense Gram route, `QDN_DENSE_BUDGET_BYTES` = 2.5 GB);
video-shaped 100x100x100x3 derivations in 3.2 s; sparse raw valence-4 sphere d = 100 (nullity 13)
in 6 s.  Every result verified by the Z-law.  Beyond d ~ 200 at valence 3 the matrix-free
restricted branch must converge on its own, and on structured tensors Arpack hits its iteration
cap there -- preconditioning that branch is the next lever toward 500..1000.

Integration branch `beta` (worktree `/Users/algeboy/CODE/OpenDleto-beta`) = `main` + every active
September branch, 13,103 tests passing; `main` itself is untouched until the user says push.

GPU (M4 Max, 40 cores; Metal is Float32-only so GPU runs are exploratory, Float64 CPU certifies):
- `sylvesterLM(...; backend=:metal)` applies the derivation operator 6-17x faster than 5 CPU
  threads on dense tensors on a quiet machine (100^3x3: 26 -> 4.6 ms; 400^3: 889 -> 53 ms;
  300^3x3: 1178 -> 169 ms); end-to-end Float32 Arpack derivations 100^3x3 372 s -> 100 s,
  200x200x100x3 1426 s -> 246 s (2000-11000 applies).  `permutedims!` on the device, not the
  GEMM, is the bottleneck.  On generic tensors QuickDer answers the same question in ~3 s, so the
  GPU operator is the accelerated FALLBACK for tensors that fail QuickDer's genericity check.
- QuickDer `device=:gpu` is a hybrid -- Gram and M*X on the device (11.6x), Cholesky/qr/svd on the
  host because Metal's MPS Cholesky is 4.5x slower than the CPU's -- 2.6x end to end at d = 200.
  QuickDer's work is small by design, so the GPU pays less there.
- Extension pitfall: an extension may only ADD methods; redefining a core method (even a
  zero-argument hook) is method overwriting and silently blocks the extension's precompilation on
  Julia 1.12.  The GPU hooks are therefore `Ref`s set in `__init__`.

Native core (`docs/design/Native-Core-Plan.md`): measured at d = 100 the dense branch spends its
time in `syrk` and Cholesky at the hardware rate, so a Rust/C++ kernel calling the same BLAS gains
nothing; the walls beyond d ~ 130 are iteration counts.  Native kernels only behind a measured 2x
gate, as a `Dleto_jll` weakdep extension so a `git clone` never breaks.

Robustness fixes from the sweeps: `solve_nullspace` confirms an iterative solver's nullity with a
doubled request before trusting it (Krylov methods returned 4..11 of a 13-fold zero eigenvalue);
`GramSolver` oversamples its subspace (a true derivation dropped out at valence 4, d = 50);
`nullspace()` on a tall residual matrix took the full SVD (7 GB `U`).  Eigensolver run-time
outliers (Arpack 127-346 s on cases that otherwise take seconds) remain unexplained.

Bugs fixed along the way: `dense_is_cheap` densified a 160000x840 map because a small side
short-circuited the byte budget (now bytes decide); `SVDSolver` used the thin SVD and so lost the
null space of wide maps; `TransverseOpsSymmetries(fr, localOp)` referenced an undefined variable;
extension solvers (KrylovKit, IterativeSolvers) now register on `using Dleto` because Dleto imports
them.  Arpack is a weak dependency: `bench/jl` stacks the shared env `@dleto-bench` that carries it.

Key design decision (QuickDer-n): restrict by *sketching* each output axis with a random
orthogonal `W_a` (d_a x r_a) rather than by a corner slice.  The corner is a special case
(`W_a = I[:, 1:r_a]`) and is what breaks on structured tensors -- the unscrambled sphere's
corner is all zeros.  The restricted system is `sum_a P[rho,a] (S_a x_a Y_a) = 0` with cross
sketches `S_a = Gamma x_{b != a} W_b` and unknowns `Y_a = M_a W_a`; it has `prod r` equations and
`sum d_a r_a` unknowns, so at d = 1000 valence 3 it is ~6000x cheaper per apply than the full
operator and is handed to `solve_nullspace` (dense SVD when small, LSMR/Arpack when not).
The lift is one least-squares solve per axis with a shared QR; a linear consistency filter
removes restricted solutions that are not restrictions of true derivations, and the Z-law on
random output slices verifies the result.

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

### The null-solver layer, centralized

**All null-solver policy lives in `solve_nullspace` (`src/solvers/NullSolvers.jl`) and is tuned
there once.** Previously `SylverLining` and `den` each carried their own copy — a hand-rolled
`globalDim < 1000` dense gate plus a dense `eigen` branch that bypassed the solver interface
entirely in one, a `__needsSquare(sym)` symbol list plus its own filter in the other. Tuning
either left the other stale. The pieces:

- **`dense_is_cheap(L)`** gates every `Matrix(L)`. The test is on **bytes**, with dimension as a
  shortcut, because dimension alone is the wrong criterion for a rectangular map: the derivation
  operator at `n = 19` is 1083×1083 — just over the 1000 dimension limit, but only 9 MB, so
  densifying is free and buys SVD accuracy — while the densor map at the same `n` is
  260642×6859, i.e. **14 GB**.
- **`AutoSolver`** is the default for `der`, `den` and `stratify`: densify when cheap, otherwise
  delegate to a matrix-free solver, squaring the map *as a composition of linear maps*.
- **Adaptive escalation.** `nd <= 0` means "a basis", and both callers used to turn that into
  "compute the entire spectrum" (`nv = globalDim(Ω)`, `nv = prod(dims)`). That is what made the
  iterative solvers look useless — at `n = 19` Arnoldi was asked for 1083 of 1083 eigenvalues,
  retried five times and returned 9; LOBPCG was handed a block of 1083 and could not factorize
  it. Now: ask for a modest `nv`, count how many values fall below `tol`, and double only while
  *every* returned value is below it — the signal that the null space is not yet bracketed. Cost
  is proportional to the true nullity.
- **`wants_square(::NullSolver)`** trait, so the central layer can form `LᵗL` on behalf of
  whichever eigensolver was named, including one an extension adds later.
- **`ShiftInvertSolver`.** Every black-box iterative eigensolver converges to an *extreme* of the
  spectrum, and most to the largest — which is why `LanczosSolver` returned dimension 0: `svdl`
  was pointed at the wrong end, not failing. The transform `(AᵗA + εI)⁻¹` sends small eigenvalues
  to large ones, so a largest-end method converges straight to the null space, and fast, because
  the gap between 0 and `λ_min` is stretched to the gap between `1/ε` and `1/(λ_min + ε)`.
  **The ε is not optional**: a genuine null vector has `λ` exactly 0 up to rounding, so an
  unshifted `1/λ` overflows precisely on the vectors being sought — the better the solver, the
  worse the blow-up. The inverse is never formed; `AᵗA + εI` is SPD, so it is applied by
  conjugate gradients (a 20-line `cg_solve` ships in core, so the transform needs no optional
  dependency). The whole chain from tensor contractions to null space stays matrix-free.

**`den` no longer densifies.** It cost `O(n⁷)` memory — 14 GB at `n = 19`, 120 GB at `n = 26` —
for an answer that is a handful of vectors and an operator that is a few tensor contractions.
`denLM` was *already* abstract, with a genuine adjoint; the two defaults (`:SVDSolver`, and
`nd = -1` meaning "the whole spectrum") threw that away.

### `LSMRSolver` — the null space by projection, never squaring (the working answer)

Getting `den` to be *correct* matrix-free took one more step. Every other iterative solver here
is an eigensolver, so it needs `AᵗA`, and on the densor map that squaring is what fails:
measured at `n = 10` (map 20000×1000) `σ_max = 1.45e3`, the ten null directions sit at
`σ ≈ 3e-13`, and the smallest **nonzero** singular value is `2.2e-2`. So the null space is
separated by **1.5e-5 relative in σ** — a huge, comfortable gap — and by **2.3e-10 in σ²**,
shedding an order of magnitude of headroom per step in `n`.

**The method is not an eigensolver — it is a projection**, which is why it needs to know nothing
about the spectrum:

    null(A) = { w − A⁺(A w) : w arbitrary }

since `A⁺A` is the orthogonal projector onto the row space. And `A⁺y` is exactly what LSMR
computes (minimum-norm least squares), using only `A` and `Aᵗ`. One LSMR solve per candidate
vector; no shift, no eigenvalue iteration, nothing squared. Three details are load-bearing:

- **Iterative refinement.** One pass leaves an error of about `lsmr_tol · κ_row`: measured
  `σ ≈ 2e-5` where the truth is `3e-13`. The projector is idempotent in exact arithmetic, so
  re-projecting an already-nearly-null vector is standard refinement — `‖Az‖` is now tiny, so the
  same *relative* tolerance buys a far smaller absolute correction. One extra pass took
  `σ` to `2.2e-13`, matching the true null singular values.
- **Rank-revealing QR gives the nullity for free.** Project `k > d` random vectors and they span
  the `d`-dimensional null space, so a column-pivoted QR reports `d`. Unpivoted `qr` would hand
  back `k` columns whatever the rank, and the extra ones would be noise indistinguishable from
  null vectors.
- **Return one column *above* threshold.** `solve_nullspace` brackets by seeing something
  non-null. A solver returning only what it believes is null can never let it bracket, so the
  request doubles to the full dimension — 868,000 map applications at `n = 10` before this was
  caught.

What was tried first and does **not** work: shift-invert subspace iteration, the textbook
approach. Its inner solve `(AᵗA + σI)x = v` is conditioned at `σ_max/√σ`, and picking `σ`
requires knowing the spectral gap in advance. With `σ` small enough to separate the null space
the stacked map had `κ ≈ 1e7`, needing thousands of LSMR iterations; capped at 400 it returned
mid-spectrum directions (`σ ≈ 250–500` out of 1451), i.e. noise. The projection has no such
parameter — the only conditioning that matters is `σ_max/σ_min⁺` on the row space, 6.6e4 here,
which LSMR handles in a few hundred iterations.

Matrix-free `den` on maps far too large to densify, `LSMRSolver` vs LOBPCG-on-`AᵗA`:

| n | map | dense would need | `CGSolver` (squared) | `LSMRSolver` (unsquared) |
|---|---|---|---|---|
| 10 | 20000×1000 | 0.1 GB | 10 of 10, 2.0e-12 | **10 of 10, 2.0e-15**, 41 s |
| 15 | 101250×3375 | 2.5 GB | 15 of 15, 1.2e-11, 190 s | **15 of 15, 4.8e-15, 31 s** |
| 19 | 260642×6859 | 13.3 GB | **11 of 19**, 2.1e-05, 166 s | **19 of 19, 1.1e-14, 131 s** |

Faster *and* four orders more accurate, and it is the only one that gets `n = 19` right at all.
This closes the Algorithm-2 deviation for the densor side: the paper takes the SVD of `N`, and
nothing in this chain — projection, inner solve, or reported value — squares the operator.

**Verified range and the cost wall.** `den` is verified correct to `n = 19` (~2 min). At `n = 26`
it was measured running at **7 map applications/s** — each application is `|Δ| = 52` forward plus
52 adjoint contractions on 17576-element tensors — which extrapolates to roughly **4 hours**, and
was stopped rather than run to completion. So the memory wall is gone but a time wall remains, and
it is in the *map application*, not the solver: cost is `O(k · lsmr_iters)` applications, each
`O(|Δ| · n³) = O(n⁴)`. The lever is the LSMR iteration count, which goes like `√κ_row` and would
need a preconditioner to reduce; `nv0 = 16` also over-asks when the nullity is known to be `n`.
Neither is attempted yet.

### Progress reporting — optional, tagged, off by default

```julia
der(Γ; progress = true)               # every stage
den(Ω, P, Δ; progress = :densify)     # just the dense build
den(Ω, P, Δ; progress = [:solve])     # just the iterative applications
```

Chiseling spends its time applying *our own* maps — `sylve`/`ester` in `sylvesterLM`,
`forward`/`adjoint` in `denLM` — so every unit of work is a call we control and can count. One
wrapper (`progress_wrap`) serves both stages, because `Matrix(L)` applies the map once per
column. Consequently:

- **`:densify` has an exact denominator** and reports a percentage and ETA. This is the case that
  actually makes people wait — densifying the densor map at `n = 12` is 1728 contractions in a
  loop that previously printed nothing at all.
- **`:solve` has none**, so it reports count and rate rather than a fabricated ETA.

Nothing prints until a stage has run for a second (`delay`), so short solves stay silent; an
unknown tag errors eagerly rather than silently reporting nothing.

Implementation note that bit: the `kwargs...` on the *symbol* overloads of `der` are forwarded to
the **method constructor** (`solver=`), so a per-call option like `progress` was being handed to
`SylverLiningMethod(; progress=...)`. Per-call options must be named explicitly in every symbol
overload; `stratify` likewise must not sweep them into `method_kwargs`. Two kinds of keyword share
one splat, and the distinction is invisible at the call site.

### Conditioning is the binding constraint (measured)

Decision 4 predicted that `κ(C)` matters because the composed operator carries `CᵗC`. Session 2
measured the stronger version: `sylvester = ester ∘ sylve` **is** `AᵗA`, so its condition number
is `κ(A)²`. On the `n = 19` circulant family the spectrum spans ~25 orders of magnitude. Measured
per solver, with `nv` set to the whole space — i.e. *before* the escalation fix, so this table
also records what asking for the entire spectrum does to each method:

| solver | dim found (true: 38) | residual | note |
|---|---|---|---|
| `SVDSolver` | 38 | 4.1e-14 | correct; densifies (9 MB here, so allowed) |
| `LUSolver` | 32 | 3.1e-14 | `lu` pivots rows only, so it is **not rank revealing** |
| `KrylovSolver` | 9 | 3.5e-15 | accurate but partial: Arnoldi stops on an invariant subspace |
| `LanczosSolver` | 0 | — | `svdl` converges to the **largest** singular values — wrong end; this is what `ShiftInvertSolver` fixes |
| `CGSolver` | 19 | 7.1e-07 | LOBPCG unpreconditioned on `AᵗA`; block must be shrunk to factorize |
| `:QuickDer` | 38 | 5.3e-15 | 3× faster than `SylverLining/SVD`; solves densely on a restriction |

Reading: squaring the operator, and asking for the whole spectrum, are both avoidable. The
remaining Phase 5 item is to stop forming `AᵗA` *at all* where the solver allows the rectangular
map — take the SVD of `A`, or LSQR on `A`, exactly as Algorithm 2 says.

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
