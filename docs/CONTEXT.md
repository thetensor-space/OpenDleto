# OpenDleto — working context for AI chats

Purpose: hand this file to a new chat so it starts with the context of prior sessions.
Keep it updated as work progresses. Last updated 2026-09-04 (session 4: the whitened
restriction, then the memory wall; session 3: valence-n stratification -- see the
sections of those names).

## Session 4 (2026-09-04): the whitened restriction, QuickDer-W

Branch `feature/under-pressure/2026-09-04` (work done on a worktree branch off it).
Design note: `docs/design/QuickDer-valence-n.md` section 2a.  Numbers:
`bench/reports/2026-09-04/whitened/`.

The wall session 3 stopped at was the matrix-free restricted branch: above d ~ 200 at
valence 3 the dense/Gram route does not fit and the matrix-free branch "converged
slowly", with ARPACK hitting its iteration cap on structured tensors.  It turns out the
branch was not slow, it was **not converging at all** -- forced matrix-free on the
scrambled sphere, ARPACK exhausts its cap and returns nullity 0 at every d from 30 up.
The cause is sketch-induced and removable exactly.

The structure.  The axis-`a` column block of the restricted system is `I_{r_a} ⊗ M_a`
up to the row permutation `_qdn_row_perm` bookkeeps, where `M_a` is the transposed
mode-`a` unfolding of the cross sketch.  So the DIAGONAL blocks of the restricted Gram
are `c_a·(I ⊗ M_aᵗ M_a)` with `c_a = Σ_ρ P[ρ,a]²` -- all of the sketch's conditioning
lives in `d_a x d_a` Grams -- while the off-diagonal blocks, which are what encodes the
derivation condition, survive any per-axis change of variables.  Verified to 5e-16 at
valence 3 and 4 for universal, centroid and adjoint chisels.

The fix (`whiten = true`, now the DEFAULT, and `:Auto` inherits it): thin QR
`M_a = Q_a R_a`, solve for `Ỹ_a = R_a Y_a`.  Every diagonal Gram block becomes exactly
`c_a·I`, the spectrum lands in `[0, Σ_a c_a]`, the null space is untouched, and the cost
is `n` QRs of `(∏_{b≠a} r_b) x d_a` -- 0.01 s of an 18 s solve at d = 200.

Measured (forced matrix-free so both settings meet the same solver at the same size;
ARPACK, Float64, 5 threads; every whitened result Z-law verified):

| case | plain applies / verdict | whitened applies / verdict |
|---|---|---|
| sphere v3 d = 30  | 53632 / ARPACK cap, nullity 0  | 23732 / nullity 3, 2.8e-13 |
| sphere v3 d = 40  | 55196 / ARPACK cap, nullity 0  | 30796 / nullity 3, 1.2e-12 |
| sphere v3 d = 100 | 66900 / ARPACK cap, nullity 0  | 34636 / nullity 3, 7.8e-11 |
| sphere v3 d = 150 | 177142 / ARPACK cap, nullity 0 | 39544 / nullity 3, 1.7e-12 |
| sphere v3 d = 200 | 65182 / ARPACK cap, nullity 0  | 38262 / nullity 3, 3.3e-10 |
| randn 150^3 | 14618 / nullity 2, ok | 10156 / nullity 2, ok (1.44x) |
| randn 250^3 | 22974 / nullity 2, ok | 15712 / nullity 2, ok (1.46x) |
| sphere v4 d = 60  | 61994 / ARPACK cap, nullity 0 | 12808 / nullity 4, 1.1e-13 (4.8x) |
| sphere v4 d = 80  | 64584 / ARPACK cap, nullity 0 | 19718 / nullity 4, 2.4e-13 (3.3x) |
| sphere v4 d = 100 | 65858 / ARPACK cap, nullity 0 | 13188 / nullity 4, 3.7e-13 (5.0x) |
| video 200x200x100x3 Float32 | 56980 / ARPACK cap, nullity 0 | 11182 / nullity 3, 1.7e-5 (5.1x) |
| degenerate 40^3, mode-1 rank 38 | 59640 / nullity **26** of 82 | 4024 / nullity **82**, 8.4e-16 (14.8x) |

**FRONTIER MOVED.**  Whitened, matrix-free, ARPACK, Float64, scrambled sphere:
valence 3 d = 200 in 18.4 s (2.1 GB peak, no restricted matrix at all -- session 3's
dense Gram route needed 22 s, an 8 GB peak and a 2.2 GB matrix), **d = 300 in 105 s**
(3.8 GB) and **d = 500 in 136 s** (11.4 GB, residual 3.2e-13).  Valence 4 reaches
d = 100 in 7.2 s (11.7 GB).  The user's goal for this size was "500..1000 within an
hour"; d = 500 is 2.3 minutes.  What now stands between here and d = 1000 is MEMORY in
the harness and the sketch pass -- `build_sphere` holds three copies of the tensor and
`_qdn_pair_tensor` `permutedims` the full tensor once per lift axis, so d = 1000 needs
~22 GB against a 12 GB kill line -- and no longer convergence in the solver.

The generic-tensor row is the control and it agrees with the analysis: a random tensor's
mode unfoldings are already well conditioned (`cond(M_a) ≈ 3` against 20..150 on the
sphere), so there is nothing for the whitening to remove and it is worth 1.4x.

LOBPCG (`:CGSolver`) does NOT win on the whitened operator, despite the spectrum now
being in `[0, n]`: unpreconditioned it returns the WRONG nullity, silently (nullity 2 of
3 at d = 30 unwhitened, 0 of 3 at d = 100 and 200 both ways) and takes 2-4x more applies
than ARPACK when it does answer.  ARPACK-first stays the right default on this branch.
Whitening helps it too (168034 -> 53124 applies at d = 30, and it becomes correct there),
so the gain is a property of the operator, not of ARPACK.

Degenerate modes, and a correctness fix.  When a mode unfolding is rank deficient (smooth
video, separable tensors: `cond(M_a) ~ 3e16` measured on a smooth 20x20x10x3 box) the QR
has no inverse, so the whitening takes an SVD and truncates to the range at the standard
`min(size)·eps` cut.  The dropped directions are the trivial derivations `X_a ×_a Γ = 0`;
they are checked against Γ itself and written down exactly rather than left in the solve
as a numerically-zero eigenvalue cluster.  That also FIXES an undercount: the unwhitened
branch only ever sees them as `Y_a` with columns in `ker(M_a)`, and its rank-deficient
lift returns a minimum-norm `Z_a` with no kernel component, so on a 12^3 tensor with a
mode-1 rank of 10 the true derivation space is 26-dimensional and the old path reported
an arbitrary, `W`-dependent 16 of it.  Degenerate modes are meant to be removed upstream
by `nondeg`; this is the safety net.

Out of budget on this machine, established by `bench/WhitenedRestriction.jl estimate`
before anything was allocated (`build_sphere` holds three copies of the tensor):
valence-4 spheres at d = 150 (11.3 GB) and d = 200 (35.8 GB) in Float64, d = 200 in
Float32 (17.9 GB), and valence-3 d = 1000 (22.4 GB) -- against a 12 GB per-process kill
line.  Valence 3 reaches d = 500 and valence 4 d = 100 in Float64.

New knob for the next session: `Dleto.QDN_APPLY_COUNT[] = 0` before a call counts the
matrix-free applies the restricted solve took (forward + adjoint), so iteration count is
readable directly instead of inferred from wall time.

### Session 4, part 2 (afternoon): the memory wall, and three honesty fixes

Branch `we-gotta-hold-on-to-what-weve-got/qdn-memory-frontier` off
`feature/under-pressure/2026-09-04`.  Numbers: `bench/reports/2026-09-04/whitened/`
(README sections "Memory: where the frontier's bytes actually were" and
"Honesty"), profiler `bench/MemoryProfile.jl`.

**The memory wall was the harness, not the solver, and not where reading the
code said.**  Measured at d = 500 valence 3 (one copy 0.93 GB): the build peaks
at 10.3 GB of the run's 12.1, and the whole of QuickDer adds 1 GB on top.  Six
copies live at the worst moment for a pipeline that needs one, and the biggest
single term is `nondeg` -- a full `d^{n-1} x d` SVD per axis, which forms the
unfolding AND its left factor: 25 GB of churn, 4.7 GB of peak.
`_qdn_pair_tensor`'s per-axis transpose, the other named suspect, is 80 MB at
d = 500; it is fatal only at d = 1000, where the same transpose is 8 GB.

Fixed: `_qdn_ttm` never permutes its input (axis 1 and axis `N` are reshapes,
so single GEMMs; a middle axis runs `back` GEMMs on contiguous slices), plus
`_qdn_ttm!` (two buffers) and `_qdn_ttm_square!` (ONE buffer -- a block of
slices into 64 MB and copy back, which every square mode product can use).
`build_sphere` takes a lean path above `SPHERE_LEAN_BYTES` (256 MB per copy):
plain arrays, one tensor-sized buffer, `nondeg`'s axis bases by tall-skinny QR
(`_tsqr_axis_basis`: `R` is `d x d`, one 64 MB block, full precision, a third
of `gesdd`'s flops), `itensor` rather than `ITensor` at the boundary, and the
original sphere dropped unless `keep_S` asks.  The lean input is the same family
up to a per-axis orthogonal change of basis, not the same array; `-lean` marks
those CSV rows.

Also memory: the trivial derivations of a rank-deficient mode are published
FACTORED (`QDN_TRIVIAL_FACTORED`, one `(; axis, K)` per axis, meaning
`X_axis = K v e_q'`) and written out as operator tuples only up to
`QDN_TRIVIAL_MAX_BYTES` (256 MB), warning and naming the field when that
truncates.  At d = 500 with one degenerate mode the write-out was 3 GB of zeros
to describe a 2 MB matrix.  The degenerate 40^3 case still reports its full 82
at 8.4e-16.

**FRONTIER MOVED AGAIN.**  Whitened, matrix-free, ARPACK, Float64, scrambled
sphere: **valence 3 d = 1000 in 552 s at a 13.70 GB peak**, nullity 3, Z-law
residual 7.19e-13 (r = [57,56,56], 178752 x 168000 restricted system, 56388
applies) -- the goal for this size was "500..1000 within an hour" -- and
**valence 4 d = 150 in 31 s at 7.29 GB and d = 200 in 181 s at 17.35 GB**
(nullity 4, residuals 1.63e-13 and 1.33e-11), against the morning's d = 500 and
d = 100 ceilings -- the morning's own estimate table put valence 4 d = 200 at
35.8 GB and "over budget, not attempted".  At valence 4 the solve is now 9.6 s
of d = 200's 181 s, so the remaining cost is the build and the Z-law check, not
the solver.  d = 200 reproduces the morning's
row to the digit (38262 applies, 3.249e-10), which is the regression check on
every change here: the mode-product rewrite, the in-place kernels, the
preallocated apply and the rewritten Z-law check are numerically identical, not
merely close.  On the user's own video shape, 640x480x300x3 Float32 was killed
at 13.1 GB this morning and now peaks at 4.06 GB in 43.9 s.

Honesty, three items:

* A non-converging iterative solve reported an empty null space rather than a
  failure.  The solvers that know now say so (`ArpackSolver` from `nconv`,
  `CGSolver` from LOBPCG's flag, `KrylovSolver` from `info.converged`) and
  `NullVerdict` carries it as `status` -- `:ok` / `:unconverged` / `:capped`,
  kept separate from `certified`, which is a statement about the spectrum
  alone.  `_qdn_solve_and_lift` declines a non-`:ok` solve that returned
  nothing, so `:Auto` falls back to SylverLining.  Block Lanczos on the
  whitened restricted map at d = 300 and 500 is the measured case: it converges
  nothing, stalls at 1.1e-9 relative, and its spectrum reads `nullity 0,
  certified`.
* The d = 300 apply anomaly (76810 applies against d = 500's 51116) is NOT the
  `nv` escalation.  The restricted system there has a 13-dimensional numerical
  near-null space against 3 true derivations; the escalation chases it
  correctly and the lift filter cuts 13 to 3.  A gentler escalation step is
  measurably worse (four solves, 99170 applies, against three and 76810), so
  doubling stays; the lever is the restriction size or the null threshold.
* The lift consistency filter now cuts on a GAP in the residual spectrum
  (floor `sqrt(eps(T))`, which is where a lift residual bottoms out, ceiling
  `32*sqrt(eps)`) instead of at a hard cutoff AT that floor.  But the Float32
  undercount it was written for is not there: at d = 48 Float32 the filter
  passes all 13 of its 13 directions under both rules and
  `_fastder_restrict_to_ops` then returns 2.  **And ARPACK's start vector comes
  from a seed saved inside the library across calls**, so repeated calls in one
  process on identical input give different restricted null spaces -- the same
  setting three times gives 2, 3, 3 at d = 48 Float32.  Measure any Float32
  matrix-free result one data point per PROCESS, or with a deterministic
  solver.

Two more memory items, both found by the same profiler and both large:
`_qdn_restricted_map` now allocates its scratch ONCE rather than ~13 MB per
apply-pair (a 100k-apply solve was throwing away ~700 GB, and RSS tracked the
GC's heap target instead of the live set), and the BENCHMARK's own Z-law check
`der_residual` was the biggest single item on video shapes -- it went through
`applyDerivation`, so ITensor contraction materialised a full `d^n` intermediate
per axis AND promoted a Float32 tensor to Float64 because the chisel is
Float64.  That is why peak RSS grew ~15 GB per GB of tensor there and why
Float32 and Float64 peaked the SAME.  It now accumulates
`Σ_a P[ρ,a]·(Γ ×_a D_a)` in 256 MB blocks along the longest axis, in the
tensor's own type, and reproduces the old value exactly (degenerate 40^3:
8.40e-16 both ways).  `_qdn_ttm!` gained `mul!`'s `α`/`β` to make that
possible.

Next levers, in the order the measurements suggest: (1)
`_fastder_restrict_to_ops` is the Float32 cliff -- at d = 48 Float32 it takes
13 lifted derivations to 2; (2) the restriction sizes at d = 300 (6.8%
overdetermined) are what make its restricted near-null space 13-dimensional and
so what makes its apply count non-monotone; (3) `bench/jl`'s RSS watchdog needs
`JL_HEAP` set well below the limit on these runs -- RSS ran ~1.5x the heap
target above the live set on every large run.

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

Integration branch `beta` = `main` + every active September branch, 13,103 tests passing; `main`
itself is untouched until the user says push.  `beta` lives in its own worktree and the primary
worktree carries the active daily branch.  `beta` is an approval gate, not a working branch --
see "How work reaches beta" below.

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

## Direction set 2026-09-04 (evening): the movie regime runs in Float16

The user's target is video: 640 x 480 x F x 3 at 30 fps, F up to 1800 (one minute), and by
decision it will run in **Float16** -- 3.3 GB for the minute instead of 6.6 GB. Measured on
that shape today (Float32, whitened matrix-free QuickDer, 5 CPU threads, `labs/MovieRuntime.ipynb`):
1 s / 3 s / 10 s of video take 21.6 / 28.3 / 44.7 s, all nullity 3/3 certified; the affine fit
projects the minute at ~171 s of which ~20 s is the eigensolve (set by 640 x 480, flat in F)
and ~150 s is the tensor-touching stages (linear in F). Peak RSS is ~14x the tensor
(3.4 / 6.7 / 16.9 GB), NOT explained by input copies -- Float64 used no more than Float32 at
the same F -- so the minute would need ~92 GB today.

Measured again after the lean kernels landed (same harness, 5 threads, the machine shared with two
other agents so wall times are noisy; `labs/MovieRuntime.ipynb` has the plots):

| 640 x 480 x F x 3 | F = 30 (1 s) | F = 90 (3 s) | F = 300 (10 s) |
|---|---|---|---|
| Float32 wall / peak RSS, before | 21.6 s / 3.4 GB | 28.3 s / 6.7 GB | 44.7 s / 16.9 GB |
| Float32 wall / peak RSS, after | 22.7 s / **1.7 GB** | 18.7 s / **2.4 GB** | 34.8 s / **5.4 GB** |
| Float16 input, after | 23.8 s / 1.7 GB | 24.2 s / 2.9 GB | 61.3 s / 5.4 GB |
| nullity (oracle 3), verdict | 3, certified (all) | 3, certified (all) | 3, certified (all) |

Fit on the current tree: `t = 18 s + 0.053 s/frame`, `RSS = 1.2 GB + 0.014 GB/frame` (maxrss under
`JL_HEAP=6G`, so it carries GC headroom; the live set is ~2.8 GB per GB of tensor). The minute
projects to **~110-170 s and ~19-26 GB in Float32 on the CPU** -- it runs on this machine now.
Float16 input costs the same CPU memory as Float32 because the precision policy computes in
Float32; the Float16 storage saving is realised only where storage stays Float16, i.e. on the GPU
path being built.

Order of work, decided with the user:
1. **Memory first.** Profile the video shape per stage; find the precision-independent ~14x
   footprint; target < 15 GB for the minute. Nothing else pays off until this lands.
2. **Then GPU, designed together with (1).** Move the tensor stages (sketch, lift, verify --
   the ~150 s) to Metal on the restructured pipeline: the tensor lives once, contracted
   axis-by-axis without a full permuted copy (last night: `permutedims!` on the device, not GEMM,
   was the bottleneck). The eigensolve stays on the host: 20 s, tiny, and Float32.
3. **Float16 contract**, pending measurement (`docs/design/Float16-Metal.md` when it lands):
   expected shape is Float16 storage and half operands on the GPU with fp32 accumulation and
   output, Float32 for the restricted eigensolve, verdict floored by `data_floor(Float16)` so a
   half-precision result is never certified beyond what the data carries.

Not now: native core (gated behind a measured 2x), eigensolve on the GPU (wrong regime for
video), the d = 1000 long-axis frontier as a goal in itself.

## How work reaches beta, from 2026-09-04

OpenDleto is now developed alongside a **private downstream project** that consumes it. That
project keeps its own clone of this repository pinned to `beta`, so the worktrees here are not
shared with anything -- branches can be switched freely. Do not reintroduce an arrangement where
another project builds directly out of a working tree here; a checkout would change its build
underneath it.

- **`beta` is an approval gate.** Work happens on a *daily* branch,
  `feature/<song-lyric>/<YYYY-MM-DD>` (the first was `feature/under-pressure/2026-09-04`). At the
  end of the day the most promising, fully-tested improvements from it are submitted for approval;
  only then do they merge to `beta`, and the downstream project fetches the update on its own
  schedule. `main` is protected and untouched.
- **Beta advances only in reviewed, tested steps**, tagged as
  `v<version>-beta-<YYYY-MM-DD>`. The first is `v1.1-beta-2026-09-04`: 13,101 tests over 29
  testsets, zero failures, on Julia 1.12.3.
- **A second assistant maintains the downstream project.** Cross-project coordination -- advisory
  git locks, a message queue, and reservations for big-memory runs -- lives in a coordination
  directory *outside* both repositories, not in version control. Its rules in brief: claim the
  advisory lock around pull/push/merge/checkout and release it immediately after; never
  force-release the other side's; report memory while active; read the queue at the start of a
  session and write to it after any merge to beta. Never modify the downstream repository --
  bugs and feature requests come back through the queue instead.
- **Memory on the shared machine.** The soft target for ordinary work is small (~2 GB), which no
  benchmark fits inside. Runs over 6 GB are announced and negotiated: a joint ceiling of 56 GB of
  the machine's 64, and correctness tests always outrank benchmarks
  (`blocker` > `test` > `bench` > `explore`). Being outranked means checkpoint, exit and re-queue
  at the head -- not abandon. `bench/jl` still defaults to a 5 GB per-slot RSS kill line; raise it
  deliberately, per run, with a reservation to match.

Housekeeping note: `bench/jl` used to leave a `juliaup self update` process behind on every
invocation -- the overnight runs of 2026-09-03 accumulated 470 of them holding 3.8 GB. Fixed by
resolving the real Julia binary once and bypassing the juliaup shim.

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
