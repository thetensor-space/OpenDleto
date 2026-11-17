# What's new in OpenDleto `beta`

A preview of the improvements over `main` that matter to someone *using* the package.
Written 2026-09-08 for `beta` at tag `v1.5-beta-2026-09-04`. For every commit see
[CHANGELOG.md](CHANGELOG.md); for what is planned next see
[COMING-FEATURES.md](COMING-FEATURES.md).

The one-line version: **`main` could compute derivations of small dense 3-tensors with one
solver and could not compute a densor at all. `beta` computes derivations of tensors of any
valence up to `d = 1000` on a laptop, computes densors matrix-free, tells you how much to
trust the answer, and does so in Float64, Float32 or Float16, on the CPU or an Apple GPU.**

---

## 1. Any valence, and two orders of magnitude faster: `QuickDer`

`main` had one derivation solver, `SylverLining`, which forms the full derivation operator
and hands it to an eigensolver. It works to about `d = 60` at valence 3.

`beta` adds **`QuickDer`**, a solve-and-lift method for any valence. It restricts the
problem to a random sketch of each axis, solves the small restricted system, lifts the answer
back with one least-squares per axis, and verifies it with the Z-law. On a 30³ tensor it is
65× faster than SylverLining; on 16⁴ it is 152×.

```julia
using Dleto, ITensors
Γ = randSurfaceTensor(...)           # or any ITensor
ders = der(Γ)                        # :Auto -> QuickDer, falls back to SylverLining
ders = der(:QuickDer, Γ; whiten = true, seed = 1)
```

`stratify` now defaults to `:Auto`, so existing scripts get QuickDer for free.

**Frontier reached** (Float64, 5 CPU threads, scrambled sphere with derivation algebra of
dimension = valence):

| valence | `main` | `beta` |
|---|---|---|
| 3 | ~`d = 60` | `d = 1000` in 552 s, 13.7 GB |
| 4 | not attempted | `d = 200` in 181 s |
| video `640×480×300×3` Float32 | — | 35 s, 5.4 GB peak, certified |

## 2. The densor exists: `den`

`main` exported `den` as a stub that asserted `false`. The package named for the densor could
not compute one. `beta` implements the T-set as the transpose of the same Sylvester system
that defines derivations, as a rectangular `LinearMap` with a real adjoint, so every null
solver applies and nothing is ever densified. At `n = 19` the dense route would need 13 GB;
`den` now runs matrix-free in about two minutes and returns all 19 of 19 with residual 1e-14.

```julia
T = den(Ω, P, Δ)          # a batch of tensors in the densor of the operator set Δ
```

The **Galois adjunction** between `der` and `den` (`S ⊆ T(P,Ω)` iff `Ω ⊆ Z(S,P)`) is a
test in the suite, as are the Z-law and T-law.

## 3. Answers come with a verdict

Every derivation route can now return a `DerivationReport`:

```julia
ders, Xs, Σ, report = derTrOpsReduced(QuickDerMethod(), Ω, chisel, Γ; return_diagnostics = true)
report.certified          # true only if the spectrum has a clear gap AND the solver converged
report.undecidable        # values the data's own precision cannot decide
report.residuals          # Z-law residual of each returned derivation
report.store_eltype, report.compute_eltype
```

Behind it, `solve_nullspace` returns a `NullVerdict` whose `status` (`:ok`, `:unconverged`,
`:capped`) is separate from `certified`. **A solver that did not converge can no longer
certify an empty null space**, which `main`'s code path did silently. The Z-law check itself
is a library function, `Dleto.der_residual(Γ, D, chisel)`, bounded in memory and free of
type promotion.

## 4. One floating-point policy: Float16, Float32, Float64

`main` computed in whatever type it was handed with fixed literal tolerances. `beta` has
`Precision.jl`: every solver takes its default tolerance from the element type, Float16
input is computed in Float32, and two independent floors apply:

- `precision_floor(T)`: below this, arithmetic cannot separate a value from zero.
- `data_floor(T_stored)`: below this, the *input's rounding* cannot; a Float16 tensor
  therefore never certifies a nullity it cannot support, even when computed in Float32.

Constants were tuned by experiment (`bench/reports/precision-*`), and the design is written
up in `docs/design/Precision-Policy.md`.

## 5. Solvers you can choose, and one that chooses for you

`main` named seven null solvers, five of which raised `UndefVarError`. `beta` has a registry:

```julia
available_solvers()       # [:AutoSolver, :SVDSolver, :LUSolver, :ShiftInvertSolver, :GramSolver,
                          #  :KrylovSolver, :LanczosSolver, :CGSolver, :LSMRSolver, (:ArpackSolver)]
der(:QuickDer, Γ; solver = :ArpackSolver)
```

`AutoSolver` densifies when the matrix is cheap by *bytes* and otherwise picks a matrix-free
method, asking for a modest number of eigenvalues and doubling only while the null space is
not yet bracketed. `LSMRSolver` finds a null space by projection without ever squaring the
operator. `GramSolver` handles QuickDer's dense branch 17× faster than an SVD. Iterative
solves are **seeded** and reproducible.

## 6. Apple GPU

With `using Metal`, `sylvesterLM(...; backend = :metal)` applies the derivation operator
6–17× faster than five CPU threads, and `der(:QuickDer, Γ; device = :gpu)` runs the Gram
and sketch stages on the device. Metal is Float32-only; Float64 CPU runs remain the
certifying path.

## 7. Correct where `main` was wrong

- `nd <= 0` means "a basis". `main` silently truncated to the valency (3 of 38 at `n = 19`).
- `FastDer3Valent` solved the wrong chisel (sign of the third slot) with a wrong system
  matrix hidden behind a heuristic; it is now a transcription of the reference and serves as
  the oracle for QuickDer at valence 3.
- A trivial derivation algebra is an answer, not "increase `tol`".
- `stratify`'s output frame was on a temporary index; a stratified tensor could not be
  re-chiseled. Fixed.
- The test suite runs: 14,279 passes in 51 testsets (2026-09-08, Julia 1.12), against a
  suite on `main` that errored after 2,460.

## 8. For contributors

- `bench/jl` runs Julia inside a memory and thread budget on the shared machine.
- `docs/CONTEXT.md` is the running design record; `docs/design/` holds the decision notes.
- `beta` advances only in tagged, reviewed, tested steps (`v<version>-beta-<date>`).

---

### What has not changed

The public names `stratify`, `der`, `Chisel` builders, `TransverseOps`, the operator kinds,
and the ITensor substrate are all as on `main`. Existing notebooks in `labs/` run unchanged.
