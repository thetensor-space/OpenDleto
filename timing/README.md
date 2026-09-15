# timing — stratification runtime vs. dimension

Runtime of tensor stratification on the SphereLab problem, as a function of the
axis dimension `d` of a cubical `d × d × d` tensor with a hidden sphere, with
the **derivation solve** and the **stratification itself** timed apart.

| file | what it is |
|---|---|
| `StratifyTiming.jl` | the driver — builds the input, sweeps `d`, writes the CSV |
| `stratify-timing.csv` | the measurements, one row per `(d, eltype, ops, method, solver)` |
| `stratify-timing-literal-tol.csv` | a superseded run, kept as evidence — see *What went wrong the first time* |
| `StratifyTiming.ipynb` | Julia (IJulia) notebook: the figures. Line plots only — no tables |

Reproduce from the repo root:

```bash
bench/jl timing/StratifyTiming.jl 150 60      # maxd 150, 60 s budget
```

Watch it with `timing/sweep.log`, which the driver writes as it goes. The
**warm-up runs before the first row is written** and can take many minutes: 54
configurations, several of which iterate to convergence on a badly conditioned
system and cost tens of seconds even at `d = 6`. The log names each one as it
completes, so a quiet sweep can be told apart from a hung one. `sweep.log` is
gitignored.

The notebook runs on the `julia-1.12` IJulia kernel and activates this project
itself, so open it from inside `timing/`.

`bench/jl` is the project's Julia wrapper (thread, heap and RSS budget for a
shared machine). Never invoke bare `julia` here.

## The input

`labs/SphereLab.ipynb` section 3's construction, at scale. Axis values
`u[i] = x_i² − r²/3` with `x_i² = r² i/(d−1)`, and support where the three
`u`'s sum to zero — so the sphere's equation holds *exactly* on the lattice
and the tensor is nondegenerate. The tensor is then scrambled by a random
orthogonal change of basis on every axis and passed through `nondeg`, which is
what hides the sphere. `bench/SphereHarness.jl` builds and scores it; this
folder adds only the timing split, the precision grid and the drop-out rule.

## The two phases

`stratify(Ω, ch, Γ)` is two phases that scale quite differently, and the CSV
splits them because only the first depends on the solver:

- **`der_seconds`** — `derTrOpsReduced`: solve for the derivation space. The
  null-space problem, and the phase the method choice actually changes.
- **`strat_seconds`** — embed the chosen derivation, put it in real canonical
  form, act on `Γ`. An eigendecomposition per axis plus three contractions:
  method-independent, and the floor the derivation time is measured against.

The driver runs `stratify`'s own body, unrolled, so the two phases are the real
ones rather than a re-implementation.

## The grid

- `d` = 10, 15, 20, … stepping by 5, up to 150 — the same ceiling
  `bench/StratifyMatrix.jl` uses for full stratification.
- **Precision**: `Float32` and `Float64`.
- **Chisel**: always the universal chisel. What varies is the operator space —
  `universal` (`UniversalOp()`, unrestricted) and `symmetric` (`SymmetricOp()`).
- **Solvers and methods**, 27 configurations per (precision, operator space):
  `SylverLining` against *every registered null solver* — `AutoSolver`,
  `SVDSolver`, `LUSolver`, `GramSolver`, `ArpackSolver`, `KrylovSolver`,
  `LanczosSolver`, `CGSolver`, `LSMRSolver`, `ShiftInvertSolver` — plus `Auto`,
  `QuickDer` against three of them, `QuickDer3` and `SymmetricGram`. Not every
  one applies in every operator space (`QuickDer`/`QuickDer3` lift over all
  matrices and so are universal-only; `SymmetricGram` is symmetric-only), so the
  configuration set differs between facets by design.

## Tolerance is not accuracy

Each run solves at `solve_tol = 1e-6` and is **scored** against an accuracy target
of `1e-8` (Float32) and `1e-16` (Float64). These are different quantities and the
CSV carries both:

- `solve_tol` is the singular-value cutoff handed to the solver. It is not
  honoured literally — `qd_tolerance` floors it at
  `max(tol, sqrt(data_floor(T)), precision_floor(T))`
  ([src/solvers/Precision.jl](../src/solvers/Precision.jl)), so anything under
  ~1e-7 in Float32 is decorative, and on the null-solver path the tolerance is
  relative to the operator norm and gets *squared* when the map is squared.
- `residual` is what the run attained: the Z-law residual `Dleto.der_residual`
  of the derivation it stratified along — the defining equation, measured
  relative to the size of the data. `meets_target` is `residual <= target`.

## The solver axis is the benchmark

Which null solver runs is the single biggest lever here, and most of them live in
package extensions: `:ArpackSolver` behind Arpack (a **weak** dependency),
`:KrylovSolver` behind KrylovKit, `:LanczosSolver`/`:CGSolver`/`:LSMRSolver`
behind IterativeSolvers.

A script that does not import them does not merely lose those rows.
`:AutoSolver` picks `first(matrix_free_solvers(L))` from the solvers actually in
`SOLVER_REGISTRY`, so with Arpack missing it silently falls through to Krylov or
CG — which this repo has measured at **7–20× slower** (at `d = 150`: Arpack
10.5 s, Krylov 76 s, CG 210 s). The driver therefore **refuses to start** unless
`:ArpackSolver` is registered, rather than quietly producing a slower table.

## Drop-out

A `(method, ops, eltype)` configuration whose total time exceeds the budget
(60 s) at some `d` is not run at any larger `d`. Its curve simply ends, and
where it ends is the result. A second, looser guard skips a configuration whose
cubic extrapolation to the next `d` projects past 10× the budget, which bounds
how far any single run can overrun.

Every configuration is run twice at `d = 8` before any timing, and any
measurement under a second is taken twice and the second kept, so the reported
times are compiled code rather than JIT.

## Reading the accuracy figures

`residual` is the Z-law residual described above; `lsq_err` is the relative
reconstruction error of the recovered sphere inside the permutation-and-scale
ambiguity a stratification is defined up to — 0 perfect, ~1 nothing recovered.

The **symmetric** operator space is the well-posed one: an orthogonal conjugate of
a diagonal derivation is symmetric, so it is exactly the space that fits the
scramble. There the derivation space has nullity 3 — the two scalar derivations
plus the sphere's — and a run that sits on 3 found the sphere and nothing
spurious.

The **universal** space finds a larger derivation space on this input (nullity
13–14 at `d = 10`), so a random combination of its basis does not single out the
sphere and `lsq_err` sits near 1. That is a property of the input under an
unrestricted chisel, not a solver failure. The universal panels measure what an
unrestricted chisel costs.

Neither precision reaches its nominal target on this input: the best Float64
residuals are around 3e-16 against a 1e-16 target, the best Float32 around 4e-8
against 1e-8. That is the conditioning of the scrambled sphere, not any solver.

## What went wrong the first time

`stratify-timing-literal-tol.csv` is the first sweep, kept because its failure is
worth recording. It made two mistakes at once:

1. It never imported Arpack, so every `:AutoSolver` and `QuickDer` row silently
   measured the slow fall-through described above.
2. It passed `tol = 1e-8`/`1e-16` straight to the solvers as if tolerance were
   accuracy. Past `d ≈ 50` every Float64 row came back with **nullity 0** — no
   derivation found at all.

Together those made stratification look as though it died at `d ≈ 30`, against a
repo that has measured a full stratification at `d = 150` in 8.8 s and a
derivation solve at `d = 1000` in 552 s.

## Figures

[`StratifyTiming.ipynb`](StratifyTiming.ipynb) is line plots only, deliberately —
there is no table view anywhere in it.

1. **Every configuration, one panel each.** Small multiples, ordered by how far
   each configuration got. Colour is the precision, dash is the operator space —
   two colours, which clears every contrast and colour-vision gate on its own.
2. **The leaders.** The six configurations reaching the largest `d` in each
   facet, overlaid, each line labelled at its last measured point.
3. **Where the time goes.** Solid the derivation solve, dotted the stratification.
4. **The same split as a fraction** of the run.
5. **Accuracy attained**, with the `1e-8` and `1e-16` targets drawn as rules.
6. **Nullity found**, with a rule at 3.
