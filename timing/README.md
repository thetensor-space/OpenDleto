# timing — stratification runtime vs. dimension

Runtime of tensor stratification on the SphereLab problem, as a function of the
axis dimension `d` of a cubical `d × d × d` tensor with a hidden sphere, with
the **derivation solve** and the **stratification itself** timed apart.

| file | what it is |
|---|---|
| `StratifyTiming.jl` | the driver — builds the input, sweeps `d`, writes the CSV |
| `stratify-timing.csv` | the measurements, one row per `(d, eltype, ops, method)` |
| `StratifyTiming.ipynb` | Plotly visualization of the CSV |

Reproduce from the repo root:

```bash
bench/jl timing/StratifyTiming.jl 200 60      # maxd 200, 60 s budget
```

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

- `d` = 10, 15, 20, … stepping by 5.
- **Precision**: `Float32` solved to `tol = 1e-8`, `Float64` to `tol = 1e-16`.
  Both sit just under the respective machine epsilon, which is deliberate:
  `tol` is the singular-value cutoff separating the null space from the rest,
  so this asks each solver to call a mode null only when it is null to the last
  bit. Both settings do find the sphere — read `nullity` and `lsq_err` to
  confirm it per row.
- **Chisel**: always the universal chisel. What varies is the operator space —
  `universal` (`UniversalOp()`, unrestricted) and `symmetric` (`SymmetricOp()`,
  operators restricted to symmetric matrices).
- **Methods**: `SylverLining` with the auto and SVD null solvers, `Auto`,
  `QuickDer`, `QuickDer3`, and `SymmetricGram`. Not every method applies in
  every operator space — `QuickDer`/`QuickDer3` lift through least-squares
  solves over all matrices and so are universal-only, `SymmetricGram` is
  symmetric-only by construction — so the method set differs between the two
  facets by design, not by omission.

## Drop-out

A `(method, ops, eltype)` configuration whose total time exceeds the budget
(60 s) at some `d` is not run at any larger `d`. Its curve simply ends, and
where it ends is the result. A second, looser guard skips a configuration whose
cubic extrapolation to the next `d` projects past 10× the budget, which bounds
how far any single run can overrun.

Every configuration is run twice at `d = 8` before any timing, and any
measurement under a second is taken twice and the second kept, so the reported
times are compiled code rather than JIT.

## Reading the accuracy columns

`lsq_err` is the relative reconstruction error of the recovered sphere inside
the permutation-and-scale ambiguity a stratification is defined up to — 0 is
perfect, ~1 is nothing recovered.

The **symmetric** operator space is the well-posed one here: an orthogonal
conjugate of a diagonal derivation is symmetric, so it is exactly the space
that fits the scramble. There the derivation space has nullity 3 (two scalar
derivations plus the sphere's) and recovery reaches ~1e-6 in Float32, ~1e-14 in
Float64.

The **universal** space finds a much larger derivation space on this input
(nullity around 13–14 at `d = 10`), so a random combination of its basis does
not single out the sphere and `lsq_err` sits near 1. That is a property of the
input, not a solver failure — the universal rows are here for the runtime
comparison the unrestricted chisel gives, and their `lsq_err` should be read as
"this chisel does not pin the sphere down", not as a bug.
