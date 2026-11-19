# Complete derivation CPU sweep, 2026-11-18

For method behavior and speed at 20³–100³, use the
[large-tensor sweep](LARGE-SWEEP-2026-11-18.md). This page is a small-map
correctness and overhead check.

The previous `compare_solve` scripts timed only `svdvals` on a generic `4³`
derivation map. That operation can count a numerical null space, but it does
not return its vectors or compute a stratification. This sweep measures three
separate tasks after nondegenerate reduction:

1. **Dimension verdict:** materialize the effective derivation map and run
   `svdvals`, using a relative cutoff of `1e-8`.
2. **Complete derivations:** run the public Python `derivations` solver or each
   applicable Julia `derTrOpsReduced` method with no vector cap. Record the
   number of returned vectors and the largest defining-equation residual.
3. **Stratification:** transform the reduced tensor using a returned
   derivation. Python's `stratify` additionally computes a support mask and
   forbidden energy; Julia's `stratify(Γ, δ)` computes canonical frames and
   the transformed tensor. Their times therefore describe their respective
   APIs, not identical work.

The inputs are shared little-endian Float64 arrays in Fortran order. There are
generic dense tensors, diagonal tensors with extra derivations, and tensors
embedded in larger zero-padded spaces. `nondeg` runs first in each language;
the raw-map singular-value result is retained only to show the cost and
nullity before reduction. For the adjoint chisel, the disengaged third axis is
removed from the operator space before the dimension check. Otherwise its
arbitrary operators create a spurious 16-dimensional summand at `4³` that
Julia's derivation solver already removes.

## Measured results

Numbers below are **milliseconds**, median of three calls after one warmup,
with one BLAS and Julia thread. Package import, Julia compilation, and process
startup are excluded. Each singular-value time includes map construction and
materialization. The derivation times include the solvers' own validation;
the two Julia reference methods without diagnostics had their residuals checked
after the timed solve. Source data: [small CSV](full-sweep-small-2026-11-18.csv)
and [extended CSV](full-sweep-extended-2026-11-18.csv). The [small checks](full-sweep-small-checks-2026-11-18.json)
and [extended checks](full-sweep-extended-checks-2026-11-18.json) compare each
solver's count with dense SVD and both languages' SVD counts.

| Universal case | Core | Reduced dim | Python `svdvals` | Python basis | Julia `svdvals` | Julia SylverLining | Julia QuickDer |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generic `4³` | `4³` | 2 | 0.953 | 1.233 | 0.178 | 0.290 | 0.311 |
| diagonal `4³` | `4³` | 8 | 0.891 | 1.349 | 0.118 | 0.231 | 0.247 |
| embedded `5³` | `3³` | 2 | 0.518 | 0.779 | 0.085 | 0.168 | 0.179 |
| generic `8³` | `8³` | 2 | 7.630 | 9.496 | 2.976 | 3.354 | 1.622 |
| diagonal `8³` | `8³` | 16 | 6.715 | 8.383 | 2.033 | 3.941 | 3.413 |
| embedded `10³` | `6³` | 2 | 2.477 | 3.337 | 0.758 | 1.139 | 1.250 |

Reduction on the embedded `10³` input takes 0.123 ms in Python and 0.233 ms
in Julia. Before reduction, both see nullity 122; after reduction, both see
nullity 2. The raw `svdvals` route takes 13.131 ms in Python and 6.179 ms in
Julia, versus 2.600 ms and 0.991 ms for reduction plus reduced `svdvals`.
Thus removing radicals changes the mathematical question and saves more time
here than choosing a different reduced solver.

At `8³`, the complete Python basis costs about 1.2–1.3 times its dimension-only
`svdvals` route. The singular values alone save time, but do not replace the
basis when stratification is needed. Julia QuickDer cuts the generic `8³`
solve from 3.354 to 1.622 ms versus SylverLining, and the diagonal `8³` solve
from 3.941 to 3.413 ms. On the smaller cores, the general dense solver is
competitive. The Python port currently has one public derivation solver, so
there is no Python QuickDer counterpart to rank.

For the diagonal `8³` case, Python's support-producing stratification takes
0.198 ms after the 8.383 ms basis solve. Julia's canonical transformation
takes 0.028 ms after the 3.413 ms QuickDer solve. For generic tensors, Python
removes scalar derivations and correctly skips stratification because it has
no non-scalar vector. Julia currently permits stratification along a scalar
derivation, so that work is timed but is not evidence of a useful support
pattern. End-to-end latency should include `nondeg` plus basis plus the
applicable stratification stage.

All six universal reduced dimensions agree between languages and with the
successful complete solver methods. The adjoint `4³` cases also agree at
dimension 1 for generic data and 4 for diagonal data. The largest measured
residual of any returned vector is `5.8e-15`. Julia's Auto, SylverLining and
QuickDer diagnostic reports marked their cuts certified on these inputs.
These are numerical verdicts at the chosen tolerance, not formal proofs for
arbitrary tensors.

Two reference methods have a real domain limitation: QuickSylver failed on
the diagonal `4³` adjoint case, and QuickDer3 failed on the diagonal `8³`
universal case, both with `LinearEqualsAffine assumes the linear system matrix
has full column rank.` The runner records the errors and continues. Auto,
SylverLining, QuickDer, Python derivations, and dense SVD returned the expected
dimensions on those cases. This points to rank-deficient lift handling in the
reference solvers rather than a disagreement over the tensor's nullity.

This is a CPU, Float64, small-map sweep. Existing Metal and MPS measurements
time map applications; Python MPS has no adjoint, so a complete GPU derivation
comparison is not available from the current implementation. Larger tensors
may switch to iterative solvers, where a partial lower bound must not be
reported as a dimension proof.

## Reproduce

From the repository root, with the project-local Python environment and Julia
project dependencies installed:

```sh
bash bench/python_vs_julia/run_full_sweep.sh /tmp/opendleto-full-sweep 3 small
bash bench/python_vs_julia/run_full_sweep.sh /tmp/opendleto-full-sweep-extended 3 extended
```

The runner writes `python.jsonl`, `julia.log`, `full-sweep.csv`, and
`checks.json` to each output folder. It uses Julia's existing compiled
modules, which also avoids writes to the user package cache in a restricted
workspace. The historical `compare_solve` files remain a narrow
singular-values microbenchmark.
