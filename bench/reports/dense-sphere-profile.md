# Dense ITensor / sphere solver profile

`bench/DenseSphereProfile.jl` separates three commonly conflated properties:

1. physical `ITensor` storage,
2. the fraction of numerical entries that are nonzero, and
3. the spectral difficulty of the resulting derivation operator.

Run the modest default with:

```sh
julia -t 2 --project=. bench/DenseSphereProfile.jl --csv=bench/dense-sphere-profile.csv
```

The default uses `d = 10`, both `SymmetricOp` and `UniversalOp`, a warmed
median of nine map applications, and every available member of
`AutoSolver`, `SVDSolver`, `ArpackSolver`, and `KrylovSolver`.  Use
`--dims=10,20,30`, `--ops=symmetric`, `--solvers=AutoSolver,ArpackSolver`,
and `--reps=15` to enlarge it.  `--smoke` only checks the storage claim.

## What the harness constructs

| variant | construction | storage | numerical density |
| --- | --- | --- | --- |
| `support` | original sampled sphere | `NDTensors.Dense` | about `O(1/d)` |
| `scrambled` | `randomize_tensor(support; type=:orthogonal)` | `NDTensors.Dense` | 1.0 |
| `random` | generic `randn` tensor | `NDTensors.Dense` | 1.0 |

This settles the original question for ordinary ITensors: randomization is an
eager contraction, not a lazy wiring to the support and basis matrices.
`randomize_tensor` returns `Γ * mats` in `src/util/Random.jl`; ordinary
`ITensor(frames...)` already has `NDTensors.Dense` storage.  The generic
zero-filled support tensor therefore does **not** take a sparse contraction
path.  A physically sparse comparison requires a QN/block-sparse ITensor,
not merely a dense tensor containing zeros.

The smoke mode and `test/TestTensorDensity.jl` assert this directly: the
support and its randomized result are both `NDTensors.Dense`; the support has
some zero entries, while every randomized entry is nonzero.

## Initial measurements

Apple Silicon, Julia 1.12, two Julia threads, Float64, one seed.  Times are
warm and can move with BLAS/thread contention; map-call counts are the more
portable signal.  They include `solve_nullspace`'s 10-step norm estimate.

| d | operator | variant | numerical density | median map ms | Auto choice / calls | nullity | gap |
| ---: | --- | --- | ---: | ---: | --- | ---: | ---: |
| 10 | symmetric (165) | support | .055 | .073 | dense SVD / 185 | 3 | 2.09e11 |
| 10 | symmetric (165) | scrambled | 1.0 | .071 | dense SVD / 185 | 3 | 3.44e11 |
| 10 | symmetric (165) | random | 1.0 | .068 | dense SVD / 185 | 2 | 4.35e12 |
| 10 | universal (300) | support | .055 | .069 | dense SVD / 320 | 13 | 7.11e5 |
| 10 | universal (300) | scrambled | 1.0 | .067 | dense SVD / 320 | 13 | 7.07e5 |
| 10 | universal (300) | random | 1.0 | .065 | dense SVD / 320 | 2 | 2.95e12 |
| 20 | symmetric (630) | support | .026 | .294 | Arpack / 360 | 3 | 1.64e11 |
| 20 | symmetric (630) | scrambled | 1.0 | .226 | Arpack / 359 | 3 | 2.34e11 |
| 20 | symmetric (630) | random | 1.0 | .226 | Arpack / 363 | 2 | 7.63e12 |

For the `d=20` explicit solvers, SVD took 650 map applications and
0.315--0.459 s; Arpack used 359--363 and took 0.144--0.652 s in this short,
lightly contended run.  Krylov's count varied more (367--902) because it is
also sensitive to the spectrum.  The important result is not the small wall
time difference: numerical zeros in a `Dense` store did not make its map
application categorically cheaper.  The randomized tensor was actually a
little faster in this sample.

The universal sphere and generic random tensor make the stronger point.  Both
are physically and numerically dense at `d=10`, but the sphere has 13
derivations and asked Arpack for 2,750 map applications whereas the random
tensor has the two scalar derivations and needed 355.  Density cannot predict
that convergence behavior; the spectrum/nullity does.

## AutoSolver decision

No density routing change is justified.  At `solve_nullspace` the input is a
`LinearMaps.FunctionMap`; it exposes map dimensions and element type but not a
stable public reference to the captured ITensor.  Inspecting closure fields
would be brittle.  More importantly, solver choice is driven by both spectrum
and map cost:

- dense SVD forms one column per map dimension and then factorizes;
- Arpack/Krylov use a spectrum-dependent number of matrix-free applications;
- tensor storage only affects the cost of each application, and here all three
  inputs use the same `NDTensors.Dense` execution path.

The existing square-map policy (`N <= 400` densifies; larger maps use Arpack
when registered) remains the supported policy.  If a real QN/block-sparse
video workload later shows a reproducible crossover, add explicit operator
metadata at `sylvesterLM`/`denLM` construction and pass it to the solver
layer; do not infer it from `LinearMap` closure internals.  The profile script
already records the storage type, stored fraction, numerical density, warmed
map cost, map calls, null verdict, and gap needed to calibrate such a change.
