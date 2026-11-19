# Derivation sweep at 20³–100³, 2026-11-18

Measured on a 64 GB Apple M4 Max Mac Studio, with Julia 1.12.3,
Python 3.14.0, NumPy 2.5.5, and SciPy 1.18.1.

This supersedes the timing conclusions from the 4³–10³
[small-map sweep](FULL-SWEEP-2026-11-18.md). It tests order-three tensors with
side lengths **20, 40, 60, 80 and 100**, in both Python and Julia, using the
same little-endian Float64 input at each size. The result of a method is
recorded only when it returns vectors. A partial Ritz solve is reported as a
lower bound, and a solver's untrusted or unconverged verdict is shown as such.

## Input and questions

Each axis has full unfolding rank at every measured size. The first two axes
carry grades 0 and 1, and the third carries grades 0, 1 and 2. Random
coefficients fill the allowed blocks where the first two grades sum to the
third. Thus 37.5% of the entries are nonzero, and the grade matrices supply a
known, non-scalar derivation for the universal chisel. The tensor has enough
structure to make stratification meaningful without reducing to a tiny core.
The adjoint chisel engages the first two axes and tests QuickSylver on the
same input. `nondeg` runs before every solve and reports the resulting shape.

The sweep records three work items separately:

1. **Numerical dimension verdict:** materialize the reduced derivation map and
   run `svdvals`. The baseline 1 GiB conservative working-set cap permits
   this at 20³. A targeted 16 GiB run measures 40³. Larger sizes receive
   `skipped_memory`, with the projected bytes recorded.
2. **Complete derivations:** compute basis vectors, verify their defining
   equation, and report the solver's certification or partial status. Julia
   runs Auto, SylverLining, QuickDer, and the chisel-specific QuickDer3 or
   QuickSylver. `QuickDerFullGram` changes only the Gram stage from its
   default mixed Float32/Float64 arithmetic to full Float64; it is a
   diagnostic policy variant, not another package API. Python runs its public
   `derivations` solver, which uses dense SVD at 20³ and partial matrix-free
   ARPACK at larger sizes.
3. **Stratification:** run the package's transform on a returned non-scalar
   derivation. Python also computes a support mask and forbidden energy;
   Julia's transform returns canonical frames and transformed coefficients.
   Python labels a result based on a partial basis as such.

The fast Julia methods and Python SVD, derivation, and stratification stages
each make one target-size warmup call before their reported hot-path time.
Slow Julia SylverLining has
a small-size compilation warmup but only one target-size call; its time is
marked accordingly in the [raw CSV](large-sweep-2026-11-18.csv). BLAS,
OpenMP, and Julia use one thread. Package loading, process startup, input
generation, and compilation warmups are outside those stage times. `nondeg`
is a single target-size call after a small-size Julia warmup. The runner
records process timeouts in its manifest and the consolidated CSV.

## Results

The [raw CSV](large-sweep-2026-11-18.csv) gives every method, stage, status,
dimension, residual, and working-set estimate. A numerical dimension is not
reported for a partial Python solve. At 20³, both languages' complete dense
`svdvals` and basis solves agree: universal dimension 3, adjoint dimension 1.
At 40³, both languages' complete universal `svdvals` runs again find 3.
For larger inputs, Julia's full-precision QuickDer reports a certified cut;
the known grade derivation and scalar derivations provide an independent lower
bound on the universal dimension.

Times below are seconds. `partial` means Python returned a validated
non-scalar vector but did **not** prove the total dimension. `uncertified`
means Julia returned vectors and checked their residuals, but its null-space
verdict did not certify the count. A dash in the `svdvals` column means the
memory preflight skipped dense materialization. `timeout` means the whole
Julia size process reached its 120-second wall limit before SylverLining
returned; startup and earlier methods consumed part of that limit.

### Universal chisel: non-scalar stratification exists

| Size | Python `svdvals` | Python basis | Julia `svdvals` | Julia QuickDer, full Gram | Julia QuickDer3 | Julia SylverLining |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20³ | 0.321, dim 3 | 0.568, complete | 0.257, dim 3 | 0.030, certified (default SVD route) | 0.027 | 5.82, uncertified |
| 40³ | 25.2, dim 3 | 3.60, partial | 24.4, dim 3 | 0.025, certified | 0.246 | 14.7, uncertified |
| 60³ | — | 11.8, partial | — | 0.119, certified | 2.18 | 38.5, uncertified |
| 80³ | — | 40.6, partial | — | 0.361, certified | 7.68 | 102, invalid count (1) |
| 100³ | — | 73.0, partial | — | 1.01, certified | 21.9 | 218, uncertified count 2 |

At 40³ and every larger universal size, **default QuickDer and Auto return
zero vectors and an uncertified verdict**, despite two guaranteed scalar
derivations and the constructed non-scalar grade derivation. The controlled
`QuickDerFullGram` run changes only the Gram stage's arithmetic to Float64,
and returns three certified vectors with maximum defining-equation residual
`3.7e-15`. This is a correctness boundary before it is a speed comparison.
QuickDer3 independently returns three vectors with small residuals at every
size, but that reference method does not issue a certification report.
The [AutoDer route](../../src/solvers/AutoDer.jl) falls back only when QuickDer
throws `QuickDerDeclined`; on this family QuickDer returns an empty,
uncertified result instead. That explains why Auto propagates zero vectors.
The controlled precision toggle identifies the Gram stage as the immediate
cause of this failure in these runs; it does not by itself establish which
internal Gram step loses the null cluster.

The Python solver's large-size route returns one validated non-scalar vector
at 40³–100³. Its dimension lower bound is 3, and ARPACK reports only four
converged Ritz pairs of twelve requested. Its `partial` time is therefore a
time to a useful **candidate**, not a time for a complete basis. It can still
stratify along that candidate: at 100³, the Python support-producing stage
takes 0.0386 s and finds 375,000 allowed positions; Julia's canonical
transformation from its certified basis takes 0.0119 s. The two stratification
APIs do different work, so their timings should not be used as a direct
language speed ratio.
Python removes the scalar gauge from its returned basis, so it emits one
non-scalar vector where Julia emits that vector plus two scalar directions.
The reported total dimension at 20³ agrees despite this output convention.

Even at 20³, Julia's full-map `svdvals` takes 0.257 s versus 0.030 s for
QuickDer to return and certify the vectors. Python's `svdvals` takes 0.321 s
versus 0.568 s for its complete vector solve; it saves 0.247 s there, but
does not supply the vector needed for stratification. At 40³, values-only
SVD takes about 25 seconds in either language while Julia's certified
full-precision QuickDer basis takes 0.025 seconds. At 100³ the solver
dominates end-to-end latency: Julia's 0.048 s reduction and 0.012 s transform
are small next to its 1.01 s certified solve; Python's 0.108 s reduction and
0.039 s support computation are small next to its 73.0 s partial solve.

### Adjoint chisel: QuickSylver

The effective operator space removes the disengaged third axis before the
singular-value check. The numerical dimension is 1 at 20³ in both languages.
The larger Julia full-precision QuickDer runs certify the same dimension;
QuickSylver returns one vector with residual below `1e-15` at every size.
There is no non-scalar adjoint derivation on this input, so stratification is
correctly skipped.

| Size | Python basis | Julia QuickSylver | Julia QuickDer, full Gram | Julia SylverLining |
| --- | ---: | ---: | ---: | ---: |
| 20³ | 0.257, complete | 0.0071 | 0.022, certified (default SVD route) | 3.87, certified |
| 40³ | 1.75, partial | 0.0150 | 0.103, certified (SVD route) | 5.02, certified |
| 60³ | 5.28, partial | 0.0396 | 0.0541, certified | 7.89, certified |
| 80³ | 16.3, partial | 0.0935 | 0.154, certified | 23.7, certified |
| 100³ | 41.0, partial | 0.200 | 0.404, certified | 52.0, certified |

Default QuickDer and Auto also return zero uncertified vectors on this adjoint
case at 60³, 80³ and 100³. Full Float64 Gram restores the scalar derivation
and its certified cut. QuickSylver is the fastest successful Julia method
here; at 100³ its 0.200 s solve is about 260 times shorter than
SylverLining's 52.0 s solve. This ratio compares successful Julia basis
computations; QuickSylver's report has no certification flag.

### What scales

`nondeg` retains the full `n³` shape at every size, confirming that this sweep
measures the requested tensor dimensions after the reduction step. At 100³,
the reduction itself takes 0.108 s in Python and 0.048 s in Julia. The
universal map has 1,000,000 rows and 30,000 operator coordinates there: a
single Float64 dense copy would be 240 GB, and the conservative factorization
working-set estimate is 989 GB. The corresponding estimates are 10.6, 78.4,
and 326 GB at 40³, 60³, and 80³. The 40³ full-map SVD used a separate
16 GiB preflight cap and completed in about 25 seconds. The 60³–100³
estimates exceed this host's 64 GB memory. The restricted QuickDer solve
provides both vectors and a certified dimension at those sizes when its Gram
stage uses full Float64.

At 80³, a standalone SylverLining process returned one vector after 102 s;
its Krylov solver reported unconverged Ritz pairs and the count is below the
guaranteed scalar dimension of 2. At 100³ it returned only two scalar
directions after 218 s, missing the constructed non-scalar direction and
again reporting an unconverged verdict. The first combined processes timed
out before either result. These are times to incomplete answers, not
complete-derivation timings.

These inputs are deliberately structured and 37.5% nonzero so they have a
meaningful non-scalar derivation. The timings characterize this family, not
generic dense tensors or video data. Julia SylverLining's 20³–60³ universal
times are single target-size calls after a small-size compilation warmup;
their verdicts were unconverged. The fast-method and Python times are warmed
target-size calls. All results use one thread and the same stored Float64
input; native linear-algebra libraries and solver policies still differ.

## Reproduce

From the repository root, with `python/.venv` and the Julia project installed:

```sh
python/.venv/bin/python bench/python_vs_julia/run_large_sweep.py /tmp/opendleto-large-universal --chisel universal --python-warm-target --python-timeout 240
python/.venv/bin/python bench/python_vs_julia/run_large_sweep.py /tmp/opendleto-large-adjoint --chisel adjoint --python-warm-target --python-timeout 120
python/.venv/bin/python bench/python_vs_julia/run_large_sweep.py /tmp/opendleto-large-dense40 --sizes 40 --dimension-only --python-warm-target --dense-memory-mib 16384 --python-timeout 240 --julia-timeout 240
python/.venv/bin/python bench/python_vs_julia/run_large_sweep.py /tmp/opendleto-large-sylver --sizes 80 100 --languages julia --julia-methods SylverLining --julia-timeout 240
python/.venv/bin/python bench/python_vs_julia/summarize_large_sweep.py /tmp/opendleto-large-universal /tmp/opendleto-large-adjoint /tmp/opendleto-large-sylver /tmp/opendleto-large-dense40 --output /tmp/opendleto-large.csv
```

The runner generates inputs, starts each size in its own process, caps wall
time, writes per-size logs and `manifest.jsonl`, and writes a `COMPLETE` marker.
The `--julia-timeout`, `--python-timeout` and `--dense-memory-mib` limits are
configurable. These limits are a benchmark policy; a skipped dense run is not
a claim that no other machine could execute it.
