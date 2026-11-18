# OpenDleto Python and Julia: CPU and Apple GPU growth

This page records the original OpenBLAS-era sweep. The newer comparison with
**Apple Accelerate enabled in Julia**, an extended GPU range, and unrestricted
GPU-run host threading is in [the Apple Accelerate report](APPLE-ACCELERATE-2026-11-18.md).
The [video-sized comparison](VIDEO-2026-11-18.md) times a 640 × 480 × 300
tensor with Julia Metal and Python MPS in seconds.

We measured the universal-chisel derivation map on the **same generic order-three tensors and operator vectors** in four implementations: Julia's dense array CPU kernel, Python's NumPy CPU kernel, Julia's Metal GPU kernel, and Python's PyTorch MPS GPU kernel. All four used `float32` on an Apple M4 Max on 2026-11-18. The input values were generated once, saved in a portable binary format, and cast to `float32` in each process.

![Line plot of forward-map time versus tensor side length from 8 to 128. Julia CPU and Python NumPy CPU are fastest at small sizes and rise sharply. Both GPU lines begin near one millisecond; Python MPS reaches about 1.5 milliseconds at 128, while Julia Metal reaches about 6.8 milliseconds. The vertical axis is logarithmic.](growth-2026-11-18.png)

The plot shows **time for one forward-map application**. Lower is faster. The vertical axis uses a logarithmic scale so the small and large cases remain visible together. Each point is the median of 11 calls after five warmups, with one CPU and BLAS thread. NumPy uses Apple Accelerate, capped with `VECLIB_MAXIMUM_THREADS=1`; Julia uses OpenBLAS, capped with `BLAS.set_num_threads(1)`. Package loading, map construction, and the one-time tensor upload are excluded. Both GPU timings end with an output on the CPU; host-facing operator coordinates are supplied to both.

| Tensor | Julia CPU | Python NumPy CPU | Julia Metal GPU | Python MPS GPU |
| --- | ---: | ---: | ---: | ---: |
| 8³ | 0.0018 | 0.0197 | 1.1545 | 1.0430 |
| 16³ | 0.0131 | 0.0318 | 1.2054 | 0.6415 |
| 24³ | 0.0483 | 0.0477 | 1.0785 | 0.6163 |
| 32³ | 0.1233 | 0.0700 | 1.0199 | 0.5991 |
| 48³ | 0.4928 | 0.2264 | 1.0490 | 0.6467 |
| 64³ | 1.5503 | 1.6155 | 1.0913 | 0.6765 |
| 96³ | 6.8992 | 4.1975 | 6.0097 | 1.0810 |
| 128³ | 20.1945 | 15.1653 | 6.7810 | 1.4079 |

All times are milliseconds. At `128³`, Julia Metal was about 3.0 times faster than Julia CPU, and Python MPS was about 10.8 times faster than Python NumPy CPU. The Python MPS forward call was about 4.8 times faster than the Julia Metal call in this particular implementation and run. Around `64³`, both GPU paths began to beat their own CPU counterparts. Julia Metal's forward time rose from about 1 ms at `64³` to about 6–7 ms at `96³`; a second sweep reproduced that jump. These are measurements of the present kernels, including their host/device interface costs, rather than a general language or GPU ranking.

## Adjoint and numerical agreement

Julia Metal also supports the adjoint map. Its measured adjoint times were 1.031 ms at `64³`, 1.312 ms at `96³`, and 1.763 ms at `128³`; Julia CPU took 1.459, 6.344, and 19.244 ms respectively. Python NumPy CPU took 0.715, 2.174, and 6.630 ms. The current Python MPS backend does not implement the GPU adjoint, so no four-way adjoint or full GPU solver comparison is claimed.

The saved outputs passed [the parity check](check_parity.py): maximum relative difference across forward and available adjoint outputs was `2.34e-6`. Forward outputs from the two CPU implementations matched exactly in `float32`; Julia Metal and Python MPS forward outputs agreed with Julia CPU to at most `6.8e-8` relative error. Float32 rounding explains the larger adjoint differences.

A separate complete dense SVD on the same generic `4³` tensor took 0.109 ms in Julia and 0.922 ms in Python; both found nullity 2 at relative tolerance `1e-8`. That is a CPU-only small-map comparison. It does not compare the packages' different high-level solver policies.

## What this benchmark says about performance

The CPU columns compare different native matrix libraries as well as different OpenDleto code: this NumPy wheel links to Apple Accelerate, while Julia used OpenBLAS for this historical run. The current `beta-python` branch defaults to Accelerate on supported Macs. A separate one-thread, column-major `128² × 128` by `128 × 128` matrix multiply took about 0.585 ms with Accelerate through NumPy and 4.889 ms with OpenBLAS through Julia. Those isolated numbers do not by themselves explain every map timing, but they show why the CPU gap cannot be attributed to Python versus Julia as languages. Both maps also spend time moving between tensor layouts and accumulating axis actions.

The Julia Metal forward kernel still has an optimization opportunity. An isolated `96³` stage check measured about 2.16 ms for its first matrix multiply, 1.90 ms for the middle multiply, and 0.35 ms for the tensor permutation. At `128³`, the corresponding times were 1.87, 1.70, and 0.51 ms. The sharp rise above `64³` is mainly in the matrix multiplies in this check; the precise driver or kernel-selection cause remains unproven. The Python GPU path also has avoidable overhead: it prepares operator matrices on the CPU and uploads them on every call.

For the **normal operator** `D*D`, the solver's repeated hot path, the available comparison at `128³` is: Julia CPU 38.86 ms, Python NumPy CPU 20.65 ms, and Julia Metal 7.71 ms. Python MPS has no adjoint or normal-operator implementation yet. Thus the forward-map plot cannot rank complete derivation solves. Julia Metal already gives a substantial speedup for the full map that its solver actually uses.

## Reproduce

The measured data are in [results-2026-11-18.csv](results-2026-11-18.csv). The scripts generate fresh shared inputs, save numerical outputs for parity checks, and regenerate the plot. Run from the repository root with Julia 1.12, the shared `@dleto-bench` Julia environment containing Metal, the project-local Python environment containing `opendleto[torch,dev]`, and a plotting Python environment containing Matplotlib:

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python/.venv/bin/python bench/python_vs_julia/compare.py --output /tmp/opendleto-perf --sizes 4 8 16 24 32 48 64 96 128 --repeats 11
DLETO_BLAS=openblas JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --compiled-modules=existing --project=. bench/python_vs_julia/compare.jl /tmp/opendleto-perf float32 8 16 24 32 48 64 96 128 | tee /tmp/opendleto-perf/julia-cpu-f32.log
DLETO_BLAS=openblas JULIA_LOAD_PATH='@:@dleto-bench:@stdlib' JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --compiled-modules=existing --project=. bench/python_vs_julia/compare_metal.jl /tmp/opendleto-perf 8 16 24 32 48 64 96 128 | tee /tmp/opendleto-perf/julia-metal.log
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python/.venv/bin/python bench/python_vs_julia/compare_accel.py --input /tmp/opendleto-perf --sizes 8 16 24 32 48 64 96 128 --repeats 11
python/.venv/bin/python bench/python_vs_julia/check_parity.py /tmp/opendleto-perf
python3 bench/python_vs_julia/plot_growth.py /tmp/opendleto-perf
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python/.venv/bin/python bench/python_vs_julia/profile_normal.py /tmp/opendleto-perf
DLETO_BLAS=openblas JULIA_LOAD_PATH='@:@dleto-bench:@stdlib' JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --compiled-modules=existing --project=. bench/python_vs_julia/profile_normal.jl /tmp/opendleto-perf
```

The two GPU commands require a process with Apple GPU access. On this machine the tool sandbox hides both Metal and MPS, so the GPU runs used an unsandboxed process. The CSV also includes a separate Python MPS device-resident forward time. MPS and Metal do not offer `float64` arithmetic here; the earlier CPU-only `float64` comparison remains a separate result.
