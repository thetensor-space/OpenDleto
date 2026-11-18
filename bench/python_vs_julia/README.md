# OpenDleto Python and Julia: CPU and Apple GPU growth

We measured the universal-chisel derivation map on the **same generic order-three tensors and operator vectors** in four implementations: Julia's dense array CPU kernel, Python's NumPy CPU kernel, Julia's Metal GPU kernel, and Python's PyTorch MPS GPU kernel. All four used `float32` on an Apple M4 Max on 2026-11-18. The input values were generated once, saved in a portable binary format, and cast to `float32` in each process.

![Line plot of forward-map time versus tensor side length from 8 to 128. Julia CPU and Python NumPy CPU are fastest at small sizes and rise sharply. Both GPU lines begin near one millisecond; Python MPS reaches about 1.5 milliseconds at 128, while Julia Metal reaches about 6.8 milliseconds. The vertical axis is logarithmic.](growth-2026-11-18.png)

The plot shows **time for one forward-map application**. Lower is faster. The vertical axis uses a logarithmic scale so the small and large cases remain visible together. Each point is the median of 11 calls after five warmups, with one CPU and BLAS thread. Package loading, map construction, and the one-time tensor upload are excluded. Both GPU timings end with an output on the CPU; host-facing operator coordinates are supplied to both.

| Tensor | Julia CPU | Python NumPy CPU | Julia Metal GPU | Python MPS GPU |
| --- | ---: | ---: | ---: | ---: |
| 8³ | 0.0018 | 0.0204 | 1.1545 | 1.0958 |
| 16³ | 0.0131 | 0.0315 | 1.2054 | 0.7007 |
| 24³ | 0.0483 | 0.0470 | 1.0785 | 0.6093 |
| 32³ | 0.1233 | 0.0690 | 1.0199 | 0.5902 |
| 48³ | 0.4928 | 0.2261 | 1.0490 | 0.6402 |
| 64³ | 1.5503 | 1.5686 | 1.0913 | 0.7078 |
| 96³ | 6.8992 | 4.2345 | 6.0097 | 1.0750 |
| 128³ | 20.1945 | 14.2777 | 6.7810 | 1.5181 |

All times are milliseconds. At `128³`, Julia Metal was about 3.0 times faster than Julia CPU, and Python MPS was about 9.4 times faster than Python NumPy CPU. The Python MPS forward call was about 4.5 times faster than the Julia Metal call in this particular implementation and run. Around `64³`, both GPU paths began to beat their own CPU counterparts. Julia Metal's forward time rose from about 1 ms at `64³` to about 6–7 ms at `96³`; a second sweep reproduced that jump. These are measurements of the present kernels, including their host/device interface costs, rather than a general language or GPU ranking.

## Adjoint and numerical agreement

Julia Metal also supports the adjoint map. Its measured adjoint times were 1.031 ms at `64³`, 1.312 ms at `96³`, and 1.763 ms at `128³`; Julia CPU took 1.459, 6.344, and 19.244 ms respectively. Python NumPy CPU took 0.669, 2.241, and 6.535 ms. The current Python MPS backend does not implement the GPU adjoint, so no four-way adjoint or full GPU solver comparison is claimed.

The saved outputs passed [the parity check](check_parity.py): maximum relative difference across forward and available adjoint outputs was `2.34e-6`. Forward outputs from the two CPU implementations matched exactly in `float32`; Julia Metal and Python MPS forward outputs agreed with Julia CPU to at most `6.8e-8` relative error. Float32 rounding explains the larger adjoint differences.

A separate complete dense SVD on the same generic `4³` tensor took 0.109 ms in Julia and 0.922 ms in Python; both found nullity 2 at relative tolerance `1e-8`. That is a CPU-only small-map comparison. It does not compare the packages' different high-level solver policies.

## Reproduce

The measured data are in [results-2026-11-18.csv](results-2026-11-18.csv). The scripts generate fresh shared inputs, save numerical outputs for parity checks, and regenerate the plot. Run from the repository root with Julia 1.12, the shared `@dleto-bench` Julia environment containing Metal, the project-local Python environment containing `opendleto[torch,dev]`, and a plotting Python environment containing Matplotlib:

```sh
OPENBLAS_NUM_THREADS=1 python/.venv/bin/python bench/python_vs_julia/compare.py --output /tmp/opendleto-perf --sizes 4 8 16 24 32 48 64 96 128 --repeats 11
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --compiled-modules=existing --project=. bench/python_vs_julia/compare.jl /tmp/opendleto-perf float32 8 16 24 32 48 64 96 128 | tee /tmp/opendleto-perf/julia-cpu-f32.log
JULIA_LOAD_PATH='@:@dleto-bench:@stdlib' JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --compiled-modules=existing --project=. bench/python_vs_julia/compare_metal.jl /tmp/opendleto-perf 8 16 24 32 48 64 96 128 | tee /tmp/opendleto-perf/julia-metal.log
OPENBLAS_NUM_THREADS=1 python/.venv/bin/python bench/python_vs_julia/compare_accel.py --input /tmp/opendleto-perf --sizes 8 16 24 32 48 64 96 128 --repeats 11
python/.venv/bin/python bench/python_vs_julia/check_parity.py /tmp/opendleto-perf
python3 bench/python_vs_julia/plot_growth.py /tmp/opendleto-perf
```

The two GPU commands require a process with Apple GPU access. On this machine the tool sandbox hides both Metal and MPS, so the GPU runs used an unsandboxed process. The CSV also includes a separate Python MPS device-resident forward time. MPS and Metal do not offer `float64` arithmetic here; the earlier CPU-only `float64` comparison remains a separate result.
