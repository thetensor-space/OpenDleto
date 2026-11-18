# Apple silicon comparison: Julia and Python

This sweep applies the same universal-chisel forward map to shared `float32`
order-three tensors. The CPU comparison uses Apple Accelerate in both Julia and
NumPy through `128³`. The larger cases time only Julia Metal and PyTorch MPS.
It ran on an Apple M4 Max with 64 GiB unified memory and macOS 27.2.1, using
Julia 1.12.3, NumPy 2.5.5, and PyTorch 2.15.0.

The GPU timings include each backend's public host-facing call: the map applies
the GPU operations and returns an output on the CPU. The tensor upload and map
construction occur before timing. MPS device-resident times are saved in the
CSV as additional context; Julia Metal's public map has no matching
device-resident call. Julia Metal also supports an adjoint, whose timings are
saved in the CSV; Python MPS currently has no GPU adjoint.

Each Julia size runs in a fresh process. Packages were precompiled before the
sweep, and five calls warm up each map before 11 calls are timed. The median is
reported. The same warmup and repeat counts apply to Python. Both GPU paths
synchronize each timed call. Julia was configured with eight host threads,
Python's PyTorch reported twelve, and no thread-limit environment variables
were set. The graph labels these host thread settings. The number of workers
actually used by Accelerate and the GPU devices was not measured; device
parallelism is distinct from host thread counts.
The later [video-sized comparison](VIDEO-2026-11-18.md) reruns both GPU paths
with 12 configured host threads and reports clip-equivalent frames per second.

The Julia GPU run used Metal.jl 1.11.1 in a temporary environment. The pinned
Metal.jl 1.9.3 in this checkout failed to compile its permutation kernel on
macOS 27.2.1. The temporary environment developed this local Dleto checkout
and did not change the repository's dependency files.

## Results

The plot and table below show time for one forward-map application. Lower is
faster. The CPU lines stop at `128³`; the GPU lines extend to `512³`. The
vertical axis is logarithmic.

![Line plot comparing Julia and Python forward-map times on Apple silicon. Legend labels show eight configured Julia host threads for the Julia CPU and GPU runs, twelve PyTorch host threads for Python MPS, and automatic, unmeasured Accelerate worker counts for both CPU runs. The CPU lines end at tensor side length 128; the GPU lines continue through 512. Lower time is faster.](apple-growth-2026-11-18.png)

| Tensor | Julia CPU · Accelerate | Python CPU · Accelerate | Julia GPU · Metal | Python GPU · MPS |
| --- | ---: | ---: | ---: | ---: |
| 8³ | 0.002 | 0.021 | 0.833 | 0.629 |
| 16³ | 0.016 | 0.032 | 1.460 | 0.667 |
| 24³ | 0.040 | 0.047 | 1.148 | 0.616 |
| 32³ | 0.096 | 0.070 | 1.024 | 0.591 |
| 48³ | 0.270 | 0.228 | 0.968 | 0.639 |
| 64³ | 0.763 | 1.605 | 0.991 | 0.667 |
| 96³ | 2.626 | 4.343 | 1.311 | 0.916 |
| 128³ | 6.445 | 14.822 | 2.373 | 1.479 |
| 192³ | — | — | 6.206 | 3.262 |
| 256³ | — | — | 10.116 | 8.127 |
| 384³ | — | — | 33.868 | 25.613 |
| 512³ | — | — | 80.929 | 68.628 |

All table values are milliseconds. At `128³`, Julia with Accelerate took
6.45 ms on the CPU, versus 14.82 ms for Python with Accelerate. The old Julia
CPU result with OpenBLAS was 20.19 ms, but that was a separate run with a
different BLAS and thread setup. Python MPS was faster than Julia Metal for the
host-facing forward call at every measured size; its lead narrowed from about
1.60× at `128³` to about 1.18× at `512³`. These times describe the current
map implementations and package versions, not a general language ranking.

The 16 sampled output entries matched exactly between the two GPU paths at
every size. Their full-output norms differed by at most `3.51e-5` relative at
`512³`. Sampling and norms provide a useful check, not a full elementwise
validation of the largest outputs. The full `128³` CPU forward outputs differed
by `2.84e-7` relative. Python MPS still has no GPU adjoint, so
this forward-map comparison does not rank complete derivation solves.

The measurements are in [apple-results-2026-11-18.csv](apple-results-2026-11-18.csv).

## Reproduce

Run from the repository root. The `--resume` options reuse completed per-size
checkpoints. Create a temporary Julia environment with Metal.jl 1.11.1 and this
checkout developed into it; keep the repository's pinned manifest unchanged.

```sh
julia --project=/tmp/dleto-metal-bench -e 'using Pkg; Pkg.add(name="Metal", version="1.11.1"); Pkg.develop(path=pwd())'
python/.venv/bin/python bench/python_vs_julia/generate_inputs.py --output /tmp/opendleto-apple --sizes 8 16 24 32 48 64 96 128 192 256 384 512
DLETO_BENCH_APPLE=1 JULIA_NUM_THREADS=8 julia --compiled-modules=existing --project=. bench/python_vs_julia/compare.jl /tmp/opendleto-apple float32 8 16 24 32 48 64 96 128 > /tmp/opendleto-apple/julia-cpu-accelerate.log
JULIA_NUM_THREADS=8 python3 bench/python_vs_julia/run_metal.py --input /tmp/opendleto-apple --sizes 8 16 24 32 48 64 96 128 192 256 384 512 --project /tmp/dleto-metal-bench --resume > /tmp/opendleto-apple/run-metal.log
python/.venv/bin/python bench/python_vs_julia/compare_apple_accel.py --input /tmp/opendleto-apple --sizes 8 16 24 32 48 64 96 128 192 256 384 512 --repeats 11 --resume > /tmp/opendleto-apple/run-python.log
python/.venv/bin/python bench/python_vs_julia/check_apple_parity.py /tmp/opendleto-apple --sizes 8 16 24 32 48 64 96 128 192 256 384 512
MPLCONFIGDIR=/tmp/opendleto-mpl python3 bench/python_vs_julia/plot_apple_growth.py /tmp/opendleto-apple
```

The older [OpenBLAS comparison](README.md) remains available for reference.
