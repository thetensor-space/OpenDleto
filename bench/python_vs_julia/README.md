# Minimal Python–Julia and CPU–MPS comparison

Measured on 2026-11-18 on an Apple M4 Max (macOS 27.2.1), with Python 3.14, NumPy 2.5.5, SciPy 1.18.1, PyTorch 2.15.0, and Julia 1.12.3. One CPU thread and one BLAS thread were used. The same deterministic generic tensors and coordinate vectors were saved once as little-endian Float64 binaries in Julia column-major order. Each timing is the median of seven calls after one warmup; startup, package loading, and map construction are excluded. For MPS, each timed call includes synchronization and the current implementation's small operator transfers to the GPU, but excludes transferring the input tensor and output back to CPU.

## Forward map, `float32` (milliseconds)

| Tensor | Julia CPU | Python NumPy CPU | Python PyTorch MPS |
| --- | ---: | ---: | ---: |
| 8³ | 0.0018 | 0.0239 | 0.8124 |
| 32³ | 0.1333 | 0.0699 | 0.7400 |
| 64³ | 1.5410 | 1.5737 | 0.8308 |

At 64³, the MPS forward call was about 1.9 times faster than NumPy CPU on this machine. At 8³ and 32³ it was slower. These results apply to the current implementations and tensor sizes, not to a full derivation solve. MPS does not support `float64` or `complex128` and the Python GPU adjoint is not implemented.

## CPU map, `float64` (milliseconds)

| Tensor | Julia forward | Python forward | Julia adjoint | Python adjoint |
| --- | ---: | ---: | ---: | ---: |
| 8³ | 0.0021 | 0.0282 | 0.0016 | 0.0191 |
| 32³ | 0.1984 | 0.2232 | 0.1927 | 0.0548 |
| 64³ | 2.7327 | 2.0517 | 2.5342 | 0.3581 |

For the same input bytes, Julia and Python forward outputs matched exactly in both precisions at all three sizes. Relative adjoint differences were at most `2.3e-15` in `float64` and `1.3e-6` in `float32`. The Python CPU and MPS forward outputs agreed within `6.8e-8` relative error.

## Small complete dense SVD

On the same generic 4³ tensor, constructing the dense 64×48 derivation matrix and computing its singular values took 0.109 ms in Julia and 0.922 ms in Python; both reported nullity 2 at relative tolerance `1e-8`. This comparison measures the complete small-map decomposition. It does not compare the packages' distinct high-level solver policies.

## Reproduce

From the repository root, with `python/.venv` containing `opendleto[torch,dev]`:

```sh
OPENBLAS_NUM_THREADS=1 python/.venv/bin/python bench/python_vs_julia/compare.py --output /tmp/opendleto-perf --sizes 4 8 32 64 --repeats 7
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --compiled-modules=existing --project=. bench/python_vs_julia/compare.jl /tmp/opendleto-perf float64 8 32 64
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --compiled-modules=existing --project=. bench/python_vs_julia/compare.jl /tmp/opendleto-perf float32 8 32 64
OPENBLAS_NUM_THREADS=1 python/.venv/bin/python bench/python_vs_julia/compare_accel.py --input /tmp/opendleto-perf --sizes 8 32 64 --repeats 7
OPENBLAS_NUM_THREADS=1 python/.venv/bin/python bench/python_vs_julia/compare_solve.py /tmp/opendleto-perf
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --compiled-modules=existing --project=. bench/python_vs_julia/compare_solve.jl /tmp/opendleto-perf
```

The MPS command must run in a process with GPU access. On this machine, the tool sandbox hides MPS; the measured GPU run used an unsandboxed process. The binary output files from the two CPU scripts can be compared directly after selecting their corresponding dtypes.
