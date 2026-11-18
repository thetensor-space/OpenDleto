"""Benchmark Float32 NumPy (through 128) and synchronized MPS forward maps."""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import os
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from opendleto import DerivationMap


def median_time(call, *, synchronize=None, repeats=11):
    for _ in range(5):
        call()
        if synchronize:
            synchronize()
    samples = []
    for _ in range(repeats):
        if synchronize:
            synchronize()
        start = perf_counter()
        call()
        if synchronize:
            synchronize()
        samples.append(perf_counter() - start)
    return float(np.median(samples))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--sizes', nargs='+', type=int, default=[8,16,24,32,48,64,96,128,192,256,384,512])
    parser.add_argument('--repeats', type=int, default=11)
    parser.add_argument('--host-threads', type=int, default=None,
                        help='Set PyTorch intra-op host threads before MPS timing')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 1 for n in args.sizes) or (args.host_threads is not None and args.host_threads < 1):
        parser.error('sizes and repeats must be positive')
    if args.host_threads is not None:
        torch.set_num_threads(args.host_threads)
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable in this process')
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    metadata = dict(platform=platform.platform(), numpy=np.__version__, torch=torch.__version__,
                    numpy_config=config.getvalue(), torch_threads=torch.get_num_threads(),
                    thread_environment={k: os.environ.get(k) for k in
                      ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})
    for n in args.sizes:
        result_path = args.input / f'accel_{n}.json'
        if args.resume and result_path.exists():
            print(f'SKIP n={n} checkpoint={result_path}', flush=True)
            continue
        tensor = np.fromfile(args.input / f'tensor_{n}.bin', dtype='<f8').reshape((n,n,n), order='F').astype(np.float32)
        x = np.fromfile(args.input / f'x_{n}.bin', dtype='<f8').astype(np.float32)
        gpu = DerivationMap(torch.as_tensor(tensor, device='mps'))
        result = dict(size=n, dtype='float32', repeats=args.repeats, metadata=metadata,
                      gpu_mps_forward_seconds=median_time(lambda: gpu.forward(x), synchronize=torch.mps.synchronize, repeats=args.repeats),
                      gpu_mps_forward_host_seconds=median_time(lambda: gpu.forward(x).cpu().numpy(), synchronize=torch.mps.synchronize, repeats=args.repeats),
                      gpu_adjoint='unsupported')
        gpu_result = gpu.forward(x).cpu().numpy()
        flattened = gpu_result.ravel(order='F')
        indices = np.arange(16, dtype=np.int64) * (flattened.size - 1) // 15
        flattened[indices].astype('<f4', copy=False).tofile(args.input / f'python_mps_sample_{n}.bin')
        result['output_norm'] = float(np.linalg.norm(flattened))
        result['sample_count'] = 16
        if n <= 128:
            y = np.fromfile(args.input / f'y_{n}.bin', dtype='<f8').reshape((1,n,n,n), order='F').astype(np.float32)
            cpu = DerivationMap(tensor)
            result.update(cpu_numpy_forward_seconds=median_time(lambda: cpu.forward(x), repeats=args.repeats),
                          cpu_numpy_adjoint_seconds=median_time(lambda: cpu.adjoint(y), repeats=args.repeats))
            cpu_result = cpu.forward(x)
            result['relative_forward_difference'] = float(np.linalg.norm(cpu_result-gpu_result)/np.linalg.norm(cpu_result))
        temporary = result_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2))
        os.replace(temporary, result_path)
        print(f"RESULT n={n} gpu_mps_forward_host_seconds={result['gpu_mps_forward_host_seconds']} "
              f"gpu_mps_forward_seconds={result['gpu_mps_forward_seconds']} "
              f"cpu_numpy_forward_seconds={result.get('cpu_numpy_forward_seconds', 'skipped')}", flush=True)


if __name__ == '__main__':
    main()
