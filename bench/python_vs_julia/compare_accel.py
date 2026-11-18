"""Compare NumPy CPU and PyTorch MPS forward maps on shared Float32 data."""
from __future__ import annotations

import argparse
import json
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
        start = perf_counter()
        call()
        if synchronize:
            synchronize()
        samples.append(perf_counter() - start)
    return float(np.median(samples))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--sizes', nargs='+', type=int, default=[8, 32, 64])
    parser.add_argument('--repeats', type=int, default=11)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable in this process; run outside a GPU-restricting sandbox')
    report = []
    for n in args.sizes:
        tensor = np.fromfile(args.input / f'tensor_{n}.bin', dtype='<f8').reshape((n,n,n), order='F').astype(np.float32)
        x = np.fromfile(args.input / f'x_{n}.bin', dtype='<f8').astype(np.float32)
        y = np.fromfile(args.input / f'y_{n}.bin', dtype='<f8').reshape((1,n,n,n), order='F').astype(np.float32)
        cpu = DerivationMap(tensor)
        gpu = DerivationMap(torch.as_tensor(tensor, device='mps'))
        cpu_forward = median_time(lambda: cpu.forward(x), repeats=args.repeats)
        cpu_adjoint = median_time(lambda: cpu.adjoint(y), repeats=args.repeats)
        gpu_forward = median_time(lambda: gpu.forward(x), synchronize=torch.mps.synchronize, repeats=args.repeats)
        gpu_forward_host = median_time(lambda: gpu.forward(x).cpu().numpy(),
                                       synchronize=torch.mps.synchronize, repeats=args.repeats)
        cpu_result = cpu.forward(x)
        gpu_result = gpu.forward(x).cpu().numpy()
        (args.input / f"python_cpu_forward_f32_{n}.bin").write_bytes(cpu_result.tobytes(order="F"))
        (args.input / f"python_mps_forward_f32_{n}.bin").write_bytes(gpu_result.tobytes(order="F"))
        (args.input / f"python_cpu_adjoint_f32_{n}.bin").write_bytes(cpu.adjoint(y).tobytes(order="F"))
        report.append(dict(size=n, dtype='float32', cpu_numpy_forward_seconds=cpu_forward,
                           cpu_numpy_adjoint_seconds=cpu_adjoint,
                           gpu_mps_forward_seconds=gpu_forward,
                           gpu_mps_forward_host_seconds=gpu_forward_host,
                           relative_forward_difference=float(np.linalg.norm(cpu_result-gpu_result)/np.linalg.norm(cpu_result)),
                           gpu_adjoint='unsupported'))
    (args.input / 'accel.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
