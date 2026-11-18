"""Generate shared inputs and benchmark OpenDleto's NumPy matrix-free map.

Run from repository root with OPENBLAS_NUM_THREADS=1 and a Python environment
containing the editable opendleto package. The Julia companion reads the same
binary Float64 tensors and vectors; all arrays use Fortran storage order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from opendleto import DerivationMap


def write_f64(path: Path, array: np.ndarray) -> None:
    path.write_bytes(np.asarray(array, dtype='<f8').tobytes(order='F'))


def timed(call, repeats=5):
    call()
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        call()
        samples.append(perf_counter() - start)
    return float(np.median(samples))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', nargs='+', type=int, default=[8, 32])
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = []
    for n in args.sizes:
        rng = np.random.default_rng(20261118 + n)
        tensor = rng.standard_normal((n, n, n))
        x = rng.standard_normal(3 * n * n)
        y = rng.standard_normal((1, n, n, n))
        write_f64(args.output / f'tensor_{n}.bin', tensor)
        write_f64(args.output / f'x_{n}.bin', x)
        write_f64(args.output / f'y_{n}.bin', y)
        dmap = DerivationMap(tensor)
        forward = timed(lambda: dmap.forward(x), args.repeats)
        adjoint = timed(lambda: dmap.adjoint(y), args.repeats)
        write_f64(args.output / f'python_forward_{n}.bin', dmap.forward(x))
        write_f64(args.output / f'python_adjoint_{n}.bin', dmap.adjoint(y))
        report.append(dict(size=n, forward_seconds=forward,
                           adjoint_seconds=adjoint, dtype='float64',
                           backend='numpy', repeats=args.repeats))
    (args.output / 'python.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
