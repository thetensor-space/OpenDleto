"""Measured matrix-free operation timings; no dense derivation matrix.

Run: python examples/benchmark.py --sizes 32 64 --repeats 3
"""
import argparse
import json
import resource
import time

import numpy as np
from opendleto import DerivationMap


def bench(n, repeats, seed):
    tensor = np.random.default_rng(seed).standard_normal((n, n, n))
    dmap = DerivationMap(tensor)
    coords = np.random.default_rng(seed + 1).standard_normal(dmap.shape[1])
    dmap.forward(coords)  # warmup
    rows = []
    for _ in range(repeats):
        start = time.perf_counter()
        y = dmap.forward(coords)
        forward = time.perf_counter() - start
        start = time.perf_counter()
        dmap.adjoint(y)
        adjoint = time.perf_counter() - start
        rows.append((forward, adjoint))
    return {'shape': tensor.shape, 'parameters': dmap.shape[1],
            'output_elements': dmap.shape[0], 'backend': dmap.backend,
            'dtype': str(tensor.dtype), 'device': dmap.device,
            'forward_seconds': min(v[0] for v in rows),
            'adjoint_seconds': min(v[1] for v in rows),
            'peak_rss_native_units': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', nargs='+', type=int, default=[32, 64, 100, 128])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    for size in args.sizes:
        print(json.dumps(bench(size, args.repeats, args.seed)), flush=True)
