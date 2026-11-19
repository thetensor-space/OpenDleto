"""Generate full-mode-rank graded cubes at the requested tensor sizes.

The first two axes have grades 0 and 1; the third has grades 0, 1 and 2.
Only entries with grade(i)+grade(j)=grade(k) are nonzero, so the three grade
matrices give a non-scalar derivation. Random coefficients within the allowed
blocks suppress accidental extra symmetries. Arrays are shared little-endian
Float64 in Fortran order.
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--sizes', nargs='+', type=int, default=[20, 40, 60, 80, 100])
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    for n in args.sizes:
        if n < 2:
            parser.error('sizes must be at least two')
        rng = np.random.default_rng(20261118 + n)
        if n % 4:
            parser.error('sizes must be divisible by four for equal grade blocks')
        a = np.repeat((0, 1), n // 2)
        b = np.repeat((0, 1), n // 2)
        c = np.repeat((0, 1, 2), (n // 4, n // 2, n // 4))
        allowed = a[:, None, None] + b[None, :, None] == c[None, None, :]
        tensor = np.zeros((n, n, n), dtype=np.float64, order='F')
        tensor[allowed] = 0.5 + rng.random(int(allowed.sum()))
        path = args.folder / f'case_{n}.bin'
        tensor.ravel(order='F').astype('<f8').tofile(path)
        print(f'INPUT n={n} entries={tensor.size} nonzeros={int(allowed.sum())} path={path}', flush=True)


if __name__ == '__main__':
    main()
