"""Generate shared Float64 Fortran-order inputs without running a CPU map.

Files are streamed in flat Fortran order, with bounded temporary memory. Existing
complete files are reused; incomplete temporary files are replaced on retry.
"""
import argparse
import os
from pathlib import Path
import numpy as np


def generate(folder, n):
    rng = np.random.default_rng(20261118 + n)
    for name, count in ((f'tensor_{n}.bin', n**3), (f'x_{n}.bin', 3*n*n),
                        (f'y_{n}.bin', n**3)):
        path = folder / name
        # Advance the RNG even when a file exists, keeping later files identical.
        complete = path.exists() and path.stat().st_size == count * 8
        temporary = path.with_suffix('.tmp')
        stream = None if complete else temporary.open('wb')
        try:
            for start in range(0, count, 1_000_000):
                values = rng.standard_normal(min(1_000_000, count-start)).astype('<f8', copy=False)
                if stream:
                    values.tofile(stream)
        finally:
            if stream:
                stream.close()
        if not complete:
            os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', nargs='+', type=int, default=[8,16,24,32,48,64,96,128,192,256,384,512])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for n in args.sizes:
        if n <= 0:
            parser.error('sizes must be positive')
        generate(args.output, n)
        print(f'INPUT n={n} complete', flush=True)


if __name__ == '__main__':
    main()
