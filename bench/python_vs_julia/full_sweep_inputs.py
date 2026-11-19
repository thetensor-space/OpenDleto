"""Generate deterministic shared Float64 tensors for the complete solver sweep."""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261118)
    generic = rng.standard_normal((4, 4, 4))
    diagonal = np.zeros((4, 4, 4))
    for i in range(4):
        diagonal[i, i, i] = i + 1
    embedded = np.zeros((5, 5, 5))
    embedded[:3, :3, :3] = rng.standard_normal((3, 3, 3))
    generic8 = rng.standard_normal((8, 8, 8))
    diagonal8 = np.zeros((8, 8, 8))
    for i in range(8):
        diagonal8[i, i, i] = i + 1
    embedded10 = np.zeros((10, 10, 10))
    embedded10[:6, :6, :6] = rng.standard_normal((6, 6, 6))
    for name, array in (("generic4", generic), ("diagonal4", diagonal),
                        ("embedded5", embedded), ("generic8", generic8),
                        ("diagonal8", diagonal8), ("embedded10", embedded10)):
        array.ravel(order='F').astype('<f8').tofile(args.folder / f'case_{name}.bin')
        print(f'{name} {" ".join(map(str, array.shape))}', flush=True)


if __name__ == '__main__':
    main()
