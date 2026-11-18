"""Time complete dense SVD of a shared 4x4x4 derivation map."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from opendleto import DerivationMap

parser = argparse.ArgumentParser()
parser.add_argument('folder', type=Path)
args = parser.parse_args()
n = 4
tensor = np.fromfile(args.folder / 'tensor_4.bin', dtype='<f8').reshape((n,n,n), order='F')
dmap = DerivationMap(tensor)
eye = np.eye(dmap.shape[1])

def solve():
    matrix = np.column_stack([dmap.forward(eye[:,j]).ravel(order='F') for j in range(dmap.shape[1])])
    values = np.linalg.svd(matrix, compute_uv=False)
    return values

solve()
samples = []
for _ in range(7):
    start = perf_counter()
    values = solve()
    samples.append(perf_counter()-start)
print('PYTHON_SVD seconds=',median(samples),'nullity=',dmap.shape[1]-int(np.sum(values>1e-8*values[0])))
