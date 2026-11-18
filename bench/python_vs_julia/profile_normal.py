"""Time the CPU normal operator D*D on saved Float32 tensors."""
from pathlib import Path
from statistics import median
from time import perf_counter
import argparse
import numpy as np
from opendleto import DerivationMap

parser = argparse.ArgumentParser()
parser.add_argument('folder', type=Path)
args = parser.parse_args()
for n in (64, 96, 128):
    tensor = np.fromfile(args.folder / f'tensor_{n}.bin', dtype='<f8').reshape((n,n,n), order='F').astype('float32')
    x = np.fromfile(args.folder / f'x_{n}.bin', dtype='<f8').astype('float32')
    dmap = DerivationMap(tensor)
    for _ in range(5):
        dmap.normal(x)
    samples = []
    for _ in range(11):
        start = perf_counter()
        dmap.normal(x)
        samples.append(1000 * (perf_counter() - start))
    print(f'NORMAL n={n} backend=python_numpy ms={median(samples)}', flush=True)
