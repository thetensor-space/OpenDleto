"""Create shared little-endian Float32 inputs, streamed in Julia/Fortran order."""
import argparse
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('folder', type=Path)
p.add_argument('--shape', nargs='+', type=int, default=(640, 480, 3, 300))
args = p.parse_args()
if len(args.shape) not in (3, 4) or any(n < 1 for n in args.shape):
    p.error('shape must contain three or four positive dimensions')
args.folder.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261118)
for name, count in [('tensor.f32', np.prod(args.shape)),
                    ('x.f32', sum(n*n for n in args.shape))]:
    path = args.folder / name
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        for start in range(0, int(count), 1_000_000):
            rng.standard_normal(min(1_000_000, int(count)-start), dtype=np.float32).astype('<f4', copy=False).tofile(f)
    tmp.replace(path)
print(f'INPUT shape={tuple(args.shape)} coordinates={sum(n*n for n in args.shape)} folder={args.folder}', flush=True)
