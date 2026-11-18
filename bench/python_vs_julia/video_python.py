"""Time one Float32 forward map including GPU-to-host transfer, in seconds."""
import argparse
import contextlib
import io
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
from opendleto import DerivationMap

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('folder', type=Path)
p.add_argument('--shape', nargs='+', type=int, default=(640, 480, 3, 300))
p.add_argument('--backend', choices=('cpu', 'mps'), required=True)
p.add_argument('--warmups', type=int, default=3)
p.add_argument('--repeats', type=int, default=5)
a = p.parse_args()
if len(a.shape) not in (3, 4) or min(a.shape) < 1 or a.warmups < 0 or a.repeats < 1:
    p.error('shape must contain three or four positive dimensions; repeats must be positive')
tensor = np.fromfile(a.folder / 'tensor.f32', dtype='<f4')
x = np.fromfile(a.folder / 'x.f32', dtype='<f4')
assert tensor.size == np.prod(a.shape) and x.size == sum(n*n for n in a.shape)
tensor = tensor.reshape(a.shape, order='F')
config = io.StringIO()
with contextlib.redirect_stdout(config):
    np.show_config()
if a.backend == 'cpu':
    if 'accelerate' not in config.getvalue().lower():
        raise RuntimeError('NumPy is not configured with Accelerate')
    sync = lambda: None
    host = lambda y: y
    torch_version = torch_threads = None
else:
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable')
    tensor = torch.as_tensor(tensor.copy(), device='mps')
    sync = torch.mps.synchronize
    host = lambda y: y.cpu().numpy()
    torch_version, torch_threads = torch.__version__, torch.get_num_threads()
dmap = DerivationMap(tensor)
print(f'START backend={a.backend} shape={tuple(a.shape)} warmups={a.warmups} repeats={a.repeats}', flush=True)
def apply():
    sync()
    start = time.perf_counter()
    result = host(dmap.forward(x))
    sync()
    return time.perf_counter() - start, result
for _ in range(a.warmups):
    _, result = apply()
    del result
print(f'WARMUP_COMPLETE backend={a.backend} shape={tuple(a.shape)}', flush=True)
samples = []
for i in range(1, a.repeats + 1):
    elapsed, result = apply()
    assert result.shape == (1, *a.shape)
    samples.append(elapsed)
    # Only retain a few values, not the video-sized output.
    sample = result.ravel(order='F')[np.arange(16, dtype=np.int64) * (result.size-1) // 15].tolist()
    norm = float(np.linalg.norm(result))
    del result
    print(f'SAMPLE {i} seconds={elapsed}', flush=True)
print('RESULT ' + json.dumps(dict(backend=a.backend, shape=a.shape, coordinate_length=x.size,
    output_shape=[1, *a.shape], warmups=a.warmups, repeats=a.repeats,
    seconds=samples, median_seconds=float(np.median(samples)), sample=sample, output_norm=norm, clip_seconds=10, fps=30, realtime_factor=10/float(np.median(samples)),
    platform=platform.platform(), numpy=np.__version__, numpy_blas='Accelerate' if 'accelerate' in config.getvalue().lower() else 'other',
    torch=torch_version, torch_threads=torch_threads,
    thread_environment={k: os.environ.get(k) for k in ('VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})), flush=True)
