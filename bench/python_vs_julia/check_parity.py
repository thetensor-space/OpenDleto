"""Validate saved four-backend outputs on the same Float32 input tensors."""
from pathlib import Path
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('folder', type=Path)
args = parser.parse_args()
worst = 0.0
for n in (8, 16, 24, 32, 48, 64, 96, 128):
    def vec(name):
        return np.fromfile(args.folder / f'{name}_{n}.bin', dtype='<f4').astype(np.float64)
    cpu = vec('julia_forward_f32')
    for label, value in (('Python CPU', vec('python_cpu_forward_f32')),
                         ('Julia Metal', vec('julia_metal_forward')),
                         ('Python MPS', vec('python_mps_forward_f32'))):
        error = np.linalg.norm(value-cpu) / np.linalg.norm(cpu)
        worst = max(worst, error)
        print(f'{n:3d} {label:12s} forward relative error {error:.3e}')
    cpu_adj = vec('julia_adjoint_f32')
    for label, value in (('Python CPU', vec('python_cpu_adjoint_f32')),
                         ('Julia Metal', vec('julia_metal_adjoint'))):
        error = np.linalg.norm(value-cpu_adj) / np.linalg.norm(cpu_adj)
        worst = max(worst, error)
        print(f'{n:3d} {label:12s} adjoint relative error {error:.3e}')
print(f'Maximum relative error: {worst:.3e}')
if worst > 1e-5:
    raise SystemExit('Parity exceeded 1e-5')
