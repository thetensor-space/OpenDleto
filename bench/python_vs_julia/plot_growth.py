"""Merge the four-backend sweep and draw the shared-tensor growth plot.

Run with a Python environment containing matplotlib and NumPy:
  python bench/python_vs_julia/plot_growth.py /tmp/opendleto-perf
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def julia_rows(path: Path):
    result = {}
    for line in path.read_text().splitlines():
        if not line.startswith('RESULT '):
            continue
        fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
        result[int(fields['n'])] = (float(fields['forward_seconds']),
                                    float(fields['adjoint_seconds']))
    return result


def main(folder: Path):
    python_rows = {row['size']: row for row in json.loads((folder / 'accel.json').read_text())}
    julia_cpu = julia_rows(folder / 'julia-cpu-f32.log')
    julia_gpu = julia_rows(folder / 'julia-metal.log')
    sizes = sorted(set(python_rows) & set(julia_cpu) & set(julia_gpu))
    rows = []
    for n in sizes:
        py = python_rows[n]
        rows.append({
            'n': n,
            'julia_cpu_forward_ms': 1e3 * julia_cpu[n][0],
            'python_numpy_cpu_forward_ms': 1e3 * py['cpu_numpy_forward_seconds'],
            'julia_metal_forward_ms': 1e3 * julia_gpu[n][0],
            'python_mps_forward_ms': 1e3 * py['gpu_mps_forward_host_seconds'],
            'python_mps_device_forward_ms': 1e3 * py['gpu_mps_forward_seconds'],
            'julia_cpu_adjoint_ms': 1e3 * julia_cpu[n][1],
            'python_numpy_cpu_adjoint_ms': 1e3 * py['cpu_numpy_adjoint_seconds'],
            'julia_metal_adjoint_ms': 1e3 * julia_gpu[n][1],
        })
    out = Path(__file__).parent
    with (out / 'results-2026-11-18.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(9.4, 5.4), layout='constrained')
    series = [
        ('Julia CPU', 'julia_cpu_forward_ms', '#166534', 'o'),
        ('Python NumPy CPU', 'python_numpy_cpu_forward_ms', '#2563eb', 's'),
        ('Julia Metal GPU', 'julia_metal_forward_ms', '#d97706', '^'),
        ('Python MPS GPU (host output)', 'python_mps_forward_ms', '#be185d', 'D'),
    ]
    for label, column, color, marker in series:
        ax.plot(sizes, [row[column] for row in rows], marker=marker, linewidth=2,
                markersize=5.5, color=color, label=label)
    ax.set_yscale('log')
    ax.set_xticks(sizes)
    ax.set_xlabel('Tensor side length n (shape n × n × n)')
    ax.set_ylabel('Forward-map time (milliseconds, log scale)')
    ax.set_title('OpenDleto forward-map growth on the same float32 tensors')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(loc='upper left', frameon=True)
    fig.savefig(out / 'growth-2026-11-18.png', dpi=180)
    plt.close(fig)
    for row in rows:
        print(row)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
