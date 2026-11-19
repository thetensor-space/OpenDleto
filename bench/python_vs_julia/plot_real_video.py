"""Turn run_real_maps.sh logs into a CSV and a plain-language size/time chart."""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def result(path):
    line = next((line for line in path.read_text().splitlines() if line.startswith('RESULT ')), None)
    if line is None:
        raise ValueError(f'No result in {path}')
    if line.startswith('RESULT {'):
        return json.loads(line[7:])['median_seconds']
    match = re.search(r'\bmedian_seconds=([0-9.eE+-]+)', line)
    if not match:
        raise ValueError(f'No median in {path}')
    return float(match.group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', type=Path)
    parser.add_argument('output', type=Path, help='PNG chart output')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for orientation in ('landscape', 'portrait'):
        for size in (720, 1080):
            width, height = ((size * 16 // 9, size) if orientation == 'landscape'
                             else (size, size * 16 // 9))
            gib = width * height * 300 * 4 / 2**30
            for backend in ('julia', 'python'):
                case = f'{orientation}-{size}p'
                rows.append(dict(source=orientation, resolution=f'{width}x{height}',
                                 tensor_shape=f'{width}x{height}x300',
                                 elements=width * height * 300, tensor_gib=gib,
                                 operation='one derivation-map call', backend=backend,
                                 host_threads=12,
                                 median_seconds=result(args.logs / f'{case}-{backend}.log')))
    csv_path = args.output.with_suffix('.csv')
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = {'julia': '#17669b', 'python': '#c0541e'}
    markers = {'landscape': 'o', 'portrait': 's'}
    for backend in ('julia', 'python'):
        for orientation in ('landscape', 'portrait'):
            subset = sorted((r for r in rows if r['backend'] == backend and
                             r['source'] == orientation), key=lambda r: r['tensor_gib'])
            ax.plot([r['tensor_gib'] for r in subset],
                    [r['median_seconds'] for r in subset],
                    marker=markers[orientation], color=colors[backend], linewidth=1.7,
                    label=f'{"Julia Metal" if backend == "julia" else "Python MPS"} · {orientation} · 12 host threads')
    ax.set_xlabel('Decoded video tensor size (GiB, grayscale Float32, 300 frames)')
    ax.set_ylabel('Seconds for one whole-window derivation-map call')
    ax.set_title('Real 10-second video clips: 720p and 1080p')
    ax.set_xticks(sorted({r['tensor_gib'] for r in rows}),
                  [f"720p ({720 * 1280 * 300 * 4 / 2**30:.2f} GiB)",
                   f"1080p ({1080 * 1920 * 300 * 4 / 2**30:.2f} GiB)"])
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    print(f'Wrote {csv_path} and {args.output}')


if __name__ == '__main__':
    main()
