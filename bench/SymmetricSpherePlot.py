#!/usr/bin/env python3
"""Optional publication-style PNG from the demo's exported point cloud.
Usage: python3 bench/SymmetricSpherePlot.py path/to/*_points.csv
Requires matplotlib; Julia demo itself has no plotting-package dependency.
"""
import csv
import os
import re
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'sphere-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(path):
    groups = {name: [] for name in ('original', 'scrambled', 'recovered')}
    with path.open() as f:
        for row in csv.DictReader(f):
            groups[row['series']].append(tuple(float(row[k]) for k in ('x', 'y', 'z', 'value')))
    # The source CSV retains every original support point; apply the same
    # display cutoff used by the demo for the recovered tensor.
    if groups['original']:
        cutoff = .2 * max(abs(p[3]) for p in groups['original'])
        groups['original'] = [p for p in groups['original'] if abs(p[3]) >= cutoff]
    fig = plt.figure(figsize=(13, 4.8), facecolor='white', layout='constrained')
    titles = ('Original sphere octant', 'Hidden by random orthogonals', 'Recovered by symmetric chiselling')
    colors = ('#2563a6', '#b45309', '#15803d')
    for i, (name, pts) in enumerate(groups.items()):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        ax.set_title(titles[i], fontsize=11, pad=14)
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts],
                       c=colors[i], s=5, alpha=0.45 if name == 'scrambled' else 0.7,
                       linewidths=0, rasterized=True)
        else:
            ax.text2D(.15, .5, 'No non-scalar recovery', transform=ax.transAxes)
        ax.view_init(elev=24, azim=38)
        ax.set_box_aspect((1, 1, 1))
        limit = max((max(p[:3]) for p in pts), default=1.0)
        limit = max(limit, 1.0)
        for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            setter(0, limit)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.tick_params(labelsize=7)
    dim = re.search(r'_d(\d+)_', path.name)
    prefix = ' × '.join([dim.group(1)] * 3) + ' tensor: ' if dim else ''
    fig.suptitle(prefix + 'original → hidden → recovered', fontsize=14)
    fig.supxlabel('Physical coordinates for original/recovered; recovered axes aligned using the known transform.\n'
                  'Hidden panel uses array coordinates. Display cutoff: 20% of each tensor’s maximum amplitude.', fontsize=8)
    out = path.with_name(path.name.removesuffix('_points.csv') + '_recovery.png')
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(out)
    return out


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        render(Path(arg))
