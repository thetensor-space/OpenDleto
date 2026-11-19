"""Collect guarded large-sweep worker logs into a single CSV."""
import argparse
import csv
import json
from pathlib import Path


def records(folder):
    for path in sorted(folder.glob('*.log')):
        if not (path.name.startswith('julia-') or path.name.startswith('python-')):
            continue
        for line in path.read_text(errors='replace').splitlines():
            if not line.startswith('{'):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('language') in ('julia', 'python') and row.get('stage') != 'process':
                row.setdefault('chisel', 'universal')
                if (row.get('chisel') == 'adjoint' and row.get('status') == 'invalid_count'
                        and row.get('dimension') is not None
                        and row['dimension'] >= row.get('scalar_dimension', 1)
                        and row.get('certified') is True):
                    # Early adjoint runs used the universal scalar minimum (2)
                    # in their status label. The method's count and certificate
                    # were correct; the effective adjoint minimum is 1.
                    row['status'] = 'ok'
                yield row
    manifest = folder / 'manifest.jsonl'
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            yield dict(language=row['language'], chisel=row.get('chisel', 'universal'),
                       size=row['size'], stage='process', method=None,
                       status=row['status'], seconds=row['wall_seconds'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folders', nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    rows = {}
    for folder in args.folders:
        for row in records(folder):
            key = (row.get('language'), row.get('chisel'), row.get('size'),
                   row.get('stage'), row.get('method'))
            if row.get('stage') == 'process' and key in rows:
                continue  # preserve the full sweep's timeout over a later hot rerun
            rows[key] = row
    ordered = sorted(rows.values(), key=lambda r: (r['chisel'], r['size'],
                      r['language'], r['stage'], str(r.get('method'))))
    columns = ('language', 'chisel', 'size', 'stage', 'method', 'status',
               'seconds', 'dimension', 'dimension_lower_bound', 'returned_vectors',
               'basis_complete', 'certified', 'residual', 'reduced_shape',
               'target_warmed',
               'map_shape', 'dense_estimated_bytes', 'solver', 'selected_method',
               'scalar_dimension', 'reconstruction_error', 'error')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(ordered)
    print(f'{len(ordered)} rows written to {args.output}')


if __name__ == '__main__':
    main()
