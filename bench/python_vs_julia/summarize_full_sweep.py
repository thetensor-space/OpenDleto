"""Collect the two JSONL benchmark streams into one CSV and parity summary."""
import csv
import json
import sys
from pathlib import Path


def read_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith('{'):
            continue  # Julia's solver may print progress messages.
        rows.append(json.loads(line))
    return rows


def main():
    folder = Path(sys.argv[1])
    rows = []
    for language, path in (("python", folder / 'python.jsonl'),
                           ("julia", folder / 'julia.log')):
        for row in read_rows(path):
            row['language'] = language
            rows.append(row)
    columns = ('language', 'case', 'chisel', 'method', 'stage', 'seconds',
               'dimension', 'residual', 'status', 'certified', 'exact',
               'raw_shape', 'reduced_shape', 'returned_vectors',
               'nontrivial_dimension', 'scalar_dimension', 'operator_space')
    with (folder / 'full-sweep.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    checks = []
    groups = sorted({(r['case'], r['chisel']) for r in rows})
    for case, chisel in groups:
        dims = {}
        for language in ('python', 'julia'):
            group = [r for r in rows if r['case'] == case and
                     r['chisel'] == chisel and r['language'] == language]
            dense = next((r for r in group if r['method'] == 'dense_svdvals'
                          and r['stage'] == 'reduced'), None)
            if dense is None:
                raise RuntimeError(f'missing {language} {case} {chisel} reduced dimension')
            dims[language] = dense['dimension']
            for row in group:
                if row['stage'] == 'reduced_solve':
                    checks.append(dict(case=case, chisel=chisel,
                                       language=language, method=row['method'],
                                       dense_dimension=dense['dimension'],
                                       solved_dimension=row['dimension'],
                                       agrees=(row['dimension'] == dense['dimension']
                                               if row['dimension'] is not None else None),
                                       residual=row['residual'],
                                       certified=row.get('certified')))
        checks.append(dict(case=case, chisel=chisel, language='cross',
                           method='dense_svdvals',
                           python_dimension=dims['python'],
                           julia_dimension=dims['julia'],
                           agrees=dims['python'] == dims['julia']))
    (folder / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(f'{len(rows)} rows; {sum(x["agrees"] is False for x in checks)} dimension disagreements; '
          f'{sum(x["agrees"] is None for x in checks)} failed solves')
    for row in checks:
        if row['agrees'] is False:
            print('MISMATCH', row)


if __name__ == '__main__':
    main()
