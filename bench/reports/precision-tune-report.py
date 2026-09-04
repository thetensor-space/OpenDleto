#!/usr/bin/env python3
"""Render bench/reports/precision-tune-a.csv as the tables that go in the docs.

    python3 bench/reports/precision-tune-report.py [c|tol]

Cell format: `<nullity>` with `*` = certified, `U` = undecidable at this
storage precision (values above the cut below the data floor).
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KNOB = sys.argv[1] if len(sys.argv) > 1 else "c"
GRID = {
    "c": ["1.0", "2.0", "3.0", "5.0", "10.0", "20.0", "30.0", "100.0", "300.0",
          "1000.0", "10000.0"],
    "tol": ["0.001", "0.0001", "1.0e-5", "1.0e-6", "1.0e-7", "1.0e-8", "1.0e-9"],
}[KNOB]

rows = [r for r in csv.DictReader(open(os.path.join(HERE, "precision-tune-a.csv")))
        if r["knob"] == KNOB]
keys = sorted({(r["case"], r["solver"], r["T"]) for r in rows})
print("%-22s %-13s %-8s %-6s %s" % ("case", "solver", "T", "truth",
                                    " ".join(g.rjust(8) for g in GRID)))
for case, solver, T in keys:
    sub = {r["value"]: r for r in rows
           if (r["case"], r["solver"], r["T"]) == (case, solver, T)}
    truth = next(iter(sub.values()))["truth"]
    cells = []
    for g in GRID:
        r = sub.get(g)
        if r is None:
            cells.append("-".rjust(8))
            continue
        mark = "*" if r["certified"] == "true" else ""
        und = "U" if int(r["undecidable"]) > 0 else ""
        cells.append((r["nullity"] + mark + und).rjust(8))
    print("%-22s %-13s %-8s %-6s %s" % (case, solver, T, truth, " ".join(cells)))
