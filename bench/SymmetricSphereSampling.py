#!/usr/bin/env python3
"""Exact rational rank of the literal SphereLab shell's diagonal chisel.
No numerical solver or third-party dependencies. Nonzero Gaussian coefficients
cancel from each diagonal-derivation equation u_i + v_j + w_k = 0.
"""
from fractions import Fraction


def audit(d, cutoff=1.5):
    radius = d - 2
    points = [(i, j, k) for i in range(d) for j in range(d) for k in range(d)
              if abs(i*i + j*j + k*k - radius*radius) < cutoff]
    active = [sorted({p[a] for p in points}) for a in range(3)]
    columns = [{v: sum(map(len, active[:a])) + i for i, v in enumerate(active[a])}
               for a in range(3)]
    basis = {}
    for p in points:
        row = {columns[a][p[a]]: Fraction(1) for a in range(3)}
        while row:
            pivot = min(row)
            value = row[pivot]
            if pivot not in basis:
                basis[pivot] = {j: x/value for j, x in row.items()}
                break
            for j, x in basis[pivot].items():
                row[j] = row.get(j, 0) - value*x
                if not row[j]:
                    del row[j]
    return (d, radius, len(points), len(active[0]), len(basis),
            sum(map(len, active)) - len(basis))


if __name__ == '__main__':
    print('dim,radius,points,active_slices_per_axis,diagonal_chisel_rank,diagonal_nullity')
    for dimension in (20, 50, 52):
        print(','.join(map(str, audit(dimension))))
