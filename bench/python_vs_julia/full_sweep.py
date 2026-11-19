"""Time complete Python derivation stages on a shared Fortran-order tensor.

Example: python full_sweep.py /tmp/inputs --case sample --shape 4 4 4
Input is /tmp/inputs/case_sample.bin, little-endian float64 in Fortran order.
Each stdout line is a JSON object; timings exclude one warmup per stage.
"""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from scipy.linalg import svdvals
from opendleto import (AdjointChisel, DerivationMap, EmptyOp, TransverseOps,
                       UniversalOp, derivations, nondeg, stratify)


def timed(function, repeats):
    function()
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        result = function()
        samples.append(perf_counter() - start)
    return result, median(samples)


def operator_space(shape, chisel_name):
    if chisel_name == "adjoint":
        return TransverseOps(shape, [UniversalOp() if axis < 2 else EmptyOp()
                                     for axis in range(len(shape))])
    return TransverseOps(shape, UniversalOp())


def dense_dimension(array, tol, chisel, operators):
    dmap = DerivationMap(array, chisel, operators)
    eye = np.eye(dmap.shape[1])
    matrix = np.column_stack([dmap.forward(eye[:, j]).ravel() for j in range(dmap.shape[1])])
    values = svdvals(matrix)
    threshold = tol * values[0] if len(values) else 0
    return dmap.shape[1] - int(np.count_nonzero(values > threshold))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--case', required=True, help='input name in case_<name>.bin')
    parser.add_argument('--shape', type=int, nargs='+', required=True)
    parser.add_argument('--chisel', choices=('universal', 'adjoint'), default='universal',
                        help='adjoint couples axes 0 and 1')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--nondeg-tol', type=float, default=1e-10)
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 1 for n in args.shape):
        parser.error('repeats and all shape dimensions must be positive')
    array = np.fromfile(args.folder / f'case_{args.case}.bin', dtype='<f8')
    if array.size != np.prod(args.shape):
        parser.error('input file size does not match shape')
    array = array.reshape(args.shape, order='F')
    if args.chisel == 'adjoint' and array.ndim < 2:
        parser.error('adjoint chisel requires at least two axes')
    chisel = 'universal' if args.chisel == 'universal' else AdjointChisel(array.ndim, 0, 1)
    raw_shape = list(array.shape)
    reduced_shape = None

    def emit(method, stage, seconds, dimension=None, residual=None, status='ok', **extra):
        print(json.dumps(dict(case=args.case, chisel=args.chisel, method=method, stage=stage, seconds=seconds,
                              dimension=dimension, residual=residual, status=status,
                              raw_shape=raw_shape, reduced_shape=reduced_shape,
                              operator_space="effective", **extra), allow_nan=False), flush=True)

    reduction, seconds = timed(lambda: nondeg(array, tol=args.nondeg_tol), args.repeats)
    reduced_shape = list(reduction.core.shape)
    residual = float(np.linalg.norm(reduction.reconstruct() - array) / max(np.linalg.norm(array), np.finfo(float).tiny))
    emit('hosvd', 'nondeg', seconds, residual=residual, ranks=list(reduction.ranks))
    for label, value in [('raw', array), ('reduced', reduction.core)]:
        if any(n == 0 for n in value.shape):
            emit('dense_svdvals', label, 0.0, status='skipped: zero-dimensional core')
            continue
        dimension, seconds = timed(lambda: dense_dimension(value, args.tol, chisel, operator_space(value.shape, args.chisel)), args.repeats)
        emit('dense_svdvals', label, seconds, dimension=dimension)
    if any(n == 0 for n in reduction.core.shape):
        emit('derivations', 'reduced_solve', 0.0, status='skipped: zero-dimensional core')
        emit('stratify', 'reduced_stratification', 0.0, status='skipped: zero-dimensional core')
        return
    operators = operator_space(reduction.core.shape, args.chisel)
    dmap = DerivationMap(reduction.core, chisel, operators)
    # Force the complete dense path and retain every possible non-scalar vector.
    limit = dmap.shape[1] * max(dmap.shape)
    result, seconds = timed(lambda: derivations(reduction.core, chisel=chisel, operators=operators, tol=args.tol,
                                dense_max_entries=limit, max_vectors=dmap.shape[1]), args.repeats)
    emit('derivations', 'reduced_solve', seconds, dimension=result.total_nullity,
         residual=float(np.max(result.residuals)) if result.residuals.size else None,
         status='exact' if result.exact else 'partial', nontrivial_dimension=result.nontrivial_nullity_estimate,
         scalar_dimension=result.trivial_scalar_dimension, operator_space_dimension=result.operator_space_dimension,
         returned_vectors=len(result.operators),
         converged=result.converged)
    if not result.operators:
        emit('stratify', 'reduced_stratification', 0.0, status='skipped: no non-scalar derivation')
        return
    stratification, seconds = timed(lambda: stratify(reduction.core, result), args.repeats)
    emit('stratify', 'reduced_stratification', seconds, dimension=stratification.support_count,
         residual=stratification.grading_residual, status=stratification.method,
         reconstruction_error=stratification.reconstruction_error)


if __name__ == '__main__':
    main()
