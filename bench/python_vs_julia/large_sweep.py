"""Benchmark large shared cubic tensors; emit one JSON object per stage.

Inputs are case_<size>.bin, little-endian float64 in Fortran order. Each
size runs in a subprocess so --timeout bounds even a stuck native solver.
Dense SVD is enabled only if its conservative workspace estimate fits
--dense-memory-mib; otherwise derivations uses a bounded partial Ritz solve.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np


def emit(**fields):
    print(json.dumps(fields, allow_nan=False), flush=True)


def worker(args, n):
    from opendleto import (AdjointChisel, DerivationMap, EmptyOp, TransverseOps,
                            UniversalOp, derivations, nondeg, stratify)
    from scipy.linalg import svdvals

    common = dict(language='python', size=n, case=str(n), chisel=args.chisel)
    path = args.folder / f'case_{n}.bin'
    start = perf_counter()
    array = np.fromfile(path, dtype='<f8')
    if array.size != n ** 3:
        raise ValueError(f'{path}: expected {n ** 3} float64 values, got {array.size}')
    array = array.reshape((n, n, n), order='F')
    emit(**common, stage='load', status='ok', seconds=perf_counter()-start)
    start = perf_counter()
    reduction = nondeg(array, tol=args.nondeg_tol)
    elapsed = perf_counter()-start
    residual = float(np.linalg.norm(reduction.reconstruct()-array) / max(np.linalg.norm(array), np.finfo(float).tiny))
    core = reduction.core
    emit(**common, stage='nondeg', status='ok', seconds=elapsed,
         residual=residual, reduced_shape=list(core.shape), ranks=list(reduction.ranks))
    if any(d == 0 for d in core.shape):
        emit(**common, stage='derivations', status='skipped_zero_core', seconds=None)
        return
    chisel = 'universal' if args.chisel == 'universal' else AdjointChisel(3, 0, 1)
    operators = (None if args.chisel == 'universal' else
                 TransverseOps(core.shape, [UniversalOp(), UniversalOp(), EmptyOp()]))
    dmap = DerivationMap(core, chisel, operators)
    rows, cols = dmap.shape
    # Matrix construction and SVD require multiple dense matrices, plus V and
    # the eye. This is a conservative preflight, not a process RSS guarantee.
    estimate = 8 * (4 * rows * cols + 4 * cols * cols)
    dense = estimate <= args.dense_memory_mib * 1024**2
    limit = cols * max(rows, cols) if dense else 0
    max_vectors = cols if dense else args.max_vectors
    emit(**common, stage='preflight', status='dense' if dense else 'partial', seconds=None,
         map_shape=[rows, cols], dense_estimated_bytes=estimate,
         dense_memory_limit_bytes=int(args.dense_memory_mib * 1024**2),
         max_vectors=max_vectors, maxiter=args.maxiter)
    if dense:
        def dense_values():
            dense_map = DerivationMap(core, chisel, operators)
            eye = np.eye(cols, dtype=core.dtype)
            matrix = np.column_stack([dense_map.forward(eye[:, j]).ravel() for j in range(cols)])
            return svdvals(matrix)
        if args.warm_target:
            dense_values()
        start = perf_counter()
        singular = dense_values()
        dimension = cols - int(np.count_nonzero(singular > args.tol * singular[0])) if singular.size else cols
        emit(**common, stage='svdvals', status='exact', seconds=perf_counter()-start,
             dimension=dimension, dense_estimated_bytes=estimate, target_warmed=args.warm_target)
        del singular
    else:
        emit(**common, stage='svdvals', status='skipped_memory', seconds=None,
             dense_estimated_bytes=estimate)
    if args.dimension_only:
        return
    def solve():
        return derivations(core, chisel=chisel, operators=operators, tol=args.tol,
                           dense_max_entries=limit, max_vectors=max_vectors, maxiter=args.maxiter)
    if args.warm_target:
        solve()
    start = perf_counter()
    result = solve()
    elapsed = perf_counter()-start
    complete = (result.exact and result.total_nullity is not None and
                len(result.operators) == result.total_nullity - result.trivial_scalar_dimension)
    emit(**common, stage='derivations', status='exact' if complete else 'partial',
         seconds=elapsed, dimension=result.total_nullity,
         dimension_lower_bound=result.nullity_lower_bound + result.trivial_scalar_dimension,
         scalar_dimension=result.trivial_scalar_dimension,
         nontrivial_dimension=result.nontrivial_nullity_estimate,
         returned_vectors=len(result.operators), converged=result.converged,
         residual=float(np.max(result.residuals)) if result.residuals.size else None,
         message=result.message, basis_complete=complete, target_warmed=args.warm_target)
    if result.operators:
        if args.warm_target:
            stratify(core, result)
        start = perf_counter()
        decomposition = stratify(core, result)
        emit(**common, stage='stratify',
             status=decomposition.method if complete else 'partial_basis_' + decomposition.method,
             basis_complete=complete, target_warmed=args.warm_target,
             seconds=perf_counter()-start, dimension=decomposition.support_count,
             residual=decomposition.grading_residual,
             reconstruction_error=decomposition.reconstruction_error)
    else:
        emit(**common, stage='stratify', status='skipped_empty_basis', seconds=None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('--warm-target', action='store_true', help='warm solver and dense SVD on target core')
    p.add_argument('--dimension-only', action='store_true', help='stop after the svdvals stage')
    p.add_argument('--chisel', choices=('universal', 'adjoint'), default='universal')
    p.add_argument('--sizes', type=int, nargs='+', default=[20, 40, 60, 80, 100])
    p.add_argument('--timeout', type=float, default=300, help='wall seconds per size')
    p.add_argument('--dense-memory-mib', type=float, default=1024,
                   help='maximum estimated dense workspace; 0 disables dense')
    p.add_argument('--max-vectors', type=int, default=8, help='partial solver vector cap')
    p.add_argument('--maxiter', type=int, default=1000)
    p.add_argument('--tol', type=float, default=1e-8)
    p.add_argument('--nondeg-tol', type=float, default=1e-10)
    p.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    if (any(n < 1 for n in args.sizes) or args.timeout <= 0 or
            args.dense_memory_mib < 0 or args.max_vectors < 0 or args.maxiter < 1 or
            not np.isfinite([args.timeout, args.dense_memory_mib, args.tol, args.nondeg_tol]).all() or
            args.tol < 0 or args.nondeg_tol < 0):
        p.error('sizes, timeout and maxiter must be positive; limits and tolerances must be finite and nonnegative')
    if args.worker is not None:
        worker(args, args.worker)
        return
    for n in args.sizes:
        command = [sys.executable, __file__, str(args.folder), '--worker', str(n),
                   '--chisel', args.chisel, '--timeout', str(args.timeout), '--dense-memory-mib', str(args.dense_memory_mib),
                   '--max-vectors', str(args.max_vectors), '--maxiter', str(args.maxiter),
                   '--tol', str(args.tol), '--nondeg-tol', str(args.nondeg_tol)]
        if args.warm_target:
            command.append('--warm-target')
        if args.dimension_only:
            command.append('--dimension-only')
        start = perf_counter()
        proc = subprocess.Popen(command, stdout=sys.stdout, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)
        try:
            _, stderr = proc.communicate(timeout=args.timeout)
            status = 'ok' if proc.returncode == 0 else 'error'
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
            _, stderr = proc.communicate()
            status = 'timeout'
        emit(language='python', size=n, case=str(n), chisel=args.chisel, stage='process', status=status,
             seconds=perf_counter()-start, returncode=proc.returncode,
             error=stderr[-4000:] if stderr else None)


if __name__ == '__main__':
    main()
