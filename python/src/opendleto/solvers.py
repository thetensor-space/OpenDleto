"""Dense exact and matrix-free partial derivation nullspace diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import null_space
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from .derivations import DerivationMap


@dataclass(frozen=True)
class DerivationResult:
    operators: tuple[tuple[np.ndarray, ...], ...]
    coordinates: np.ndarray
    residuals: np.ndarray
    singular_values: np.ndarray
    gaps: np.ndarray
    tensor_dimensions: tuple[int, ...]
    unfolding_ranks: tuple[int, ...]
    operator_space_dimension: int
    trivial_scalar_dimension: int
    total_nullity: int | None
    nontrivial_nullity_estimate: int
    nullity_lower_bound: int
    exact: bool
    converged: bool
    tolerance_sensitivity: dict[float, int]
    backend: str
    dtype: str
    device: str
    message: str
    map: DerivationMap


def _scalar_gauge(dmap: DerivationMap):
    s = dmap.operators.scalar_vectors()
    if s.shape[1] == 0:
        return np.empty((dmap.shape[1], 0))
    # Each column of s realizes identity on one permitted axis.
    permitted = []
    for j in range(s.shape[1]):
        mats = dmap.operators.embed(s[:, j])
        permitted.append([np.trace(m) / len(m) for m in mats])
    p = dmap.chisel @ np.asarray(permitted).T
    kernel = null_space(p)
    g = s @ kernel
    return np.linalg.qr(g)[0][:, :kernel.shape[1]] if kernel.size else np.empty((dmap.shape[1], 0))


def derivations(tensor, chisel="universal", *, operators=None, tol=1e-8,
                tolerances: Sequence[float] | None = None, max_vectors=8,
                dense_max_entries=2_000_000, maxiter=1000, backend="auto"):
    """Find non-scalar derivations and report exact or partial nullity.

    Dense SVD is used only when ``prod(output_shape) * parameter_dim`` is at
    most ``dense_max_entries``. Larger problems use matrix-free Ritz pairs;
    their nullity is a lower bound, never a claim of complete rank.
    """
    dmap = DerivationMap(tensor, chisel, operators, backend)
    if dmap.backend == "torch" and dmap.tensor.device.type != "cpu":
        raise RuntimeError("The SciPy solver requires CPU; move the tensor explicitly")
    if max_vectors < 0 or maxiter < 1 or dense_max_entries < 0:
        raise ValueError("max_vectors, maxiter and dense_max_entries must be nonnegative; maxiter must be positive")
    n = dmap.shape[1]
    if n == 0:
        raise ValueError("operator space has dimension zero")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    ts = tuple(float(t) for t in (tolerances if tolerances is not None else (tol,)))
    if any(t < 0 or not np.isfinite(t) for t in ts):
        raise ValueError("tolerances must be finite and nonnegative")
    gauge = _scalar_gauge(dmap)
    ng = gauge.shape[1]
    if n * max(n, dmap.shape[0]) <= dense_max_entries:
        eye = np.eye(n, dtype=dmap.dtype)
        matrix = np.column_stack([np.asarray(dmap.forward(eye[:, j])).ravel() for j in range(n)])
        _, singular, vh = np.linalg.svd(matrix, full_matrices=(dmap.shape[0] < n))
        full_s = np.zeros(n)
        full_s[:len(singular)] = singular
        threshold = tol * (full_s[0] if len(full_s) else 0)
        mask = full_s <= threshold
        null_basis = vh.conj().T[:, mask]
        # Gauge is a subspace of the exact kernel. Remove it by projection.
        nontriv = null_basis - gauge @ (gauge.conj().T @ null_basis)
        nontriv = np.linalg.svd(nontriv, full_matrices=False)[0][:, :max(0, int(mask.sum()) - ng)] if nontriv.size else nontriv
        coords = nontriv[:, :max_vectors]
        count = max(0, int(mask.sum()) - ng)
        sensitivity = {t: max(0, int(np.sum(full_s <= t * full_s[0])) - ng) for t in ts}
        values = np.sort(full_s)
        exact = True
        total = int(mask.sum())
        converged = True
        message = "Complete dense SVD"
    else:
        op = dmap.linear_operator()
        normal = op.H @ op
        k = min(n - 1, max(ng + max_vectors + 2, 2))
        if k < 1:
            raise ValueError("iterative solve requires at least two parameters")
        try:
            eigvals, eigvecs = eigsh(normal, k=k, which="SM", tol=max(tol / 10, 1e-12), maxiter=maxiter)
            converged = True
            message = f"Partial matrix-free spectrum: {k} of {n} Ritz pairs"
        except ArpackNoConvergence as exc:
            eigvals = exc.eigenvalues
            eigvecs = exc.eigenvectors
            converged = False
            message = f"ARPACK converged {len(eigvals)} of {k} requested Ritz pairs"
        order = np.argsort(eigvals)
        values = np.sqrt(np.maximum(np.asarray(eigvals)[order], 0))
        eigvecs = np.asarray(eigvecs)[:, order]
        scale = np.linalg.norm(np.asarray(dmap.forward(np.ones(n) / np.sqrt(n))))
        scale = max(scale, np.finfo(float).tiny)
        near = values <= tol * scale
        basis = eigvecs[:, near]
        basis = basis - gauge @ (gauge.conj().T @ basis)
        if basis.size:
            u, s, _ = np.linalg.svd(basis, full_matrices=False)
            basis = u[:, s > 1e-7]
        candidates = basis[:, :max_vectors]
        candidate_residuals = np.asarray([np.linalg.norm(np.asarray(dmap.forward(candidates[:, j]))) for j in range(candidates.shape[1])])
        good = candidate_residuals <= tol * scale
        coords = candidates[:, good]
        count = int(good.sum())
        sensitivity = {t: max(0, int(np.sum(values <= t * scale)) - ng) for t in ts}
        exact = False
        total = None
    residuals = np.asarray([np.linalg.norm(np.asarray(dmap.forward(coords[:, j]))) for j in range(coords.shape[1])])
    unpacked = tuple(dmap.operators.embed(coords[:, j]) for j in range(coords.shape[1]))
    if dmap.backend == "numpy":
        ranks = tuple(int(np.linalg.matrix_rank(np.moveaxis(dmap.tensor, a, 0).reshape(dim, -1))) for a, dim in enumerate(dmap.dims))
    else:
        ranks = tuple(int(dmap.xp.linalg.matrix_rank(dmap.xp.movedim(dmap.tensor, a, 0).reshape(dim, -1)).item()) for a, dim in enumerate(dmap.dims))
    gaps = np.diff(values)
    return DerivationResult(unpacked, coords, residuals, values, gaps, dmap.dims,
                            ranks, n, ng, total, count, count, exact, converged,
                            sensitivity, dmap.backend, str(dmap.tensor.dtype),
                            dmap.device, message, dmap)


def derivation_nullity(*args, **kwargs):
    return derivations(*args, **kwargs)
