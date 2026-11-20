"""Notebook-facing tensor operations for the Chiseling 101 experiments."""

from dataclasses import dataclass

import numpy as np

from .solvers import derivations
from .stratification import stratify
from .operators import Operator, TransverseOps


def apply_transforms(value, transforms):
    """Apply a (new, old) coordinate matrix to each tensor axis."""
    result = np.asarray(value)
    if len(transforms) != result.ndim:
        raise ValueError("provide one transform per tensor axis")
    for axis, matrix in enumerate(transforms):
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[1] != result.shape[axis]:
            raise ValueError(f"transform {axis} has incompatible shape")
        result = np.moveaxis(np.tensordot(matrix, result, axes=(1, axis)), 0, axis)
    return result


def direct_sum(*values):
    """Place tensors on disjoint diagonal blocks along every axis."""
    if not values:
        raise ValueError("provide at least one tensor")
    arrays = [np.asarray(value) for value in values]
    degree = arrays[0].ndim
    if degree == 0 or any(a.ndim != degree for a in arrays):
        raise ValueError("all tensors must have the same positive number of axes")
    shape = tuple(sum(a.shape[axis] for a in arrays) for axis in range(degree))
    out = np.zeros(shape, dtype=np.result_type(*arrays))
    offsets = [0] * degree
    for array in arrays:
        sections = tuple(slice(start, start + size) for start, size in zip(offsets, array.shape))
        out[sections] = array
        offsets = [start + size for start, size in zip(offsets, array.shape)]
    return out


def randomize_tensor(value, *, rng=None, kind="invertible"):
    """Hide tensor coordinates with random invertible or orthogonal frames."""
    rng = np.random.default_rng() if rng is None else rng
    array = np.asarray(value)
    matrices = []
    for size in array.shape:
        if kind == "orthogonal":
            matrix, _ = np.linalg.qr(rng.normal(size=(size, size)))
        elif kind == "invertible":
            # QR supplies a well-conditioned invertible frame. Scaling retains
            # the non-orthogonal changes used by the Julia demonstration.
            q, _ = np.linalg.qr(rng.normal(size=(size, size)))
            scales = rng.uniform(0.7, 1.3, size)
            matrix = q @ np.diag(scales)
        else:
            raise ValueError("kind must be 'invertible' or 'orthogonal'")
        matrices.append(matrix)
    matrices = tuple(matrices)
    return apply_transforms(array, matrices), matrices


@dataclass(frozen=True)
class ChiselingResult:
    tensor: np.ndarray
    transforms: tuple[np.ndarray, ...]
    derivation_count: int
    exact: bool
    reconstruction_error: float
    relative_law_error: float | None = None
    forbidden_energy_fraction: float | None = None
    support_count: int | None = None
    frame_condition: float | None = None
    solver_message: str = ""
    trivial_scalar_dimension: int = 0

    @property
    def total_derivation_count(self):
        """Discovered nontrivial directions plus scalar gauge directions."""
        return self.derivation_count + self.trivial_scalar_dimension


def chisel_tensor(value, *, operators=None, chisel="universal", tol=1e-5,
                  max_vectors=12, maxiter=10000):
    """Find derivations and reveal one coordinate sparsity pattern.

    A generic control has no nonscalar derivations; its identity frame is
    returned. The reported count is a lower bound when ``exact`` is false.
    """
    array = np.asarray(value)
    if isinstance(operators, Operator):
        operators = TransverseOps(array.shape, operators)
    found = derivations(array, chisel=chisel, operators=operators,
                        tol=tol, max_vectors=max_vectors,
                        maxiter=maxiter)
    identities = tuple(np.eye(size) for size in array.shape)
    if not found.operators:
        return ChiselingResult(array.copy(), identities,
                               found.nontrivial_nullity_estimate, found.exact, 0.0,
                               solver_message=found.message,
                               trivial_scalar_dimension=found.trivial_scalar_dimension)
    candidates = [stratify(array, found, index=i, tol=tol)
                  for i in range(len(found.operators))]
    # Prefer the law producing the sparsest numerically accurate pattern.
    best = min(candidates, key=lambda candidate: (
        candidate.support_count if candidate.support_count is not None else array.size,
        candidate.reconstruction_error if candidate.reconstruction_error is not None else float("inf"),
        max(np.linalg.cond(frame) for frame in candidate.frames)))
    transforms = tuple(np.linalg.inv(frame) for frame in best.frames)
    transformed = apply_transforms(array, transforms)
    recovered = apply_transforms(transformed, best.frames)
    error = np.linalg.norm(recovered - array) / max(np.linalg.norm(array), 1e-15)
    relative_law_error = best.grading_residual / max(
        np.linalg.norm(array) * sum(np.linalg.norm(m) for m in best.operators),
        np.finfo(float).tiny)
    energy_fraction = (best.forbidden_energy / max(best.allowed_energy + best.forbidden_energy,
                                                  np.finfo(float).tiny)
                       if best.forbidden_energy is not None else None)
    return ChiselingResult(transformed, transforms,
                           found.nontrivial_nullity_estimate, found.exact, float(error),
                           float(relative_law_error), energy_fraction, best.support_count,
                           float(max(np.linalg.cond(frame) for frame in best.frames)),
                           found.message, found.trivial_scalar_dimension)
