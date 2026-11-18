"""Coordinate frames and algebraic support revealed by a derivation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import schur

from .solvers import DerivationResult
from .tensors import TensorElement


@dataclass(frozen=True)
class StratificationResult:
    transformed_tensor: np.ndarray
    frames: tuple[np.ndarray, ...]
    operators: tuple[np.ndarray, ...]
    eigenvalues: tuple[np.ndarray, ...] | None
    allowed_support: np.ndarray | None
    forbidden_support: np.ndarray | None
    support_count: int | None
    allowed_energy: float | None
    forbidden_energy: float | None
    grading_residual: float
    reconstruction_error: float | None
    method: str
    operator_entries: int
    frame_entries: int
    support_description_entries: int | None
    retained_coefficients: int | None
    estimated_storage_entries: int | None


def stratify(tensor, result: DerivationResult, *, index=0, tol=1e-8,
             max_condition=1e8) -> StratificationResult:
    """Reveal a coordinate null pattern from one discovered derivation.

    A coordinatewise mask is returned only when every generator is safely
    diagonalizable. Schur frames are returned otherwise, without claiming
    a coefficientwise algebraic null pattern.
    """
    if result.backend != "numpy":
        raise TypeError("stratify currently requires NumPy arrays; convert explicitly")
    array = np.asarray(tensor.data if isinstance(tensor, TensorElement) else tensor)
    if not 0 <= index < len(result.operators):
        raise IndexError("derivation index out of range")
    mats = result.operators[index]
    if array.shape != result.tensor_dimensions:
        raise ValueError("tensor shape does not match derivation result")
    frames = []
    eigs = []
    safe = True
    for m in mats:
        values, vectors = np.linalg.eig(m.T)
        if np.linalg.cond(vectors) > max_condition:
            safe = False
            break
        frames.append(vectors)
        eigs.append(values)
    if not safe:
        frames = [schur(m.T, output="complex")[1] for m in mats]
        eigs = []
    transformed = array.astype(np.result_type(array.dtype, complex), copy=True)
    for axis, frame in enumerate(frames):
        transformed = np.moveaxis(np.tensordot(np.linalg.inv(frame), transformed, axes=(1, axis)), 0, axis)
    residual = float(np.linalg.norm(np.asarray(result.map.forward(result.coordinates[:, index]))))
    operator_entries = sum(m.size for m in mats)
    frame_entries = sum(f.size for f in frames)
    if safe:
        grading = np.zeros((result.map.chisel.shape[0], *array.shape), dtype=complex)
        for axis, values in enumerate(eigs):
            shape = [1] * array.ndim
            shape[axis] = len(values)
            for row in range(result.map.chisel.shape[0]):
                grading[row] += result.map.chisel[row, axis] * values.reshape(shape)
        scale = max(1, max(float(np.max(np.abs(v))) for v in eigs))
        allowed = np.all(np.abs(grading) <= tol * scale, axis=0)
        forbidden = ~allowed
        energy = np.abs(transformed) ** 2
        allowed_energy = float(energy[allowed].sum())
        forbidden_energy = float(energy[forbidden].sum())
        reconstruction_error = float(np.sqrt(forbidden_energy / max(float(energy.sum()), np.finfo(float).tiny)))
        return StratificationResult(transformed, tuple(frames), tuple(mats),
                                    tuple(eigs), allowed, forbidden, int(allowed.sum()),
                                    allowed_energy, forbidden_energy, residual,
                                    reconstruction_error, "diagonal", operator_entries,
                                    frame_entries, sum(len(v) for v in eigs) + result.map.chisel.size,
                                    int(allowed.sum()), operator_entries + frame_entries +
                                    sum(len(v) for v in eigs) + result.map.chisel.size + int(allowed.sum()))
    return StratificationResult(transformed, tuple(frames), tuple(mats), None,
                                None, None, None, None, None, residual, None,
                                "schur", operator_entries, frame_entries, None, None, None)
