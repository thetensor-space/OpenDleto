"""Chisel matrices describing coupled axis constraints."""
import numpy as np


def _mask(value):
    if isinstance(value, (int, np.integer)):
        if value < 0:
            raise ValueError('valence must be nonnegative')
        return np.ones(value, dtype=bool)
    mask = np.asarray(value, dtype=bool)
    if mask.ndim != 1:
        raise ValueError('engaged axes must be one-dimensional')
    return mask


def UniversalChisel(axes):
    return _mask(axes).astype(float)[None, :]


def TuckerChisel(axes):
    mask = _mask(axes)
    return np.eye(mask.size)[mask]


def AdjointChisel(valence, left, right):
    if not 0 <= left < valence or not 0 <= right < valence or left == right:
        raise ValueError('axes must be distinct and in range')
    out = np.zeros((1, valence))
    out[0, left], out[0, right] = 1, -1
    return out


def CentroidChisel(axes):
    mask = _mask(axes)
    indices = np.flatnonzero(mask)
    out = np.zeros((len(indices) * (len(indices) - 1) // 2, len(mask)))
    k = 0
    for i, a in enumerate(indices):
        for b in indices[i + 1:]:
            out[k, a], out[k, b] = 1, -1
            k += 1
    return out


def engaged(chisel, cutoff=1e-6):
    matrix = np.asarray(chisel)
    if matrix.ndim != 2:
        raise ValueError('chisel must be a matrix')
    return np.any(np.abs(matrix) > cutoff, axis=0)


def normalize_chisel(chisel, cutoff=1e-6):
    matrix = np.asarray(chisel)
    mask = engaged(matrix, cutoff)
    selector = TuckerChisel(mask)
    _, singular, vh = np.linalg.svd(matrix @ selector.T, full_matrices=False)
    return vh[singular > cutoff] @ selector
