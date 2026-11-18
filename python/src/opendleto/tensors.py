"""Dense tensor reduction using NumPy higher-order SVD."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TensorElement:
    """A tensor value backed by a native NumPy array or PyTorch tensor.

    This is the Python counterpart of the Julia TensorSpace wrapper. It keeps
    the native tensor accessible as ``data`` and never imports ITensors.
    """

    data: object

    def __post_init__(self):
        from .backends import tensor_backend
        tensor_backend(self.data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __array__(self, dtype=None):
        if type(self.data).__module__.startswith("torch"):
            raise TypeError("Convert a PyTorch TensorElement explicitly with .data.detach().cpu().numpy()")
        return np.asarray(self.data, dtype=dtype)


def tensor(value):
    """Wrap a native tensor as a TensorElement."""
    return value if isinstance(value, TensorElement) else TensorElement(value)


def _mode_product(tensor, matrix, axis):
    """Multiply a tensor along one axis by a matrix of shape (new, old)."""
    return np.moveaxis(np.tensordot(matrix, tensor, axes=(1, axis)), 0, axis)


@dataclass(frozen=True)
class NondegenerateResult:
    """HOSVD core and input-space bases, ordered as the requested axes."""

    core: np.ndarray
    bases: tuple[np.ndarray, ...]
    axes: tuple[int, ...]

    @property
    def ranks(self):
        return tuple(basis.shape[1] for basis in self.bases)

    @property
    def Δ(self):
        return self.core

    @property
    def Es(self):
        return self.bases

    def reconstruct(self):
        """Map the core back into the original tensor space."""
        result = self.core
        for axis, basis in zip(self.axes, self.bases):
            result = _mode_product(result, basis, axis)
        return result


def nondeg(tensor, axes=None, *, mode="trunc", tol=1e-10):
    """Compute an HOSVD or a truncated nondegenerate tensor reduction.

    Axes are zero-based and default to all axes. Bases have orthonormal
    columns and correspond to axes in the supplied order. In ``full`` mode,
    each basis spans its entire input axis. In ``trunc`` mode, retain singular
    values strictly greater than ``tol * largest_singular_value`` for each
    original mode unfolding. A zero unfolding therefore has rank zero. The
    tolerance is relative to each mode, not an absolute singular-value cutoff.
    The core is formed by applying the conjugate transpose of each basis;
    ``result.reconstruct()`` maps it back to the original tensor space.
    """
    array = np.asarray(tensor)
    if array.ndim == 0 or any(size == 0 for size in array.shape):
        raise ValueError("tensor must have at least one nonempty axis")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("tensor must be numeric")
    if not np.all(np.isfinite(array)):
        raise ValueError("tensor must contain only finite values")
    if mode not in ("trunc", "full"):
        raise ValueError("mode must be 'trunc' or 'full'")
    if not np.isscalar(tol) or not np.isreal(tol) or not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be a finite nonnegative real number")
    if axes is None:
        axes = tuple(range(array.ndim))
    else:
        axes = tuple(axes)
        if any(not isinstance(axis, (int, np.integer)) or isinstance(axis, (bool, np.bool_)) for axis in axes):
            raise TypeError("axes must contain integers")
        if any(axis < 0 or axis >= array.ndim for axis in axes):
            raise ValueError("axis out of range")
        if len(set(axes)) != len(axes):
            raise ValueError("axes must be unique")
    bases = []
    for axis in axes:
        unfolding = np.moveaxis(array, axis, 0).reshape(array.shape[axis], -1)
        u, singular_values, _ = np.linalg.svd(unfolding, full_matrices=(mode == "full"))
        if mode == "trunc":
            rank = np.count_nonzero(singular_values > tol * singular_values[0])
            u = u[:, :rank]
        bases.append(u)
    core = array
    for axis, basis in zip(axes, bases):
        core = _mode_product(core, basis.conj().T, axis)
    return NondegenerateResult(core, tuple(bases), axes)
