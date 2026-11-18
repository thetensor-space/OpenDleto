"""Matrix-free chisel derivation maps, following Julia ``sylvesterLM``."""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .backends import tensor_backend
from .chisels import UniversalChisel
from .operators import TransverseOps, UniversalOp
from .tensors import TensorElement


class DerivationMap:
    """Map operator coordinates to chisel equations and expose its adjoint.

    The complex inner product uses ``vdot(x, y)`` (conjugate linear in x).
    Coordinate matrices follow Julia ``sylvesterLM``: their transpose acts
    on the output index of each tensor mode.
    """

    def __init__(self, tensor, chisel=None, operators=None, backend="auto"):
        if isinstance(tensor, TensorElement):
            tensor = tensor.data
        self.backend = tensor_backend(tensor, backend)
        self.tensor = tensor
        self.dims = tuple(tensor.shape)
        if not self.dims or any(n < 1 for n in self.dims):
            raise ValueError("tensor must have positive dimensions")
        if isinstance(chisel, str):
            if chisel.lower() != "universal":
                raise ValueError("Unknown chisel name")
            chisel = None
        self.chisel = np.asarray(UniversalChisel(len(self.dims)) if chisel is None else chisel)
        if self.chisel.ndim != 2 or self.chisel.shape[1] != len(self.dims):
            raise ValueError("chisel must have one column per tensor axis")
        if not np.all(np.isfinite(self.chisel)):
            raise ValueError("chisel coefficients must be finite")
        if np.iscomplexobj(self.chisel) and not (np.issubdtype(tensor.dtype, np.complexfloating) if self.backend == "numpy" else tensor.is_complex()):
            raise TypeError("complex chisels require a complex tensor dtype")
        self.operators = operators or TransverseOps(self.dims, UniversalOp())
        if tuple(self.operators.dims) != self.dims:
            raise ValueError("operator dimensions do not match tensor")
        self.shape = (int(self.chisel.shape[0] * np.prod(self.dims)), self.operators.dimension)
        if self.backend == "torch":
            import torch
            self.xp = torch
            self._chisel = torch.as_tensor(self.chisel, dtype=tensor.dtype, device=tensor.device)
            self.dtype = np.dtype("complex128" if tensor.is_complex() else "float64")
            self.device = str(tensor.device)
        else:
            self.xp = np
            self._chisel = self.chisel.astype(tensor.dtype, copy=False)
            self.dtype = tensor.dtype
            self.device = "cpu"
        self._unfoldings = [self._moveaxis(tensor, a, 0).reshape(n, -1) for a, n in enumerate(self.dims)]

    def _moveaxis(self, x, source, destination):
        return self.xp.movedim(x, source, destination) if self.backend == "torch" else np.moveaxis(x, source, destination)

    def _as_native(self, x):
        if self.backend == "torch":
            return self.xp.as_tensor(x, dtype=self.tensor.dtype, device=self.tensor.device)
        return np.asarray(x, dtype=self.tensor.dtype)

    def forward(self, coordinates):
        """Apply D_T, returning shape (chisel rows, *tensor dimensions)."""
        matrices = self.operators.embed(np.asarray(coordinates))
        out = self.xp.zeros((self.chisel.shape[0], *self.dims), dtype=self.tensor.dtype,
                            **({"device": self.tensor.device} if self.backend == "torch" else {}))
        for axis, matrix in enumerate(matrices):
            if not np.any(self.chisel[:, axis]):
                continue
            m = self._as_native(matrix)
            acted = self._moveaxis((m.T @ self._unfoldings[axis]).reshape((self.dims[axis],) + self.dims[:axis] + self.dims[axis + 1:]), 0, axis)
            weights = self._chisel[:, axis].reshape((-1,) + (1,) * len(self.dims))
            out = out + weights * acted
        return out

    def adjoint(self, output):
        """Apply D_T* to a chisel-output tensor."""
        y = self._as_native(output).reshape((self.chisel.shape[0], *self.dims))
        matrices = []
        for axis, n in enumerate(self.dims):
            m = self.xp.zeros((n, n), dtype=self.tensor.dtype,
                              **({"device": self.tensor.device} if self.backend == "torch" else {}))
            tstar = self._unfoldings[axis].conj().T
            for row in range(self.chisel.shape[0]):
                if self.chisel[row, axis] != 0:
                    yu = self._moveaxis(y[row], axis, 0).reshape(n, -1)
                    m = m + self._chisel[row, axis].conj() * (yu @ tstar)
            if self.backend == "torch":
                if m.device.type != "cpu":
                    raise RuntimeError("Torch GPU adjoint coordinates require explicit CPU transfer; use forward on GPU or move tensor to CPU")
                m = m.detach().numpy()
            matrices.append(m.T)
        return self.operators.adjoint(matrices)

    def normal(self, coordinates):
        return self.adjoint(self.forward(coordinates))

    def linear_operator(self):
        """SciPy matrix-free LinearOperator (NumPy or CPU PyTorch only)."""
        if self.backend == "torch" and self.tensor.device.type != "cpu":
            raise RuntimeError("SciPy eigensolvers require CPU; transfer tensor explicitly")
        def matvec(x):
            y = self.forward(x)
            return np.asarray(y).reshape(-1)
        def rmatvec(y):
            return np.asarray(self.adjoint(y))
        return LinearOperator(self.shape, matvec=matvec, rmatvec=rmatvec, dtype=self.dtype)


def sylvesterLM(tensor, chisel=None, operators=None, backend="auto"):
    """Return the matrix-free derivation map and normal operator."""
    derivation = DerivationMap(tensor, chisel, operators, backend)
    d = derivation.linear_operator()
    normal = LinearOperator((d.shape[1], d.shape[1]), matvec=lambda x: derivation.normal(x),
                            dtype=d.dtype)
    return normal, d
