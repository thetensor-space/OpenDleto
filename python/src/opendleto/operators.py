"""Local matrix spaces and independent transverse operator spaces."""
import numpy as np


class Operator:
    """A linear space of square matrices with a fixed coordinate basis."""
    def dimension(self, n):
        raise NotImplementedError

    def embed(self, n, data):
        data = np.asarray(data)
        if n < 0 or data.ndim != 1 or len(data) != self.dimension(n):
            raise ValueError('incompatible operator coordinates')
        return self._embed(n, data)

    def coordinates(self, matrix):
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return None
        data = self._coordinates(matrix)
        if data is None or not np.allclose(self._embed(len(matrix), data), matrix, atol=1e-8, rtol=0):
            return None
        return data

    def transpose_embed(self, matrix):
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError('matrix must be square')
        return self._transpose_embed(matrix)


class UniversalOp(Operator):
    def dimension(self, n): return n * n
    def _embed(self, n, data): return data.reshape((n, n), order='F')
    def _coordinates(self, m): return m.ravel(order='F')
    def _transpose_embed(self, m): return m.ravel(order='F')


class DiagonalOp(Operator):
    def dimension(self, n): return n
    def _embed(self, n, data): return np.diag(data)
    def _coordinates(self, m): return np.diag(m).copy()
    def _transpose_embed(self, m): return np.diag(m).copy()


class SymmetricOp(Operator):
    def dimension(self, n): return n * (n + 1) // 2
    def _embed(self, n, data):
        out = np.zeros((n, n), dtype=data.dtype)
        k = 0
        for j in range(n):
            out[:j + 1, j] = data[k:k + j + 1]
            out[j, :j + 1] = data[k:k + j + 1]
            k += j + 1
        return out
    def _coordinates(self, m): return np.array([m[i, j] for j in range(len(m)) for i in range(j + 1)], dtype=m.dtype)
    def _transpose_embed(self, m): return np.array([m[i, j] + m[j, i] if i != j else m[i, i] for j in range(len(m)) for i in range(j + 1)], dtype=m.dtype)


class AntiSymmetricOp(Operator):
    def dimension(self, n): return n * (n - 1) // 2
    def _embed(self, n, data):
        out = np.zeros((n, n), dtype=data.dtype)
        k = 0
        for j in range(n):
            out[:j, j] = data[k:k + j]
            out[j, :j] = -data[k:k + j]
            k += j
        return out
    def _coordinates(self, m): return np.array([m[i, j] for j in range(len(m)) for i in range(j)], dtype=m.dtype)
    def _transpose_embed(self, m): return np.array([m[i, j] - m[j, i] for j in range(len(m)) for i in range(j)], dtype=m.dtype)


class ScalarOp(Operator):
    def dimension(self, n): return 1
    def _embed(self, n, data): return np.eye(n, dtype=data.dtype) * data[0]
    def _coordinates(self, m): return np.array([m[0, 0]], dtype=m.dtype) if len(m) else np.zeros(1, dtype=m.dtype)
    def _transpose_embed(self, m): return np.array([np.trace(m)], dtype=m.dtype)


class EmptyOp(Operator):
    def dimension(self, n): return 0
    def _embed(self, n, data): return np.zeros((n, n), dtype=data.dtype)
    def _coordinates(self, m): return np.empty(0, dtype=m.dtype)
    def _transpose_embed(self, m): return np.empty(0, dtype=m.dtype)


class TransverseOps:
    """Direct sum of independently chosen matrix spaces on each axis."""
    def __init__(self, dims, operators=None, symmetries=None):
        self.dims = tuple(int(n) for n in dims)
        if any(n < 0 for n in self.dims):
            raise ValueError('dimensions must be nonnegative')
        if symmetries is not None:
            raise NotImplementedError('coupled transverse symmetries are not implemented')
        if operators is None:
            operators = UniversalOp()
        if isinstance(operators, Operator):
            operators = [operators] * len(self.dims)
        self.operators = tuple(operators)
        if len(self.operators) != len(self.dims) or not all(isinstance(op, Operator) for op in self.operators):
            raise ValueError('one operator is required per axis')
        self._sizes = tuple(op.dimension(n) for n, op in zip(self.dims, self.operators))
        self.dimension = sum(self._sizes)

    def embed(self, data):
        data = np.asarray(data)
        if data.ndim != 1 or len(data) != self.dimension:
            raise ValueError('incompatible transverse coordinates')
        offsets = np.cumsum((0,) + self._sizes)
        return tuple(op.embed(n, data[offsets[i]:offsets[i + 1]]) for i, (n, op) in enumerate(zip(self.dims, self.operators)))

    def coordinates(self, matrices):
        matrices = tuple(matrices)
        if len(matrices) != len(self.dims):
            raise ValueError('one matrix is required per axis')
        parts = []
        for n, op, matrix in zip(self.dims, self.operators, matrices):
            if np.shape(matrix) != (n, n):
                raise ValueError('incompatible matrix shape')
            part = op.coordinates(matrix)
            if part is None:
                return None
            parts.append(part)
        return np.concatenate(parts) if parts else np.empty(0)

    def adjoint(self, matrices):
        matrices = tuple(matrices)
        if len(matrices) != len(self.dims) or any(np.shape(m) != (n, n) for n, m in zip(self.dims, matrices)):
            raise ValueError('incompatible transverse matrices')
        parts = [op.transpose_embed(m) for op, m in zip(self.operators, matrices)]
        return np.concatenate(parts) if parts else np.empty(0)

    def scalar_vectors(self):
        columns = []
        for axis, n in enumerate(self.dims):
            matrices = [np.zeros((d, d)) for d in self.dims]
            matrices[axis] = np.eye(n)
            column = self.coordinates(matrices)
            if column is not None:
                columns.append(column)
        return np.column_stack(columns) if columns else np.empty((self.dimension, 0))


class TransverseOpsSymmetries(TransverseOps):
    """Axis operators shared by signed representative labels.

    ``relations`` uses Julia's signed one-based notation: ``[1, -1, 3]``
    shares the first operator with its transpose on axis 1, while axis 2
    has an independent operator. Representatives must point to themselves.
    """

    def __init__(self, dims, operators=None, relations=None):
        if relations is None:
            raise ValueError('relations are required')
        self.dims = tuple(int(n) for n in dims)
        if any(n <= 0 for n in self.dims):
            raise ValueError('dimensions must be positive')
        if operators is None:
            operators = UniversalOp()
        if isinstance(operators, Operator):
            operators = [operators] * len(self.dims)
        self.operators = tuple(operators)
        if len(self.operators) != len(self.dims) or len(relations) != len(self.dims):
            raise ValueError('one operator and relation per axis are required')
        self.relations = tuple(int(r) for r in relations)
        self.roots = tuple(abs(r) - 1 for r in self.relations)
        self.duals = tuple(r < 0 for r in self.relations)
        for axis, root in enumerate(self.roots):
            if root > axis or root < 0 or self.roots[root] != root or self.duals[root]:
                raise ValueError('invalid representative relation')
            if self.dims[axis] != self.dims[root] or type(self.operators[axis]) is not type(self.operators[root]):
                raise ValueError('related axes must have equal dimensions and operator spaces')
        self.representatives = tuple(i for i, root in enumerate(self.roots) if i == root)
        self._sizes = tuple(self.operators[i].dimension(self.dims[i]) for i in self.representatives)
        self.dimension = sum(self._sizes)

    def embed(self, data):
        data = np.asarray(data)
        if data.ndim != 1 or len(data) != self.dimension:
            raise ValueError('incompatible transverse coordinates')
        offsets = np.cumsum((0,) + self._sizes)
        root_mats = {root: self.operators[root].embed(self.dims[root], data[offsets[j]:offsets[j + 1]])
                     for j, root in enumerate(self.representatives)}
        return tuple(root_mats[root].T if dual else root_mats[root]
                     for root, dual in zip(self.roots, self.duals))

    def adjoint(self, matrices):
        matrices = tuple(np.asarray(m) for m in matrices)
        if len(matrices) != len(self.dims) or any(m.shape != (n, n) for m, n in zip(matrices, self.dims)):
            raise ValueError('incompatible transverse matrices')
        parts = []
        for root in self.representatives:
            total = sum((m.T if self.duals[i] else m for i, m in enumerate(matrices) if self.roots[i] == root))
            parts.append(self.operators[root].transpose_embed(total))
        return np.concatenate(parts) if parts else np.empty(0)

    def coordinates(self, matrices):
        matrices = tuple(np.asarray(m) for m in matrices)
        if len(matrices) != len(self.dims) or any(m.shape != (n, n) for m, n in zip(matrices, self.dims)):
            raise ValueError('incompatible transverse matrices')
        parts = []
        for root in self.representatives:
            reference = matrices[root]
            part = self.operators[root].coordinates(reference)
            if part is None:
                return None
            for i, m in enumerate(matrices):
                if self.roots[i] == root and not np.allclose(m, reference.T if self.duals[i] else reference):
                    return None
            parts.append(part)
        return np.concatenate(parts) if parts else np.empty(0)

    def scalar_vectors(self):
        columns = []
        for root in self.representatives:
            matrices = [np.eye(n) if self.roots[i] == root else np.zeros((n, n))
                        for i, n in enumerate(self.dims)]
            column = self.coordinates(matrices)
            if column is not None:
                columns.append(column)
        return np.column_stack(columns) if columns else np.empty((self.dimension, 0))
