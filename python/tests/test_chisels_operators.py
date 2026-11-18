import unittest
import numpy as np
from opendleto import (UniversalChisel, TuckerChisel, AdjointChisel, CentroidChisel,
                       engaged, normalize_chisel, UniversalOp, DiagonalOp,
                       SymmetricOp, AntiSymmetricOp, ScalarOp, EmptyOp, TransverseOps)


class ChiselOperatorTests(unittest.TestCase):
    def test_chisels(self):
        np.testing.assert_array_equal(UniversalChisel([1, 0, 1]), [[1, 0, 1]])
        np.testing.assert_array_equal(TuckerChisel([1, 0, 1]), [[1, 0, 0], [0, 0, 1]])
        np.testing.assert_array_equal(AdjointChisel(3, 0, 2), [[1, 0, -1]])
        np.testing.assert_array_equal(CentroidChisel(3), [[1, -1, 0], [1, 0, -1], [0, 1, -1]])
        p = np.array([[2., 0, -2], [1, 0, -1]])
        q = normalize_chisel(p)
        np.testing.assert_array_equal(engaged(p), [True, False, True])
        np.testing.assert_allclose(q @ q.T, np.eye(1))
        np.testing.assert_allclose(p @ (np.eye(3) - q.T @ q), 0, atol=1e-14)

    def test_local_spaces(self):
        for op in (UniversalOp(), DiagonalOp(), SymmetricOp(), AntiSymmetricOp(), ScalarOp(), EmptyOp()):
            data = np.arange(op.dimension(3), dtype=float) + 1
            matrix = op.embed(3, data)
            np.testing.assert_allclose(op.coordinates(matrix), data)
            probe = np.arange(9, dtype=float).reshape(3, 3)
            np.testing.assert_allclose(np.sum(matrix * probe), data @ op.transpose_embed(probe))

    def test_membership_signs(self):
        self.assertIsNone(AntiSymmetricOp().coordinates(np.eye(2)))
        self.assertIsNone(EmptyOp().coordinates([[0, 1], [0, 0]]))
        np.testing.assert_array_equal(AntiSymmetricOp().embed(2, [3]), [[0, 3], [-3, 0]])
        np.testing.assert_array_equal(UniversalOp().embed(2, [1, 2, 3, 4]), [[1, 3], [2, 4]])

    def test_transverse(self):
        ops = TransverseOps([2, 3], [SymmetricOp(), AntiSymmetricOp()])
        data = np.arange(ops.dimension, dtype=float)
        mats = ops.embed(data)
        np.testing.assert_array_equal(ops.coordinates(mats), data)
        probe = (np.arange(4).reshape(2, 2), np.arange(9).reshape(3, 3))
        np.testing.assert_allclose(sum(np.sum(a * b) for a, b in zip(mats, probe)), data @ ops.adjoint(probe))
        self.assertEqual(ops.scalar_vectors().shape, (ops.dimension, 1))


if __name__ == '__main__':
    unittest.main()
