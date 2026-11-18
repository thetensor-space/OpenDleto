import unittest
import numpy as np

from opendleto import (DerivationMap, UniversalChisel, AdjointChisel, TuckerChisel,
                       TransverseOps, TransverseOpsSymmetries, DiagonalOp, UniversalOp, derivations, stratify, tensor)


class DerivationTests(unittest.TestCase):
    def test_native_tensor_element(self):
        value = tensor(np.ones((2, 2, 2)))
        self.assertEqual(derivations(value).total_nullity, 8)
        self.assertEqual(value.data.shape, (2, 2, 2))

    def test_complex_adjoint(self):
        rng = np.random.default_rng(23)
        tensor = rng.normal(size=(2, 3, 2)) + 1j * rng.normal(size=(2, 3, 2))
        for chisel in (UniversalChisel(3), AdjointChisel(3, 0, 2), TuckerChisel(3)):
            d = DerivationMap(tensor, chisel)
            x = rng.normal(size=d.shape[1]) + 1j * rng.normal(size=d.shape[1])
            y = rng.normal(size=(chisel.shape[0], *tensor.shape)) + 1j * rng.normal(size=(chisel.shape[0], *tensor.shape))
            self.assertAlmostEqual(abs(np.vdot(d.forward(x), y) - np.vdot(x, d.adjoint(y))), 0, places=11)

    def test_matrix_multiplication_nullity(self):
        tensor = np.zeros((4, 4, 4))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    tensor[2*i+j, 2*j+k, 2*k+i] = 1
        r = derivations(tensor, max_vectors=12)
        self.assertTrue(r.exact)
        self.assertEqual((r.total_nullity, r.trivial_scalar_dimension, r.nontrivial_nullity_estimate), (11, 2, 9))
        self.assertLess(max(r.residuals), 1e-10)

    def test_generic_negative_and_basis_change(self):
        rng = np.random.default_rng(11)
        tensor = rng.normal(size=(3, 3, 3))
        r = derivations(tensor)
        self.assertEqual((r.total_nullity, r.nontrivial_nullity_estimate), (2, 0))
        matrices = [rng.normal(size=(3, 3)) for _ in range(3)]
        changed = tensor.copy()
        for axis, m in enumerate(matrices):
            changed = np.moveaxis(np.tensordot(m, changed, axes=(1, axis)), 0, axis)
        self.assertEqual(derivations(changed).total_nullity, r.total_nullity)

    def test_operator_covariance(self):
        rng = np.random.default_rng(17)
        base = np.zeros((2, 2, 2))
        base[0, 0, 0], base[1, 1, 1] = 1, 2
        mats = (np.diag([0., 1.]), np.diag([0., -1.]), np.zeros((2, 2)))
        self.assertLess(np.linalg.norm(DerivationMap(base).forward(TransverseOps(base.shape).coordinates(mats))), 1e-12)
        transforms = [rng.normal(size=(2, 2)) for _ in range(3)]
        changed = base.copy()
        transformed_ops = []
        for axis, (frame, op) in enumerate(zip(transforms, mats)):
            changed = np.moveaxis(np.tensordot(frame, changed, axes=(1, axis)), 0, axis)
            transformed_ops.append(frame @ op @ np.linalg.inv(frame))
        d = DerivationMap(changed)
        coordinates = d.operators.coordinates(transformed_ops)
        self.assertLess(np.linalg.norm(d.forward(coordinates)), 1e-10)

    def test_graded_support(self):
        tensor = np.zeros((3, 3, 3))
        tensor[0, 0, 0], tensor[1, 1, 1], tensor[2, 2, 2] = 2, 3, 4
        r = derivations(tensor)
        s = stratify(tensor, r)
        self.assertEqual(s.method, "diagonal")
        self.assertEqual(s.support_count, 3)
        self.assertLess(s.reconstruction_error, 1e-10)

    def test_iterative_is_partial(self):
        tensor = np.random.default_rng(2).normal(size=(3, 3, 3))
        r = derivations(tensor, dense_max_entries=1, max_vectors=3)
        self.assertFalse(r.exact)
        self.assertIsNone(r.total_nullity)
        self.assertGreaterEqual(r.nullity_lower_bound, 0)

    def test_coupled_transverse_adjoint_and_gauge(self):
        rng = np.random.default_rng(91)
        t = rng.normal(size=(2, 2, 2))
        ops = TransverseOpsSymmetries(t.shape, UniversalOp(), [1, -1, 3])
        d = DerivationMap(t, operators=ops)
        x = rng.normal(size=d.shape[1])
        y = rng.normal(size=(1, *t.shape))
        self.assertAlmostEqual(float(np.vdot(d.forward(x), y)), float(np.vdot(x, d.adjoint(y))), places=12)
        self.assertEqual(derivations(t, operators=ops).trivial_scalar_dimension, 1)

    def test_diagonal_operator_map(self):
        tensor = np.random.default_rng(2).normal(size=(2, 2, 2))
        ops = TransverseOps(tensor.shape, DiagonalOp())
        d = DerivationMap(tensor, operators=ops)
        rng = np.random.default_rng(7)
        x = rng.normal(size=d.shape[1])
        y = rng.normal(size=(1, *tensor.shape))
        self.assertAlmostEqual(float(np.vdot(d.forward(x), y)), float(np.vdot(x, d.adjoint(y))), places=12)


if __name__ == '__main__':
    unittest.main()

class JuliaParityTests(unittest.TestCase):
    def test_stored_julia_fixture(self):
        import json
        from pathlib import Path
        fixture = json.loads((Path(__file__).parent / 'fixtures/julia_diagonal_222.json').read_text())
        tensor = np.zeros(fixture['tensor_shape'])
        for i, j, k, value in fixture['tensor_nonzero']:
            tensor[i, j, k] = value
        d = DerivationMap(tensor, np.array(fixture['chisel']))
        e1 = np.eye(d.shape[1])[0]
        np.testing.assert_allclose(d.forward(e1).ravel(order='F'), fixture['forward_e1'])
        matrix = np.column_stack([d.forward(v).ravel(order='F') for v in np.eye(d.shape[1])])
        np.testing.assert_allclose(np.linalg.svd(matrix, compute_uv=False), fixture['singular_values'], atol=1e-12)
        self.assertEqual(derivations(tensor).total_nullity, fixture['nullity'])
