import unittest
import numpy as np
from opendleto import nondeg


class TensorTests(unittest.TestCase):
    def test_low_rank(self):
        rng = np.random.default_rng(12)
        for complex_input in (False, True):
            factors = [rng.normal(size=(n, r)) for n, r in [(4, 2), (5, 3), (3, 2)]]
            core = rng.normal(size=(2, 3, 2))
            if complex_input:
                factors = [a + 1j * rng.normal(size=a.shape) for a in factors]
                core = core + 1j * rng.normal(size=core.shape)
            tensor = np.einsum('ijk,ai,bj,ck->abc', core, *factors)
            result = nondeg(tensor)
            self.assertEqual(result.ranks, (2, 3, 2))
            np.testing.assert_allclose(result.reconstruct(), tensor, atol=1e-11)
            for basis in result.bases:
                np.testing.assert_allclose(basis.conj().T @ basis, np.eye(basis.shape[1]), atol=1e-12)

    def test_full_and_selected_axes(self):
        tensor = np.arange(24.).reshape(2, 3, 4)
        result = nondeg(tensor, axes=[2, 0], mode='full')
        self.assertEqual(result.axes, (2, 0))
        self.assertEqual(result.ranks, (4, 2))
        np.testing.assert_allclose(result.reconstruct(), tensor, atol=1e-12)

    def test_threshold_and_zero(self):
        matrix = np.diag([10., 1., .1])
        self.assertEqual(nondeg(matrix, axes=[0], tol=.1).ranks, (1,))
        self.assertEqual(nondeg(matrix * 1e-12, axes=[0], tol=.1).ranks, (1,))
        zero = nondeg(np.zeros((2, 3)))
        self.assertEqual(zero.ranks, (0, 0))
        np.testing.assert_array_equal(zero.reconstruct(), np.zeros((2, 3)))

    def test_invalid_options(self):
        for axes in ([0, 0], [2], [-1]):
            with self.assertRaises(ValueError):
                nondeg(np.ones((2, 2)), axes)
        with self.assertRaises(ValueError):
            nondeg(np.ones(2), tol=-1)
        with self.assertRaises(ValueError):
            nondeg(np.ones(2), mode='unknown')


if __name__ == '__main__':
    unittest.main()
