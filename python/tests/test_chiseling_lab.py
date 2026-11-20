import unittest

import numpy as np

from opendleto import apply_transforms, chisel_tensor, direct_sum, nondeg, randomize_tensor


class ChiselingLabTests(unittest.TestCase):
    def test_tucker_and_block_recovery(self):
        rng = np.random.default_rng(42)
        tucker = direct_sum(rng.normal(size=(5, 5, 5)), np.zeros((2, 1, 0)))
        self.assertEqual(tucker.shape, (7, 6, 5))
        mixed, frames = randomize_tensor(tucker, rng=rng)
        np.testing.assert_allclose(apply_transforms(tucker, frames), mixed)
        reduction = nondeg(mixed, mode="trunc")
        self.assertEqual(reduction.core.shape, (5, 5, 5))
        np.testing.assert_allclose(reduction.reconstruct(), mixed, atol=1e-10)
        found = chisel_tensor(mixed)
        self.assertGreater(found.derivation_count, 0)
        self.assertTrue(found.exact)
        np.testing.assert_allclose(apply_transforms(mixed, found.transforms), found.tensor)
        self.assertEqual(np.count_nonzero(np.abs(found.tensor) > 1e-5), 125)

        blocks = direct_sum(rng.normal(size=(3, 2, 7)),
                            rng.normal(size=(4, 5, 2)),
                            rng.normal(size=(5, 3, 3)))
        mixed, _ = randomize_tensor(blocks, rng=rng)
        found = chisel_tensor(mixed)
        self.assertGreater(found.derivation_count, 0)
        self.assertTrue(found.exact)
        self.assertLess(np.count_nonzero(np.abs(found.tensor) > 1e-5), 200)

    def test_generic_control_has_no_extra_derivations(self):
        rng = np.random.default_rng(7)
        control = rng.normal(size=(7, 6, 5))
        found = chisel_tensor(control)
        self.assertEqual(found.derivation_count, 0)
        self.assertTrue(found.exact)
        np.testing.assert_array_equal(found.tensor, control)


if __name__ == "__main__":
    unittest.main()
