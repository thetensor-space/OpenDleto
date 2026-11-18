"""Optional backend checks; skipped when PyTorch is not installed."""
import unittest
import numpy as np
from opendleto import DerivationMap

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch optional extra is not installed')
class TorchBackendTests(unittest.TestCase):
    def test_cpu_matches_numpy_and_adjoint(self):
        rng = np.random.default_rng(31)
        array = rng.normal(size=(2, 3, 2))
        for dtype in (torch.float64, torch.complex128):
            native = array.astype(np.complex128) + 1j * rng.normal(size=array.shape) if dtype.is_complex else array
            td = DerivationMap(torch.as_tensor(native, dtype=dtype))
            nd = DerivationMap(native)
            x = rng.normal(size=td.shape[1])
            y = rng.normal(size=(1, *array.shape))
            np.testing.assert_allclose(td.forward(x).numpy(), nd.forward(x), atol=1e-12)
            np.testing.assert_allclose(td.adjoint(y), nd.adjoint(y), atol=1e-12)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA is not available')
    def test_cuda_forward_matches_cpu(self):
        array = np.random.default_rng(3).normal(size=(2, 3, 2))
        x = np.arange(17, dtype=float)  # 2² + 3² + 2²
        gpu = DerivationMap(torch.tensor(array, device='cuda'))
        cpu = DerivationMap(array)
        np.testing.assert_allclose(gpu.forward(x).cpu().numpy(), cpu.forward(x), atol=1e-12)

    @unittest.skipUnless(torch is not None and torch.backends.mps.is_available(), 'MPS is not available')
    def test_mps_forward_matches_cpu(self):
        rng = np.random.default_rng(8)
        array = rng.normal(size=(2, 3, 2)).astype(np.float32)
        x = rng.normal(size=17).astype(np.float32)
        gpu = DerivationMap(torch.tensor(array, device='mps'))
        cpu = DerivationMap(array)
        actual = gpu.forward(x)
        torch.mps.synchronize()
        self.assertEqual(actual.device.type, 'mps')
        np.testing.assert_allclose(actual.cpu().numpy(), cpu.forward(x), atol=2e-6, rtol=2e-6)


if __name__ == '__main__':
    unittest.main()
