import numpy as np
import pytest

from opendleto import TensorElement, rand_surface_tensor


def test_surface_support_and_column_major_draws():
    axes = ([-1, 0, 1], [0, 1], [0, 1])
    result = rand_surface_tensor(*axes, 1, rng=np.random.default_rng(42))
    assert isinstance(result, TensorElement)
    assert result.shape == (3, 2, 2)
    expected = np.zeros(result.shape, order="F")
    rng = np.random.default_rng(42)
    for k, z in enumerate(axes[2]):
        for j, y in enumerate(axes[1]):
            for i, x in enumerate(axes[0]):
                if abs(x + y + z) < 1:
                    expected[i, j, k] = rng.standard_normal()
    np.testing.assert_array_equal(result.data, expected)


def test_seed_reproducibility_and_empty_support():
    args = ([0, 1], [0, 1], [0, 1], 2)
    a = rand_surface_tensor(*args, rng=np.random.default_rng(19))
    b = rand_surface_tensor(*args, rng=np.random.default_rng(19))
    np.testing.assert_array_equal(a.data, b.data)
    assert not np.any(rand_surface_tensor([0], [0], [0], 0).data)
    with pytest.raises(ValueError, match="one-dimensional"):
        rand_surface_tensor([[0]], [0], [0], 1)
