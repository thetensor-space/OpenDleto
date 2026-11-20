"""Random tensors supported near prescribed surfaces."""

import numpy as np

from .tensors import tensor


def rand_surface_tensor(xes, yes, zes, cutoff, *, rng=None):
    """Sample a tensor where ``abs(x[i] + y[j] + z[k]) < cutoff``.

    Supported entries are independent standard normal draws; other entries
    are zero. Supply a NumPy Generator to reproduce samples. Julia and NumPy
    use different random streams, even for the same seed.
    """
    axes = tuple(np.asarray(axis) for axis in (xes, yes, zes))
    if any(axis.ndim != 1 for axis in axes):
        raise ValueError("surface coordinates must be one-dimensional")
    if rng is None:
        rng = np.random.default_rng()
    support = np.abs(axes[0][:, None, None] + axes[1][None, :, None]
                     + axes[2][None, None, :]) < cutoff
    # Julia visits CartesianIndices in column-major order and draws only on support.
    flat_support = support.ravel(order="F")
    values = np.zeros(flat_support.size)
    values[flat_support] = rng.standard_normal(np.count_nonzero(flat_support))
    return tensor(values.reshape(support.shape, order="F"))
