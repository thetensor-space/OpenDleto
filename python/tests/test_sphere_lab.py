"""Small end-to-end checks for the Python sphere lab API."""

import numpy as np

from opendleto import (SymmetricOp, apply_transforms, nondeg,
                       rand_surface_tensor, randomize_tensor, stratify)


def test_densor_sphere_recovery_and_symmetric_restriction():
    coordinates = np.arange(7) ** 2 - 25 / 3
    surface = rand_surface_tensor(coordinates, coordinates, coordinates, 1.5,
                                  rng=np.random.default_rng(20261118)).data
    randomized, _ = randomize_tensor(surface,
                                     rng=np.random.default_rng(20261118))
    reduced = nondeg(randomized)
    recovered = stratify(reduced.core, tol=1e-7, max_vectors=4)
    assert recovered.derivation_count > 0
    assert np.allclose(apply_transforms(reduced.core, recovered.transforms),
                       recovered.tensor)
    assert recovered.forbidden_energy_fraction < 1e-10

    orthogonal, frames = randomize_tensor(surface,
                                          rng=np.random.default_rng(20261119),
                                          kind="orthogonal")
    assert all(np.allclose(frame.T @ frame, np.eye(len(frame))) for frame in frames)
    symmetric = stratify(orthogonal, operators=SymmetricOp(),
                         tol=1e-7, max_vectors=4)
    assert symmetric.derivation_count > 0
    assert symmetric.forbidden_energy_fraction < 1e-10
