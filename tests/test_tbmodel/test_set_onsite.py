from pythtb import TBModel, Lattice
import pytest
import numpy as np

lat_vecs = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
orbital_pos = [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]]

SIGMA_0 = np.array([[1, 0], [0, 1]], dtype=complex)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
SIGMAS = [SIGMA_0, SIGMA_X, SIGMA_Y, SIGMA_Z]


@pytest.mark.parametrize("spinful", [False, True])
def test_set_onsite(spinful):
    """
    Test the TBModel with nspin=1.
    """
    latt = Lattice(lat_vecs=lat_vecs, orb_vecs=orbital_pos, periodic_dirs=[0, 1, 2])
    test_model = TBModel(lattice=latt, spinful=spinful)

    if not spinful:
        # setting with list for each orbital
        onsite_values = [1.0, 2.0, 3.0]
        test_model.set_onsite(onsite_values)

        # needs to be real number
        onsite_values = 1
        test_model.set_onsite(onsite_values, ind_i=0)

    else:
        # setting with list of pauli components for each orbital
        onsite_values = [[0, 1, 2, 2], [0, 1, 2, 2], [0, 1, 2, 2]]
        test_model.set_onsite(onsite_values)

        # onsite should be sum of Pauli matrices
        onsite_check = np.zeros((test_model.norb, 2, 2), dtype=complex)
        for i in range(len(onsite_values)):
            onsite_check[i] = np.sum(
                [onsite_values[i][j] * SIGMAS[j] for j in range(4)], axis=0
            )
        assert np.allclose(test_model.onsite, onsite_check)

        # Now try a list of numbers, should be proprto SIGMA_0 for each orbital
        onsite_values = [1, 2, 3]
        test_model.set_onsite(onsite_values)
        onsite_check = np.zeros((test_model.norb, 2, 2), dtype=complex)
        for i in range(len(onsite_values)):
            onsite_check[i] = onsite_values[i] * SIGMA_0

        assert np.allclose(test_model.onsite, onsite_check)
