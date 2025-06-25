from pythtb import TBModel
import pytest
import numpy as np

lat_vecs = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
orbital_pos = [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]]

SIGMA_0 = np.array([[1, 0], [0, 1]], dtype=complex)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
SIGMAS = [SIGMA_0, SIGMA_X, SIGMA_Y, SIGMA_Z]

@pytest.mark.parametrize("nspin", [1, 2])
def test_set_onsite(nspin):
    """
    Test the TBModel with nspin=1.
    """
    # Create a TBModel instance with nspin=1
    test_model = TBModel(3, 3, lat=lat_vecs, orb=orbital_pos, nspin=nspin)

    if nspin == 1:
        # setting with list for each orbital
        onsite_values = [1.0, 2.0, 3.0]
        test_model.set_onsite(onsite_values)

        # needs to be real number
        onsite_values = 1
        test_model.set_onsite(onsite_values, ind_i=0)

    elif nspin == 2:
        # setting with list of pauli components for each orbital
        onsite_values = [[0, 1, 2, 2], [0, 1, 2, 2], [0, 1, 2, 2]]
        test_model.set_onsite(onsite_values)

        # onsite should be sum of Pauli matrices
        onsite_check = np.zeros((test_model.norb, 2, 2), dtype=complex)
        for i in range(len(onsite_values)):
            onsite_check[i] = np.sum(
                [onsite_values[i][j] * SIGMAS[j] for j in range(4)], axis=0
            )
        assert np.allclose(test_model.site_energies, onsite_check)

        # Now try a list of numbers, should be proprto SIGMA_0 for each orbital
        onsite_values = [1, 2, 3]
        test_model.set_onsite(onsite_values)
        onsite_check = np.zeros((test_model.norb, 2, 2), dtype=complex)
        for i in range(len(onsite_values)):
            onsite_check[i] = onsite_values[i] * SIGMA_0
        
        assert np.allclose(test_model.site_energies, onsite_check)

