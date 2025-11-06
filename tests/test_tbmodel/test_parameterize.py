import numpy as np
from pythtb import TBModel, Lattice
import pytest


def test_ribbon():
    """
    Test the TBModel initialization for a ribbon geometry.
    """
    lat = Lattice(
        lat_vecs=[[1, 0], [0, 2]],
        orb_vecs=[[0, 0], [0.5, 0.5], [0.5, 1.5]],
        periodic_dirs=[1],
    )
    tb = TBModel(lattice=lat, spinful=False)

    # Check dimensions
    assert tb.dim_k == 1, "dim_k should be 1 for ribbon"
    assert tb.dim_r == 2, "dim_r should be 2 for ribbon"
    assert tb.norb == 3, "norb should be 3 for ribbon"

    # Check lattice vectors and orbital positions
    np.testing.assert_array_equal(
        tb.lat_vecs, [[1, 0], [0, 2]], "lat_vecs should match"
    )
    np.testing.assert_array_equal(
        tb.orb_vecs, [[0, 0], [0.5, 0.5], [0.5, 1.5]], "orbital positions should match"
    )

    tb.set_onsite([0.0, 1.0, 0.5])  # onsite energies for all three orbitals
    tb.set_hop(1.0, 0, 1, [0, 0])  # hopping from orbital 0 to 1 within home unit cell
    tb.set_hop(-0.8, 1, 2, [0, 1])  # one cell away along the periodic axis

    np.testing.assert_array_equal(
        tb.onsite, [0.0, 1.0, 0.5], "Onsite energies should match"
    )
    assert tb.hoppings[0]["amplitude"] == 1.0, "Hopping from 0 to 1 should be 1.0"
    assert tb.hoppings[1]["amplitude"] == -0.8, "Hopping from 1 to 2 should be -0.8"
    assert len(tb.hoppings) == 2, "There should be 2 hoppings defined"

    tb.set_onsite("U", ind_i=0)  # onsite energy for orbital 0 is parameter 'U'
    assert tb.onsite[0] == 0, (
        "Onsite energy for orbital 0 should be 0 as a placeholder for parameter"
    )

    tb.set_parameters({"U": 2.5})
    assert tb.onsite[0] == 2.5, "Onsite energy for orbital 0 should be updated to 2.5"

    tb.set_hop(
        "t1", 2, 0, [0, -1]
    )  # hopping from orbital 2 to 0 shifted by one cell back along periodic axis
    assert len(tb.hoppings) == 2, (
        "There should still be 2 hoppings defined after adding parameterized hopping"
    )

    tb.set_parameters(t1=0.3)
    assert len(tb.hoppings) == 3, (
        "There should now be 3 hoppings defined after setting parameters"
    )
    assert tb.hoppings[2]["amplitude"] == 0.3, (
        "Hopping from 2 to 0 should be updated to 0.3"
    )

    tb.set_onsite(
        lambda U: np.cos(U), ind_i=2
    )  # onsite energy for orbital 2 is parameter 'U'
    tb.set_hop(
        lambda t: t**2, 1, 2, [0, 0]
    )  # hopping from orbital 1 to 2 is parameter 't'

    assert len(tb.parameters) == 2, "There should be 2 parameters defined"

    evals, evecs = tb.solve_ham(
        k_pts=np.linspace(0, 1, 10), return_eigvecs=True, U=0.4, t=0.7
    )
    assert evals.shape == (10, 3), "Eigenvalues shape should be (10, 3)"
    assert evecs.shape == (10, 3, 3), "Eigenvectors shape should be (10, 3, 3)"
    assert len(tb.parameters) == 2, "There should still be 2 parameters defined"

    U_vals = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    t_vals = [0.0, 0.5, 1.0, 1.5]
    evals, evecs = tb.solve_ham(
        k_pts=np.linspace(0, 1, 10), return_eigvecs=True, U=U_vals, t=t_vals
    )
    assert evals.shape == (10, 4, 4, 3), "Eigenvalues shape should be (10, 4, 4, 3)"
    assert evecs.shape == (10, 4, 4, 3, 3), (
        "Eigenvectors shape should be (10, 4, 4, 3, 3)"
    )


def test_parameter_normalization():
    from pythtb.tbmodel import TBModel, Lattice

    lat = Lattice([[1]], [[0]], periodic_dirs=[0])
    tb = TBModel(lat)
    # parameter with period 2pi
    vals, step, periodic, trimmed = tb._normalize_parameter_axis(
        np.linspace(0, 2 * np.pi, 5, endpoint=True),
        name="beta",
        period=2 * np.pi,
    )
    # values include period, so should be trimmed and periodic
    assert periodic and trimmed
    # One less value after trimming
    assert vals.size == 4
    # step size should now be pi/2
    assert step == pytest.approx(np.pi / 2)
