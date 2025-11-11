import numpy as np
from pythtb import Lattice, Mesh, WFArray


def make_1d_wfa(include_endpoint: bool) -> WFArray:
    """Utility that builds a minimal 1D WFArray with one orbital."""
    lattice = Lattice(lat_vecs=[[1.0]], orb_vecs=[[0.0]], periodic_dirs=[0])
    mesh = Mesh(["k"])
    mesh.build_grid([4], k_endpoints=include_endpoint)
    return WFArray(lattice, mesh)


def test_setitem_does_not_touch_open_axis():
    """Updating an open axis should not modify the opposite edge."""
    wfa = make_1d_wfa(include_endpoint=False)
    n_pts = wfa.mesh.shape_axes[0]
    states = np.ones((n_pts, 1, 1), dtype=complex)
    wfa.set_states(states)

    wfa[0] = np.array([[2.0]], dtype=complex)
    np.testing.assert_allclose(wfa.wfs[0, 0, 0], 2.0)
    np.testing.assert_allclose(wfa.wfs[-1, 0, 0], 1.0)


def test_set_states_enforces_closed_axis_phase():
    """Closing a loop should make the endpoint match the starting point after set_states."""
    wfa = make_1d_wfa(include_endpoint=True)
    n_pts = wfa.mesh.shape_axes[0]
    states = np.zeros((n_pts, 1, 1), dtype=complex)
    states[0, 0, 0] = 1.0
    states[-1, 0, 0] = 2.0  # should be overwritten by enforcement

    wfa.set_states(states)
    np.testing.assert_allclose(wfa.wfs[-1, 0, 0], wfa.wfs[0, 0, 0])

    # __setitem__ should keep the endpoint in sync as well.
    wfa[0] = np.array([[np.exp(1j)]], dtype=complex)
    np.testing.assert_allclose(wfa.wfs[-1, 0, 0], wfa.wfs[0, 0, 0])
