import numpy as np

from pythtb import Lattice, Mesh, WFArray


def _make_1d_wfa(include_endpoint: bool) -> WFArray:
    """Utility that builds a minimal 1D WFArray with one orbital."""
    lattice = Lattice(lat_vecs=[[1.0]], orb_vecs=[[0.0]], periodic_dirs=[0])
    mesh = Mesh(dim_k=1, axis_types=["k"])
    mesh.build_grid([4], k_endpoints=include_endpoint)
    return WFArray(lattice, mesh)


def test_links_periodic_without_endpoints():
    """Links should wrap through PBC when the axis has no explicit endpoint."""
    wfa = _make_1d_wfa(include_endpoint=False)
    n_pts = wfa.mesh.shape_mesh[0]
    states = np.ones((n_pts, 1, 1), dtype=complex)
    wfa.set_states(states)

    links = wfa.links()
    assert links.shape == (1, n_pts, 1, 1)
    np.testing.assert_allclose(links[0, :, 0, 0], 1.0)


def test_links_zero_out_closed_endpoint():
    """When the mesh stores the endpoint explicitly, the last link should vanish."""
    wfa = _make_1d_wfa(include_endpoint=True)
    n_pts = wfa.mesh.shape_mesh[0]
    states = np.ones((n_pts, 1, 1), dtype=complex)
    wfa.set_states(states)

    links = wfa.links()
    assert links.shape == (1, n_pts, 1, 1)
    np.testing.assert_allclose(links[0, :-1, 0, 0], 1.0)
    np.testing.assert_allclose(links[0, -1, 0, 0], 0.0)


def test_set_states_enforces_closed_axis_phase():
    """Closing a loop should make the endpoint match the starting point after set_states."""
    wfa = _make_1d_wfa(include_endpoint=True)
    n_pts = wfa.mesh.shape_mesh[0]
    states = np.zeros((n_pts, 1, 1), dtype=complex)
    states[0, 0, 0] = 1.0
    states[-1, 0, 0] = 2.0  # should be overwritten by enforcement

    wfa.set_states(states)
    np.testing.assert_allclose(wfa.wfs[-1, 0, 0], wfa.wfs[0, 0, 0])

    # __setitem__ should keep the endpoint in sync as well.
    wfa[0] = np.array([[np.exp(1j)]], dtype=complex)
    np.testing.assert_allclose(wfa.wfs[-1, 0, 0], wfa.wfs[0, 0, 0])


def test_setitem_does_not_touch_open_axis():
    """Updating an open axis should not modify the opposite edge."""
    wfa = _make_1d_wfa(include_endpoint=False)
    n_pts = wfa.mesh.shape_mesh[0]
    states = np.ones((n_pts, 1, 1), dtype=complex)
    wfa.set_states(states)

    wfa[0] = np.array([[2.0]], dtype=complex)
    np.testing.assert_allclose(wfa.wfs[0, 0, 0], 2.0)
    np.testing.assert_allclose(wfa.wfs[-1, 0, 0], 1.0)
