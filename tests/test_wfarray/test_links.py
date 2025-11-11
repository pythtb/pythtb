import numpy as np
import pytest

from pythtb import Lattice, Mesh, WFArray
from pythtb.models import haldane, kane_mele


def make_1d_wfa(include_endpoint: bool) -> WFArray:
    """Utility that builds a minimal 1D WFArray with one orbital."""
    lattice = Lattice(lat_vecs=[[1.0]], orb_vecs=[[0.0]], periodic_dirs=[0])
    mesh = Mesh(["k"])
    mesh.build_grid([4], k_endpoints=include_endpoint)
    return WFArray(lattice, mesh)


def test_links_periodic_without_endpoints():
    """Links should wrap through PBC when the axis has no explicit endpoint."""
    wfa = make_1d_wfa(include_endpoint=False)
    n_pts = wfa.mesh.shape_axes[0]
    states = np.ones((n_pts, 1, 1), dtype=complex)
    wfa.set_states(states)

    links = wfa.links()
    assert links.shape == (1, n_pts, 1, 1)
    np.testing.assert_allclose(links[0, :, 0, 0], 1.0)


def test_links_zero_out_closed_endpoint():
    """When the mesh stores the endpoint explicitly, the last link should vanish."""
    wfa = make_1d_wfa(include_endpoint=True)
    n_pts = wfa.mesh.shape_axes[0]
    states = np.ones((n_pts, 1, 1), dtype=complex)
    wfa.set_states(states)

    links = wfa.links()
    assert links.shape == (1, n_pts, 1, 1)
    np.testing.assert_allclose(links[0, :-1, 0, 0], 1.0)
    assert np.isnan(links[0, -1, 0, 0])


@pytest.mark.parametrize("delta", np.linspace(-2, 2, 5))
def test_berry_connection_phase(delta):
    t = 1
    t2 = -0.3
    model = haldane(delta, t, t2)
    nkx = nky = 100
    mesh = Mesh(["k", "k"])
    mesh.build_grid((nkx, nky))
    wfa = WFArray(model.lattice, mesh=mesh)
    wfa.solve_model(model)

    # Berry connection along kx only (axis 0)
    A_kx = wfa.berry_connection(state_idx=[0], axis_idx=0)
    A_kx = np.squeeze(A_kx[0])  # shape: (nkx, nky)

    dkx = 1.0 / nkx
    ky_idx = nky // 2  # choose a representative line

    # Drop NaNs from the duplicate endpoint, if any
    loop_vals = A_kx[:, ky_idx]
    valid = ~np.isnan(loop_vals)
    phase_from_A = np.sum(loop_vals[valid] * dkx)

    # Wrap into (-π, π]
    phase_from_A = np.angle(np.exp(1j * phase_from_A))

    # Built-in Wilson loop / Berry phase along kx
    phase_builtin = wfa.berry_phase(axis_idx=0, state_idx=[0], contin=True)[ky_idx]

    diff = phase_from_A - phase_builtin
    wrapped = (diff + np.pi) % (2 * np.pi) - np.pi

    np.testing.assert_allclose(wrapped, 0, atol=1e-6)


def test_berry_connection_hermiticity():
    model = kane_mele(1.0, 0.6, 0.1, 0.1)

    nkx = nky = 20
    mesh = Mesh(["k", "k"])
    mesh.build_grid((nkx, nky))
    wfa = WFArray(model.lattice, mesh=mesh, spinful=True)
    wfa.solve_model(model)

    A = wfa.berry_connection(
        state_idx=[0, 1], axis_idx=(0, 1)
    )  # shape: (2, nkx, nky, 2, 2)

    assert A.shape == (2, nkx, nky, 2, 2)

    np.testing.assert_allclose(A, np.conj(np.swapaxes(A, -2, -1)))


def test_berry_connection_invalid_state_idx():
    model = kane_mele(1.0, 0.6, 0.1, 0.1)

    nkx = nky = 20
    mesh = Mesh(["k", "k"])
    mesh.build_grid((nkx, nky))
    wfa = WFArray(model.lattice, mesh=mesh, spinful=True)
    wfa.solve_model(model)

    with pytest.raises(IndexError):
        wfa.berry_connection(state_idx=[100], axis_idx=(0,))


@pytest.mark.parametrize("delta", np.linspace(-2, 2, 5))
def test_berry_connection_finite_diff(delta):
    t = 1
    t2 = -0.3
    model = haldane(delta, t, t2)
    nkx = nky = 100
    mesh = Mesh(["k", "k"])
    mesh.build_grid((nkx, nky))
    wfa = WFArray(model.lattice, mesh=mesh)
    wfa.solve_model(model)

    # helper to unwrap phases against a reference
    def unwrap_to_ref(phase, ref):
        out = np.array(phase, copy=True)
        mask = ~np.isnan(out)
        out[mask] = np.unwrap(out[mask], period=2 * np.pi)

        ref_arr = np.broadcast_to(ref, out.shape)
        ref_mask = ~np.isnan(ref_arr)

        if not np.any(ref_mask):
            return out

        shift = np.round((ref_arr[ref_mask] - out[ref_mask]) / (2 * np.pi))
        out[ref_mask] += shift * 2 * np.pi
        return out

    # build a finite-difference estimate for comparison
    dk = [1.0 / nkx, 1.0 / nky]
    state = 0
    A_kx = wfa.berry_connection(
        state_idx=[state], axis_idx=(0,)
    )  # shape: (1, nkx, nky, 1, 1)
    A_kx = np.squeeze(A_kx)  # shape: (nkx, nky)
    psi = wfa.states(state_idx=state, flatten_spin_axis=True)
    psi_shift_x = wfa.roll_states_with_pbc(
        [1, 0], flatten_spin_axis=True, strip_boundary=True
    )[..., state, :]
    overlap_x = np.squeeze(np.einsum("...a,...a->...", psi.conj(), psi_shift_x))
    branch = unwrap_to_ref(np.angle(overlap_x), np.angle(overlap_x[..., 0]))
    fd_Ax = 1j * branch / dk[0]
    # drop NaNs for the equality check
    mask = ~np.isnan(A_kx[..., 0, 0])
    print("A_x diag match:", np.nanmax(np.abs(A_kx[..., 0, 0][mask] - fd_Ax[mask])))


def test_berry_connection_cartesian_step():
    lattice = Lattice(
        lat_vecs=[[1, 0], [0, 1]], orb_vecs=[[0, 0]], periodic_dirs=[0, 1]
    )
    mesh = Mesh(["k"], dim_k=2)
    points = np.linspace([0, 0], [1, 1], 6, endpoint=False)
    mesh.build_custom(points)
    wfa = WFArray(lattice, mesh)
    gradient = np.array([0.3, 0.1])  # phase ramp in reduced k
    phases = np.exp(2j * np.pi * (points @ gradient))
    wfa.set_states(phases[:, None, None])

    A_red = wfa.berry_connection(axis_idx=(0,), cartesian=False)[0, :-1, 0, 0].real
    A_cart = wfa.berry_connection(axis_idx=(0,), cartesian=True)[0, :-1, 0, 0].real

    step = points[1] - points[0]
    phase_step = 2 * np.pi * (gradient @ step)

    expected_red = -phase_step / step[0]
    expected_cart = -phase_step / np.linalg.norm(step @ lattice.recip_lat_vecs)

    np.testing.assert_allclose(A_red, expected_red, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(A_cart, expected_cart, rtol=1e-12, atol=1e-12)
