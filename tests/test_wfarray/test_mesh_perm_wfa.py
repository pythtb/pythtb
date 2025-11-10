import numpy as np
from pythtb import TBModel, Mesh, WFArray, Lattice


def _phase_state_from_coords(mesh: Mesh) -> np.ndarray:
    coords = mesh.points
    phase = np.exp(2j * np.pi * (coords[..., 0] + coords[..., -1]))
    return phase[..., np.newaxis, np.newaxis]


def test_solve_model_respects_arbitrary_axis_order():
    lat = Lattice([[1.0]], [[0.0]], periodic_dirs=[0])
    mesh = Mesh(["l", "k"], axis_names=["phi", "k"])
    mesh.build_grid(
        shape=(2, 3),
        gamma_centered=[False],
        k_endpoints=[False],
        lambda_start=[0.0],
        lambda_stop=[np.pi],
        lambda_endpoints=[True],
    )

    wfa = WFArray(lat, mesh, spinful=False)
    model = TBModel(lat, spinful=False)
    model.set_onsite(lambda phi: phi, ind_i=0)

    wfa.solve_model(model)

    lambda_axis_idx = mesh.lambda_axis_indices[0]
    lambda_component_idx = mesh.lambda_component_indices[0]
    param_vals = mesh.get_axis_range(lambda_axis_idx, lambda_component_idx)

    energies = wfa.energies[..., 0]
    expected = np.broadcast_to(param_vals[:, None], energies.shape)
    np.testing.assert_allclose(energies, expected)


def _assign_phase_states(wfa):
    coords = wfa.mesh.points
    k_vals = coords[..., 0]
    lam_vals = coords[..., wfa.mesh.dim_k]
    phase = 0.7 * k_vals + 0.31 * lam_vals
    psi = np.exp(1j * phase)
    wfs = psi[..., np.newaxis, np.newaxis]
    wfa.set_states(wfs, is_cell_periodic=True, is_spin_axis_flat=False)
    wfa._energies = np.zeros(wfa.mesh.shape_axes + (wfa.nstates,))


def _assign_flux_states(wfa):
    coords = wfa.mesh.points
    kx = coords[..., 0]
    ky = coords[..., 1]
    lam_vals = coords[..., wfa.mesh.dim_k]
    theta = 0.4 * (kx + lam_vals)
    phi = ky
    vec = np.stack(
        [np.cos(theta), np.sin(theta) * np.exp(1j * phi)],
        axis=-1,
    )
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True)
    wfs = vec[..., np.newaxis, :]
    wfa.set_states(wfs, is_cell_periodic=True, is_spin_axis_flat=False)
    wfa._energies = np.zeros(wfa.mesh.shape_axes + (wfa.nstates,))


def test_berry_phase_handles_axis_reordering():
    lat = Lattice([[1.0]], [[0.0]], periodic_dirs=[0])

    mesh_kl = Mesh(["k", "l"])
    mesh_kl.build_grid(
        shape=(8, 5),
        gamma_centered=[True],
        k_endpoints=[False],
        lambda_start=[0.0],
        lambda_stop=[np.pi],
        lambda_endpoints=[True],
    )
    mesh_lk = Mesh(["l", "k"])
    mesh_lk.build_grid(
        shape=(5, 8),
        gamma_centered=[True],
        k_endpoints=[False],
        lambda_start=[0.0],
        lambda_stop=[np.pi],
        lambda_endpoints=[True],
    )

    wfa_kl = WFArray(lat, mesh_kl, nstates=1, spinful=False)
    wfa_lk = WFArray(lat, mesh_lk, nstates=1, spinful=False)
    _assign_phase_states(wfa_kl)
    _assign_phase_states(wfa_lk)

    phase_kl = wfa_kl.berry_phase(axis_idx=0, state_idx=[0], contin=False)
    phase_lk = wfa_lk.berry_phase(axis_idx=1, state_idx=[0], contin=False)
    np.testing.assert_allclose(phase_kl, phase_lk)


def test_berry_flux_handles_axis_reordering():
    lat = Lattice(
        [[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.5, 0.5]], periodic_dirs=[0, 1]
    )

    mesh_kkl = Mesh(["k", "k", "l"])
    mesh_kkl.build_grid(
        shape=(4, 5, 3),
        gamma_centered=[True, False],
        k_endpoints=[False, False],
        lambda_start=[0.0],
        lambda_stop=[np.pi / 2],
        lambda_endpoints=[True],
    )

    mesh_lkk = Mesh(["l", "k", "k"])
    mesh_lkk.build_grid(
        shape=(3, 4, 5),
        gamma_centered=[True, False],
        k_endpoints=[False, False],
        lambda_start=[0.0],
        lambda_stop=[np.pi / 2],
        lambda_endpoints=[True],
    )

    wfa_kkl = WFArray(lat, mesh_kkl, nstates=1, spinful=False)
    wfa_lkk = WFArray(lat, mesh_lkk, nstates=1, spinful=False)
    _assign_flux_states(wfa_kkl)
    _assign_flux_states(wfa_lkk)

    flux_kkl = wfa_kkl.berry_flux(state_idx=[0], plane=(0, 1))
    flux_lkk = wfa_lkk.berry_flux(state_idx=[0], plane=(1, 2))

    perm = wfa_lkk._mesh_axes_to_canonical()
    flux_lkk = np.transpose(flux_lkk, perm)
    np.testing.assert_allclose(flux_kkl, flux_lkk)


def test_links_with_reordered_axes():
    lat = Lattice(
        lat_vecs=[[1.0, 0.0], [0.0, 1.0]], orb_vecs=[[0.0, 0.0]], periodic_dirs=[0, 1]
    )
    mesh_kl = Mesh(["k", "k", "l"])
    mesh_kl.build_grid(shape=(5, 5, 3), gamma_centered=True, k_endpoints=False)

    mesh_lk = Mesh(["l", "k", "k"])
    mesh_lk.build_grid(shape=(3, 5, 5), gamma_centered=True, k_endpoints=False)

    wfa_kl = WFArray(lat, mesh_kl)
    wfa_lk = WFArray(lat, mesh_lk)
    wfa_kl.set_states(_phase_state_from_coords(mesh_kl))
    wfa_lk.set_states(_phase_state_from_coords(mesh_lk))

    links_kl = wfa_kl.links(axis_idx=0)
    links_lk = wfa_lk.links(axis_idx=1)
    perm = wfa_lk._mesh_axes_to_canonical()
    # add 1 to account for leading axis_idx dim
    perm = tuple(p + 1 for p in perm)
    links_lk = np.transpose(links_lk, (0,) + perm + (4, 5))

    np.testing.assert_allclose(links_kl, links_lk)


def test_roll_states_with_pbc_axis_reordering():
    lat = Lattice(
        lat_vecs=[[1.0, 0.0], [0.0, 1.0]], orb_vecs=[[0.0, 0.0]], periodic_dirs=[0, 1]
    )
    mesh_kl = Mesh(["k", "k", "l"])
    mesh_kl.build_grid(
        shape=(5, 5, 3), gamma_centered=[False, False], k_endpoints=[True, True]
    )
    mesh_lk = Mesh(["l", "k", "k"])
    mesh_lk.build_grid(
        shape=(3, 5, 5), gamma_centered=[False, False], k_endpoints=[True, True]
    )

    wfa_kl = WFArray(lat, mesh_kl)
    wfa_lk = WFArray(lat, mesh_lk)
    wfa_kl.set_states(_phase_state_from_coords(mesh_kl))
    wfa_lk.set_states(_phase_state_from_coords(mesh_lk))

    rolled_kl = wfa_kl.roll_states_with_pbc([1, 0], flatten_spin_axis=False)
    rolled_lk = wfa_lk.roll_states_with_pbc([0, 1], flatten_spin_axis=False)

    perm = wfa_lk._mesh_axes_to_canonical()
    rolled_lk = np.transpose(rolled_lk, perm + (3, 4))
    np.testing.assert_allclose(rolled_kl, rolled_lk)
