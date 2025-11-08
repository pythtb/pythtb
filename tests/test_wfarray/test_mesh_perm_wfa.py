import numpy as np
from pythtb import TBModel, Mesh, WFArray, Lattice


def test_solve_model_respects_arbitrary_axis_order():
    lat = Lattice([[1.0]], [[0.0]], periodic_dirs=[0])
    mesh = Mesh(dim_k=1, dim_lambda=1, axis_types=["l", "k"], axis_names=["phi", "k"])
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

    mesh_kl = Mesh(dim_k=1, dim_lambda=1, axis_types=["k", "l"])
    mesh_kl.build_grid(
        shape=(8, 5),
        gamma_centered=[True],
        k_endpoints=[False],
        lambda_start=[0.0],
        lambda_stop=[np.pi],
        lambda_endpoints=[True],
    )
    mesh_lk = Mesh(dim_k=1, dim_lambda=1, axis_types=["l", "k"])
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

    mesh_kkl = Mesh(dim_k=2, dim_lambda=1, axis_types=["k", "k", "l"])
    mesh_kkl.build_grid(
        shape=(4, 5, 3),
        gamma_centered=[True, False],
        k_endpoints=[False, False],
        lambda_start=[0.0],
        lambda_stop=[np.pi / 2],
        lambda_endpoints=[True],
    )

    mesh_lkk = Mesh(dim_k=2, dim_lambda=1, axis_types=["l", "k", "k"])
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
