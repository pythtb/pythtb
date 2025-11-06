import numpy as np
import pytest

from pythtb.lattice import Lattice
from pythtb.tbmodel import TBModel


def test_hamiltonian_spinless(ssh_model):
    k = np.linspace(-np.pi, np.pi, 5)
    H = ssh_model.hamiltonian(k_pts=k)
    assert H.shape == (5, 2, 2)
    # check hermiticity
    np.testing.assert_allclose(H, np.conj(np.transpose(H, (0, 2, 1))))


def test_hamiltonian_spinful(fkm_model):
    k_pts = fkm_model.k_uniform_mesh([4, 4, 4])
    H_nonflat = fkm_model.hamiltonian(k_pts=k_pts, beta=np.pi)
    assert H_nonflat.shape == (64, 2, 2, 2, 2)
    H_flat = fkm_model.hamiltonian(k_pts=k_pts, beta=np.pi, flatten_spin_axis=True)
    assert H_flat.shape == (64, 4, 4)
    # check hermiticity
    np.testing.assert_allclose(
        H_nonflat, np.conj(np.transpose(H_nonflat, (0, 3, 4, 1, 2)))
    )
    np.testing.assert_allclose(H_flat, np.conj(np.transpose(H_flat, (0, 2, 1))))


def test_velocity_parametric_ssh(ssh_model):
    # Set parametric onsite energy
    ssh_model.set_onsite("delta", ind_i=0)
    # Pass range of delta values
    vel_c = ssh_model.velocity(
        k_pts=np.linspace(0, 1, 10),
        delta=np.array(np.linspace(0, 1, 15)),
        diff_scheme="central",
    )
    # Velocity should have an additional value on axis 0 for delta
    assert vel_c.shape[0] == ssh_model.dim_k + 1

    vel_f = ssh_model.velocity(
        k_pts=np.linspace(0, 1, 10),
        delta=np.array(np.linspace(0, 1, 15)),
        diff_scheme="forward",
    )
    # Velocity should have an additional value on axis 0 for delta
    assert vel_f.shape[0] == ssh_model.dim_k + 1

    # Should be roughly equal
    np.testing.assert_allclose(vel_c, vel_f, rtol=1e-10, atol=1e-10)

    vel_flat = ssh_model.velocity(
        k_pts=np.linspace(0, 1, 10).flatten(),
        delta=np.array(np.linspace(0, 1, 15)),
        diff_scheme="central",
        flatten_spin_axis=True,
    )

    assert vel_flat.shape[-1] == ssh_model.norb * ssh_model.nspin

    # create 1d model with orbs at origin, cart and non-cart vel should coincide

    model = TBModel(
        lattice=Lattice(lat_vecs=[[1.0]], orb_vecs=2, periodic_dirs=[0]), spinful=True
    )

    model.set_onsite([0.0, 0.0])
    model.set_hop(1.0, 0, 1, [0])

    k_pts = model.k_uniform_mesh([10])
    vel_cart = model.velocity(k_pts=k_pts, cartesian=True)
    vel_frac = model.velocity(k_pts=k_pts, cartesian=False)
    np.testing.assert_allclose(vel_cart, vel_frac, rtol=1e-10, atol=1e-10)

    # now shift orbitals away from origin, cart and frac vel should differ
    model_shift = TBModel(
        lattice=Lattice(lat_vecs=[[1.0]], orb_vecs=[[0.25], [0.75]], periodic_dirs=[0]),
        spinful=True,
    )
    model_shift.set_onsite([0.0, 0.0])
    model_shift.set_hop(1.0, 0, 1, [0])
    vel_cart_shift = model_shift.velocity(k_pts=k_pts, cartesian=True)
    vel_frac_shift = model_shift.velocity(k_pts=k_pts, cartesian=False)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            vel_cart_shift, vel_frac_shift, rtol=1e-10, atol=1e-10
        )


# Same test as above but for 3D FKM model
def test_velocity_parametric_fkm(fkm_model):
    k_pts = fkm_model.k_uniform_mesh([5, 5, 5])
    vel = fkm_model.velocity(k_pts=k_pts, beta=np.array(np.linspace(0, 1, 15)))
    assert vel.shape == (
        fkm_model.dim_k + 1,
        k_pts.shape[0],
        15,
        fkm_model.norb,
        fkm_model.nspin,
        fkm_model.norb,
        fkm_model.nspin,
    )
    vel_flat = fkm_model.velocity(
        k_pts=k_pts, beta=np.array(np.linspace(0, 1, 15)), flatten_spin_axis=True
    )
    assert vel_flat.shape == (
        fkm_model.dim_k + 1,
        k_pts.shape[0],
        15,
        fkm_model.norb * fkm_model.nspin,
        fkm_model.norb * fkm_model.nspin,
    )

    vel_noncart = fkm_model.velocity(
        k_pts=k_pts, beta=np.array(np.linspace(0, 1, 15)), cartesian=False
    )
    vel_cart = fkm_model.velocity(
        k_pts=k_pts, beta=np.array(np.linspace(0, 1, 15)), cartesian=True
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(vel_noncart, vel_cart, rtol=1e-10, atol=1e-10)

    vel_per_ep = fkm_model.velocity(
        k_pts=k_pts,
        beta=np.array(np.linspace(0, 2 * np.pi, 15, endpoint=True)),
        cartesian=True,
        param_periods={"beta": 2 * np.pi},
    )
    assert vel_per_ep.shape == (
        fkm_model.dim_k + 1,
        k_pts.shape[0],
        15,
        fkm_model.norb,
        fkm_model.nspin,
        fkm_model.norb,
        fkm_model.nspin,
    )

    vel_per_nep = fkm_model.velocity(
        k_pts=k_pts,
        beta=np.array(np.linspace(0, 2 * np.pi, 15, endpoint=False)),
        cartesian=True,
        param_periods={"beta": 2 * np.pi},
    )
    assert vel_per_nep.shape == (
        fkm_model.dim_k + 1,
        k_pts.shape[0],
        15,
        fkm_model.norb,
        fkm_model.nspin,
        fkm_model.norb,
        fkm_model.nspin,
    )
