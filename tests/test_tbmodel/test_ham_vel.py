import numpy as np
import pytest

from pythtb.lattice import Lattice
from pythtb.tbmodel import TBModel


def test_1d_two_orbitals():
    """1D chain with two orbitals at t_0=0.0 and t_1=0.3 (reduced coords).

    Hoppings:
      - on-site: e_0 = 1.0, e_1 = -1.0
      - set_hop(t, 0, 1, [0])  -> H_{01}(R=0) = t        (intra-cell)
      - set_hop(s, 0, 1, [1])  -> H_{01}(R=1) = s        (inter-cell)

    The conjugate hoppings are added automatically:
      - H_{10}(R= 0) = t*   from the first hop
      - H_{10}(R=-1) = s*   from the second hop

    Convention I Hamiltonian at wave-vector k (reduced):
      H^k_{00} = e_0
                 (no hoppings with i=j=0 were set)

      H^k_{11} = e_1
                 (no hoppings with i=j=1 were set)

      H^k_{01} = sum_R e^{2*pi*i * k*(R + t_1 - t_0)} * H_{01}(R)
               = t * e^{2*pi*i * k*(0 + 0.3)}
               + s * e^{2*pi*i * k*(1 + 0.3)}

      H^k_{10} = [H^k_{01}]*   (Hermiticity)
    """
    lat = [[1.0]]
    orb = [[0.0], [0.3]]
    model = TBModel(lattice=Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=...))

    e0 = 1.0
    e1 = -1.0
    model.set_onsite([e0, e1])

    t = 0.8 + 0.6j
    s = 0.3 - 0.2j
    model.set_hop(t, 0, 1, [0])
    model.set_hop(s, 0, 1, [1])

    # Test at several k-points
    for k in [0.0, 0.25, 0.1, -0.3, 0.5]:
        ham = model.hamiltonian([k])

        dt = 0.3  # t_1 - t_0

        h01_expected = t * np.exp(2.0j * np.pi * k * (0 + dt)) + s * np.exp(
            2.0j * np.pi * k * (1 + dt)
        )

        expected = np.array(
            [
                [e0, h01_expected],
                [h01_expected.conjugate(), e1],
            ]
        )

        np.testing.assert_allclose(
            ham[0], expected, atol=1e-12, err_msg=f"Hamiltonian mismatch at k={k}"
        )


def test_2d_checkerboard_single_k():
    """2D model with orbitals at (0,0) and (0.5,0.5), one hopping.

    set_hop(t, 1, 0, [0,0])  ->  H_{10}(R=(0,0)) = t
    Auto-conjugate:              H_{01}(R=(0,0)) = t*

    H^k_{10} = t * e^{2*pi*i * k . (R + t_0 - t_1)}
             = t * e^{2*pi*i * k . ((0,0) + (0,0) - (0.5,0.5))}
             = t * e^{2*pi*i * (-(k1+k2)/2)}

    H^k_{01} = t* * e^{2*pi*i * k . ((0,0) + (0.5,0.5) - (0,0))}
             = t* * e^{2*pi*i * (k1+k2)/2}
             = [H^k_{10}]*       (consistent)
    """
    lat = [[1.0, 0.0], [0.0, 1.0]]
    orb = [[0.0, 0.0], [0.5, 0.5]]
    model = TBModel(lattice=Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=...))

    delta = 1.1
    t = 0.6
    model.set_onsite([-delta, delta])
    model.set_hop(t, 1, 0, [0, 0])

    k = [0.25, 0.1]
    ham = model.hamiltonian(k)

    dt = np.array([0.0, 0.0]) - np.array([0.5, 0.5])  # t_0 - t_1
    rv = np.array([0, 0]) + dt  # R + t_j - t_i where i=1, j=0

    h10 = t * np.exp(2.0j * np.pi * np.dot(k, rv))

    expected = np.array(
        [
            [-delta, h10.conjugate()],
            [h10, delta],
        ]
    )

    np.testing.assert_allclose(ham[0], expected, atol=1e-10)


def test_hermiticity():
    """Verify the Hamiltonian is Hermitian for a model with complex hoppings."""
    lat = [[1.0, 0.0], [0.0, 1.0]]
    orb = [[0.0, 0.0], [0.25, 0.35]]
    model = TBModel(lattice=Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=...))

    model.set_onsite([0.5, -0.5])
    model.set_hop(0.7 + 0.3j, 0, 1, [0, 0])
    model.set_hop(0.2 - 0.1j, 0, 1, [1, 0])
    model.set_hop(0.4 + 0.2j, 0, 1, [0, 1])

    for k1 in np.linspace(-0.5, 0.5, 5):
        for k2 in np.linspace(-0.5, 0.5, 5):
            ham = model.hamiltonian([k1, k2])
            ham_hc = np.swapaxes(ham.conjugate(), -1, -2)
            np.testing.assert_allclose(
                ham, ham_hc, atol=1e-12, err_msg=f"Not Hermitian at k=({k1},{k2})"
            )


def test_phases_from_reduced_dot_products():
    frac = np.array([[0.125, 0.25, -0.5], [0.0, -0.375, 0.625]])
    shifts = np.array([[1e8, -1e10, 1e6], [5e7, 2e9, -3e8]])
    k_dot_r = shifts + frac

    expected = np.exp(1j * 2 * np.pi * k_dot_r)
    phases = TBModel._phases_from_reduced_dot_products(k_dot_r.copy())
    np.testing.assert_allclose(phases, expected, atol=1e-14, rtol=1e-14)


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
