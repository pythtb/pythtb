import numpy as np
import pytest

def test_hamiltonian_spinless(ssh_model):
    k = np.linspace(-np.pi, np.pi, 5)
    H = ssh_model.hamiltonian(k_pts=k)
    assert H.shape == (5, 2, 2)
    np.testing.assert_allclose(H[0, 0, 1], np.conj(H[0, 1, 0]))

def test_parameter_normalization():
    from pythtb.tbmodel import TBModel, Lattice
    lat = Lattice([[1]], [[0]], periodic_dirs=[0])
    tb = TBModel(lat)
    vals, step, periodic, trimmed = tb._normalize_parameter_axis(
        np.linspace(0, 2*np.pi, 5, endpoint=True),
        name="beta",
        period=2*np.pi,
    )
    assert periodic and trimmed
    assert vals.size == 4
    assert step == pytest.approx(np.pi/2)

def test_velocity_parametric_ssh(ssh_model):
    ssh_model.set_onsite("delta", ind_i=0)
    vel = ssh_model.velocity(
        k_pts = np.linspace(0,1,10), 
        delta = np.array(np.linspace(0, 1, 15))
    )
    assert vel.shape[0] == ssh_model.dim_k + 1

def test_velocity_parametric_fkm(fkm_model):
    k_pts = fkm_model.k_uniform_mesh([5,5,5])
    vel = fkm_model.velocity(
        k_pts = k_pts,
        beta = np.array(np.linspace(0, 1, 15))
    )
    assert vel.shape[0] == fkm_model.dim_k + 1

def test_berry_curvature_second_chern(fkm_model):
    betas = np.linspace(0, 2*np.pi, 10, endpoint=False)
    betas, theta_curve, c2 = fkm_model.axion_angle(
        nks=(13, 13, 13),
        beta=betas,
        param_periods={"beta": 2*np.pi},
        return_second_chern=True,
    )
    assert 0.8 < c2 < 1.2   # coarse grid tolerance
    assert theta_curve.shape[0] == betas.size

