import numpy as np
from pythtb import TBModel, Lattice
from pythtb.utils import pauli_decompose


def ssh(v, w):
    lat = [[1]]
    orb = [[0], [1]]
    lattice = Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=[0])
    my_model = TBModel(lattice=lattice, spinful=False)

    my_model.set_hop(v, 1, 0, [1])
    my_model.set_hop(w, 0, 1, [0])

    return my_model


def run():
    v = -1  # intercell hopping
    w_init = -0.5  # initial intracell hopping

    # define a path in k-space
    k_vec, _, _ = ssh(v, w_init).k_path("full", 100)

    model = ssh(v, w_init)
    _, evecs = model.solve_ham(k_vec, return_eigvecs=True)
    ham = model.hamiltonian(k_vec)

    # Compute phase difference
    numerator = evecs[..., 1]
    denominator = evecs[..., 0]
    # Mask invalid (zero) denominators to avoid warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        phase_diff = np.angle(numerator / denominator)
        # Where both components are zero, set phase to NaN
        phase_diff[~np.isfinite(phase_diff)] = np.nan

    # plot path of endpoints of d-vec in dx dy plane as wave-vector sweeps from 0 to 2pi
    d_vec = np.zeros((len(k_vec), 4), dtype=complex)
    for k in range(len(k_vec)):
        d_vec[k] = pauli_decompose(ham[k])

    return evecs, d_vec
