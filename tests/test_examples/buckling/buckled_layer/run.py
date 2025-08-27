import numpy as np
from pythtb import *

def buckled_model():
    "Return a buckled layer model on a rectangular lattice."

    # define lattice vectors
    lat = [[1.0, 0.0, 0.0], [0.0, 1.25, 0.0], [0.0, 0.0, 3.0]]
    # define coordinates of orbitals
    orb = [[0.0, 0.0, -0.15], [0.5, 0.5, 0.15]]

    # only first two lattice vectors repeat, so k-space is 2D
    my_model = tb_model(2, 3, lat, orb)

    return my_model

def run():
    my_model = buckled_model()

    delta = 1.1
    t = 0.6

    my_model.set_onsite([-delta, delta])
    my_model.set_hop(t, 1, 0, [0, 0, 0])
    my_model.set_hop(t, 1, 0, [1, 0, 0])
    my_model.set_hop(t, 1, 0, [0, 1, 0])
    my_model.set_hop(t, 1, 0, [1, 1, 0])

    # path: [Gamma, X, M, Gamma]
    path = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.5], [0.0, 0.0]]
    k_vec, _, _ = my_model.k_path(path, 81)

    evals = my_model.solve_all(k_vec)

    return evals