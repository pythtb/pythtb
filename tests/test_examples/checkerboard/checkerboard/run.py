import numpy as np
from pythtb import *

def checkerboard_model():
    "Return a checkerboard model on a rectangular lattice."
    lat = [[1.0, 0.0], [0.0, 1.0]]
    orb = [[0.0, 0.0], [0.5, 0.5]]
    my_model = tb_model(2, 2, lat, orb)
    return my_model

def run():
    my_model = checkerboard_model()

    delta = 1.1
    t = 0.6

    my_model.set_onsite([-delta, delta])
    my_model.set_hop(t, 1, 0, [0, 0])
    my_model.set_hop(t, 1, 0, [1, 0])
    my_model.set_hop(t, 1, 0, [0, 1])
    my_model.set_hop(t, 1, 0, [1, 1])

    # path in k-space: [Gamma, X, M, Gamma]
    path = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.5], [0.0, 0.0]]
    k_vec, _, _ = my_model.k_path(path, 301)

    evals = my_model.solve_all(k_vec)

    return evals
