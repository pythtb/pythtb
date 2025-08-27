import numpy as np
from pythtb import *

def trestle_model():
    "Return a trestle-like model on a rectangular lattice."

    lat = [[2.0, 0.0], [0.0, 1.0]]
    orb = [[0.0, 0.0], [0.5, 1.0]]

    my_model = tb_model(1, 2, lat, orb, per=[0])

    return my_model

def run():
    my_model = trestle_model()

    t_first = 0.8 + 0.6j
    t_second = 2.0

    my_model.set_hop(t_second, 0, 0, [1, 0])
    my_model.set_hop(t_second, 1, 1, [1, 0])
    my_model.set_hop(t_first, 0, 1, [0, 0])
    my_model.set_hop(t_first, 1, 0, [1, 0])

    # path: [-pi, 0, pi]
    k_vec, _, _ = my_model.k_path("fullc", 100)

    evals = my_model.solve_all(k_vec)

    return evals
