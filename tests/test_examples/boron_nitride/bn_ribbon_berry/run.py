import numpy as np
import os
import sys

# Now import your module
from pythtb import *

def bn_model(t, delta):
    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    orb = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]
    my_model = tb_model(2, 2, lat, orb)
    my_model.set_onsite([-delta, delta])
    my_model.set_hop(t, 0, 1, [0, 0])
    my_model.set_hop(t, 1, 0, [1, 0])
    my_model.set_hop(t, 1, 0, [0, 1])
    return my_model

def run():
    t = -1.0
    delta = 0.4

    model_orig = bn_model(t, delta).cut_piece(3, 1, glue_edgs=False)

    numk = 41
    k_vec, _, _ = model_orig.k_path([[-0.5], [0.5]], numk, report=False)
    eval = model_orig.solve_all(k_vec, eig_vectors=False)
    n_bands = eval.shape[0]

    wf = wf_array(model_orig, [numk])
    wf.solve_on_grid([0.0])
    n_occ = n_bands // 2
    berry_phase = wf.berry_phase(range(n_occ), dir=0)

    model_perp = model_orig.change_nonperiodic_vector(1)
    numk = 41
    k_vec, _, _ = model_perp.k_path([[-0.5], [0.5]], numk, report=False)
    eval = model_perp.solve_all(k_vec, eig_vectors=False)
    n_bands = eval.shape[0]

    wf = wf_array(model_perp, [numk])
    wf.solve_on_grid([0.0])
    n_occ = n_bands // 2
    berry_phase2 = wf.berry_phase(range(n_occ), dir=0)

    return berry_phase, berry_phase2