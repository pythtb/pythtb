import numpy as np
from pythtb import tb_model

def haldane():
    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    orb = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]

    my_model = tb_model(2, 2, lat, orb)
    delta = 0.0
    t = -1.0
    t2 = 0.15 * np.exp((1.0j) * np.pi / 2.0)
    t2c = t2.conjugate()

    my_model.set_onsite([-delta, delta])
    my_model.set_hop(t, 0, 1, [0, 0])
    my_model.set_hop(t, 1, 0, [1, 0])
    my_model.set_hop(t, 1, 0, [0, 1])
    my_model.set_hop(t2, 0, 0, [1, 0])
    my_model.set_hop(t2, 1, 1, [1, -1])
    my_model.set_hop(t2, 1, 1, [0, 1])
    my_model.set_hop(t2c, 1, 1, [1, 0])
    my_model.set_hop(t2c, 0, 0, [1, -1])
    my_model.set_hop(t2c, 0, 0, [0, 1])

    return my_model


def run():
    my_model = haldane()
  
    tmp_model = my_model.cut_piece(10, 0, glue_edgs=False)
    fin_model = tmp_model.cut_piece(10, 1, glue_edgs=False)

    tmp_model_half = my_model.cut_piece(10, 0, glue_edgs=True)
    fin_model_half = tmp_model_half.cut_piece(10, 1, glue_edgs=False)

    (evals, evecs) = fin_model.solve_all(eig_vectors=True)
    (evals_half, evecs_half) = fin_model_half.solve_all(eig_vectors=True)

    return evals, evecs, evals_half, evecs_half