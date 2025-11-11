import numpy as np
from pythtb import tb_model

def run():
    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    orb = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]

    my_model = tb_model(2, 2, lat, orb)

    delta = 0.0
    t = -1.0

    my_model.set_onsite([-delta, delta])
    my_model.set_hop(t, 0, 1, [0, 0])
    my_model.set_hop(t, 1, 0, [1, 0])
    my_model.set_hop(t, 1, 0, [0, 1])

    sc_model = my_model.make_supercell([[2, 1], [-1, 2]], to_home=True)

    slab_model = sc_model.cut_piece(6, 1, glue_edgs=False)

    (k_vec, k_dist, k_node) = slab_model.k_path("full", 100)
    evals = slab_model.solve_all(k_vec)

    return evals

