import numpy as np
from pythtb import tb_model

def graphene_model():
    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    orb = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]
    my_model = tb_model(2, 2, lat, orb)

    return my_model

def run():
    my_model = graphene_model()

    delta = 0.0
    t = -1.0
    my_model.set_onsite([-delta, delta])
    my_model.set_hop(t, 0, 1, [0, 0])
    my_model.set_hop(t, 1, 0, [1, 0])
    my_model.set_hop(t, 1, 0, [0, 1])

    # path: Gamma, K, M, Gamma
    path = [[0.0, 0.0], [2.0 / 3.0, 1.0 / 3.0], [0.5, 0.5], [0.0, 0.0]]
    nk = 121
    k_vec, _, _ = my_model.k_path(path, nk)
   
    evals = my_model.solve_all(k_vec)

    return evals


