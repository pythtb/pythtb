import numpy as np
from pythtb import TBModel

def one_dim_model():   
    lat = [[1.0]]
    orb = [[0.0]]
    my_model = TBModel(1, 1, lat, orb)
    my_model.set_hop(-1.0, 0, 0, [1])
    return my_model

def run():
    my_model = one_dim_model()
    k_vec, _, _ = my_model.k_path("full", 100)
    evals = my_model.solve_ham(k_vec)
    return evals

