import numpy as np
from pythtb import tb_model

def haldane_model():
    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    orb = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]

    my_model = tb_model(2, 2, lat, orb)

    delta = 0.2
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
    my_model = haldane_model() 
    
    path = [
        [0.0, 0.0],
        [2.0 / 3.0, 1.0 / 3.0],
        [0.5, 0.5],
        [1.0 / 3.0, 2.0 / 3.0],
        [0.0, 0.0],
    ]
    k_vec, _, _ = my_model.k_path(path, 101)

    evals = my_model.solve_all(k_vec)

    kmesh = 20
    kpts = []
    for i in range(kmesh):
        for j in range(kmesh):
            kpts.append([float(i) / float(kmesh), float(j) / float(kmesh)])
    evals_dos = my_model.solve_all(kpts)
    evals_dos = evals_dos.flatten()

    return evals, evals_dos