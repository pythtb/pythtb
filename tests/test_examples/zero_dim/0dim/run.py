import numpy as np
from pythtb import tb_model 

def run():
    lat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    sq32 = np.sqrt(3.0) / 2.0
    orb = [
        [(2.0 / 3.0) * sq32, 0.0, 0.0],
        [(-1.0 / 3.0) * sq32, 1.0 / 2.0, 0.0],
        [(-1.0 / 3.0) * sq32, -1.0 / 2.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    my_model = tb_model(0, 3, lat, orb)

    delta = 0.5
    t_first = 1.0

    my_model.set_onsite([-delta, -delta, -delta, delta])
    my_model.set_hop(t_first, 0, 1)
    my_model.set_hop(t_first, 0, 2)
    my_model.set_hop(t_first, 0, 3)
    my_model.set_hop(t_first, 1, 2)
    my_model.set_hop(t_first, 1, 3)
    my_model.set_hop(t_first, 2, 3)

    evals = my_model.solve_all()

    return evals
