import numpy as np
from pythtb import TBModel, Lattice


def haldane():
    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    orb = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]
    lat = Lattice(lat, orb, periodic_dirs=[0, 1])

    my_model = TBModel(lattice=lat)
    delta = 0.0
    t = -1.0
    t2 = 0.15 * np.exp(1j * np.pi / 2.0)
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

    tmp_model = my_model.cut_piece(10, 0, glue_edges=False)
    fin_model = tmp_model.cut_piece(10, 1, glue_edges=False)

    tmp_model_half = my_model.cut_piece(10, 0, glue_edges=True)
    fin_model_half = tmp_model_half.cut_piece(10, 1, glue_edges=False)

    evals, evecs = fin_model.solve_ham(return_eigvecs=True)
    evals_half, evecs_half = fin_model_half.solve_ham(return_eigvecs=True)

    # golden_evals_half = np.load("golden_outputs/evals_half.npy")
    # print("Golden evals_half:", golden_evals_half.round(6))
    # print("Computed evals_half:", evals_half.round(6))

    return evals, evecs, evals_half, evecs_half
