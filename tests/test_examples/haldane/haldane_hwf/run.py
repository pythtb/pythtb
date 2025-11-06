import numpy as np
from pythtb import TBModel, WFArray, Lattice, Mesh


def haldane_model():
    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    orb = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]
    lattice = Lattice(lat, orb, periodic_dirs=[0, 1])

    my_model = TBModel(lattice)

    delta = -0.2
    t = -1.0
    t2 = 0.05 - 0.15j
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

    efermi = 0.25
    len_0 = 100
    len_1 = 10

    mesh = Mesh(dim_k=2, axis_types=["k", "k"])
    mesh.build_grid(shape=(len_0, len_1), k_endpoints=True)
    my_array = WFArray(my_model.lattice, mesh)
    my_array.solve_model(my_model)
    phi_1 = my_array.berry_phase(state_idx=[0], axis_idx=1, contin=True)

    ribbon_model = my_model.cut_piece(len_1, periodic_dir=1, glue_edges=False)
    (k_vec, k_dist, k_node) = ribbon_model.k_path([0.0, 0.5, 1.0], len_0, report=False)
    rib_eval, rib_evec = ribbon_model.solve_ham(k_vec, return_eigvecs=True)
    rib_eval -= efermi

    jump_k = []
    for i in range(rib_eval.shape[0] - 1):
        nocc_i = np.sum(rib_eval[i, :] < 0.0)
        nocc_ip = np.sum(rib_eval[i + 1, :] < 0.0)
        if nocc_i != nocc_ip:
            jump_k.append(i)

    pos_exps = []
    for i in range(rib_evec.shape[0]):
        pos_exps.append(ribbon_model.position_expectation(rib_evec[i, :], pos_dir=1))

    hwfcs = []
    for i in range(rib_evec.shape[0]):
        occ_evec = rib_evec[i, rib_eval[i, :] < 0.0]
        hwfcs.append(ribbon_model.position_hwf(occ_evec, pos_dir=1))

    hwfcs = np.array(hwfcs, dtype=object)
    rib_eval = rib_eval.T  # transpose in v2
    jump_k = np.array(jump_k, dtype=object)
    pos_exps = np.array(pos_exps, dtype=object)
    phi_1 = np.array(phi_1, dtype=object)

    return phi_1, rib_eval, jump_k, pos_exps, hwfcs
