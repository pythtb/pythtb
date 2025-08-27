import numpy as np
from pythtb import tb_model, wf_array

def haldane_model():
    
    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    orb = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]

    my_model = tb_model(2, 2, lat, orb)

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

    my_array = wf_array(my_model, [len_0, len_1])
    my_array.solve_on_grid([0.0, 0.0])
    phi_1 = my_array.berry_phase(occ=[0], dir=1, contin=True)

    ribbon_model = my_model.cut_piece(len_1, fin_dir=1, glue_edgs=False)
    (k_vec, k_dist, k_node) = ribbon_model.k_path([0.0, 0.5, 1.0], len_0, report=False)
    rib_eval, rib_evec = ribbon_model.solve_all(k_vec, eig_vectors=True)
    rib_eval -= efermi

    jump_k = []
    for i in range(rib_eval.shape[1] - 1):
        nocc_i = np.sum(rib_eval[:, i] < 0.0)
        nocc_ip = np.sum(rib_eval[:, i + 1] < 0.0)
        if nocc_i != nocc_ip:
            jump_k.append(i)

    pos_exps = []
    for i in range(rib_evec.shape[1]):
        pos_exps.append(ribbon_model.position_expectation(rib_evec[:, i], dir=1))

    hwfcs = []
    for i in range(rib_evec.shape[1]):
        occ_evec = rib_evec[rib_eval[:, i] < 0.0, i]
        hwfcs.append(ribbon_model.position_hwf(occ_evec, 1))

    hwfcs = np.array(hwfcs, dtype=object)
    # rib_eval = np.array(rib_eval, dtype=object)
    jump_k = np.array(jump_k, dtype=object)
    pos_exps = np.array(pos_exps, dtype=object)
    phi_1 = np.array(phi_1, dtype=object)

    return phi_1, rib_eval, jump_k, pos_exps, hwfcs

       

