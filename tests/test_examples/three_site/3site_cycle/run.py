import numpy as np
from pythtb import tb_model, wf_array

def three_site_chain(t, delta, lmbd):
    lat = [[1.0]]
    orb = [[0.0], [1.0 / 3.0], [2.0 / 3.0]]
    model = tb_model(1, 1, lat, orb)
    model.set_hop(t, 0, 1, [0])
    model.set_hop(t, 1, 2, [0])
    model.set_hop(t, 2, 0, [1])
    onsite = [
        delta * -np.cos(2.0 * np.pi * (lmbd - i / 3.0))
        for i in range(3)
    ]
    model.set_onsite(onsite)
    return model

def run(t, delta):
    path_steps = 21
    num_kpt = 31
    all_lambda = np.linspace(0, 1, path_steps, endpoint=True)

    my_model = three_site_chain(t, delta, 0.0)
    wf_kpt_lambda = wf_array(my_model, [num_kpt, path_steps])
    for i_lambda in range(path_steps):
        lmbd = all_lambda[i_lambda]
        my_model = three_site_chain(t, delta, lmbd)
        
        k_vec, _, _ = my_model.k_path([[-0.5], [0.5]], num_kpt, report=False)
        _, evec = my_model.solve_all(k_vec, eig_vectors=True)
        for i_kpt in range(num_kpt):
            wf_kpt_lambda[i_kpt, i_lambda] = evec[:, i_kpt, :]

    wf_kpt_lambda.impose_pbc(0, 0)
    phase = wf_kpt_lambda.berry_phase([0], 0)
    wann_center = phase / (2.0 * np.pi)
    final = wf_kpt_lambda.berry_flux([0])

    return wann_center, final
