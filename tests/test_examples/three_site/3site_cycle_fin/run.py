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
    all_lambda = np.linspace(0.0, 1.0, path_steps, endpoint=True)

    my_model = three_site_chain(t, delta, 0.0)
    (k_vec, _, _) = my_model.k_path([[-0.5], [0.5]], num_kpt, report=False)

    wf_kpt_lambda = wf_array(my_model, [path_steps, num_kpt])

    for i_lambda, lmbd in enumerate(all_lambda):
        model = three_site_chain(t, delta, lmbd)
        _, evec = model.solve_all(k_vec, eig_vectors=True)
        for i_kpt in range(num_kpt):
            wf_kpt_lambda[i_lambda, i_kpt] = evec[:,i_kpt,:]

    fluxes = np.array([
        wf_kpt_lambda.berry_flux([0]),
        wf_kpt_lambda.berry_flux([1]),
        wf_kpt_lambda.berry_flux([2]),
        wf_kpt_lambda.berry_flux([0, 1]),
        wf_kpt_lambda.berry_flux([0, 1, 2])
    ])

    path_steps = 241
    all_lambda = np.linspace(0.0, 1.0, path_steps)
    num_cells = 10
    num_orb = 3 * num_cells
    ch_eval = np.zeros((num_orb, path_steps))
    ch_xexp = np.zeros((num_orb, path_steps))

    for i, lmbd in enumerate(all_lambda):
        model = three_site_chain(t, delta, lmbd)
        fin_model = model.cut_piece(num_cells, 0)
        evals, evecs = fin_model.solve_all(eig_vectors=True)
        ch_eval[:, i] = evals
        ch_xexp[:, i] = fin_model.position_expectation(evecs, 0)

    return fluxes, ch_eval, ch_xexp
