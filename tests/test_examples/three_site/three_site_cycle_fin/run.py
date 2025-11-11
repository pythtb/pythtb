import numpy as np
from pythtb import TBModel, WFArray, Lattice, Mesh


def three_site_chain(t, delta):
    lat = [[1.0]]
    orb = [[0.0], [1.0 / 3.0], [2.0 / 3.0]]
    lattice = Lattice(lat, orb, periodic_dirs=[0])
    model = TBModel(lattice=lattice, spinful=False)
    model.set_hop(t, 0, 1, [0])
    model.set_hop(t, 1, 2, [0])
    model.set_hop(t, 2, 0, [1])

    onsite = [
        lambda lam: delta * -np.cos(2 * np.pi * (lam - 0 / 3)),
        lambda lam: delta * -np.cos(2 * np.pi * (lam - 1 / 3)),
        lambda lam: delta * -np.cos(2 * np.pi * (lam - 2 / 3)),
    ]

    model.set_onsite(onsite)
    return model


def run(t, delta):
    path_steps = 21
    num_kpt = 31
    all_lambda = np.linspace(0.0, 1.0, path_steps, endpoint=True)

    my_model = three_site_chain(t, delta)
    # my_model.set_parameters(lam=0.0)
    (k_vec, _, _) = my_model.k_path([[-0.5], [0.5]], num_kpt, report=False)

    mesh = Mesh(dim_k=1, axis_types=["k", "l"], axis_names=["k_x", "lam"])
    mesh.build_grid(
        shape=(num_kpt, path_steps),
        k_endpoints=True,
        lambda_endpoints=True,
        gamma_centered=True,
    )
    wf_kpt_lambda = WFArray(my_model.lattice, mesh)
    wf_kpt_lambda.solve_model(my_model)

    # for i_lambda, lmbd in enumerate(all_lambda):
    #     model = three_site_chain(t, delta)

    #     _, evec = model.solve_ham(k_vec, return_eigvecs=True)
    #     for i_kpt in range(num_kpt):
    #         wf_kpt_lambda[i_lambda, i_kpt] = evec[i_kpt]

    fluxes = np.array(
        [
            np.sum(wf_kpt_lambda.berry_flux(state_idx=[0], plane=(1, 0))),
            np.sum(wf_kpt_lambda.berry_flux(state_idx=[1], plane=(1, 0))),
            np.sum(wf_kpt_lambda.berry_flux(state_idx=[2], plane=(1, 0))),
            np.sum(wf_kpt_lambda.berry_flux(state_idx=[0, 1], plane=(1, 0))),
            np.sum(wf_kpt_lambda.berry_flux(state_idx=[0, 1, 2], plane=(1, 0))),
        ]
    )

    path_steps = 241
    all_lambda = np.linspace(0.0, 1.0, path_steps)
    num_cells = 10
    num_orb = 3 * num_cells
    ch_eval = np.zeros((num_orb, path_steps))
    ch_xexp = np.zeros((num_orb, path_steps))

    for i, lmbd in enumerate(all_lambda):
        model = three_site_chain(t, delta)
        model.set_parameters(lam=lmbd)
        fin_model = model.cut_piece(num_cells, 0)
        evals, evecs = fin_model.solve_ham(return_eigvecs=True)
        ch_eval[:, i] = evals
        ch_xexp[:, i] = fin_model.position_expectation(evecs, 0)

    return fluxes, ch_eval, ch_xexp
