import numpy as np
from pythtb import tb_model, wf_array


def get_kane_mele(topological):
    "Return a Kane-Mele model in the normal or topological phase."

    # define lattice vectors
    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    # define coordinates of orbitals
    orb = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]

    # make two dimensional tight-binding Kane-Mele model
    ret_model = tb_model(2, 2, lat, orb, nspin=2)

    # set model parameters depending on whether you are in the topological
    # phase or not
    if topological == "even":
        esite = 2.5
    elif topological == "odd":
        esite = 1.0
    # set other parameters of the model
    thop = 1.0
    spin_orb = 0.6 * thop * 0.5
    rashba = 0.25 * thop

    # set on-site energies
    ret_model.set_onsite([esite, (-1.0) * esite])

    # set hoppings (one for each connected pair of orbitals)
    # (amplitude, i, j, [lattice vector to cell containing j])

    # useful definitions
    sigma_x = np.array([0.0, 1.0, 0.0, 0])
    sigma_y = np.array([0.0, 0.0, 1.0, 0])
    sigma_z = np.array([0.0, 0.0, 0.0, 1])

    # spin-independent first-neighbor hoppings
    ret_model.set_hop(thop, 0, 1, [0, 0])
    ret_model.set_hop(thop, 0, 1, [0, -1])
    ret_model.set_hop(thop, 0, 1, [-1, 0])

    # second-neighbour spin-orbit hoppings (s_z)
    ret_model.set_hop(-1.0j * spin_orb * sigma_z, 0, 0, [0, 1])
    ret_model.set_hop(1.0j * spin_orb * sigma_z, 0, 0, [1, 0])
    ret_model.set_hop(-1.0j * spin_orb * sigma_z, 0, 0, [1, -1])
    ret_model.set_hop(1.0j * spin_orb * sigma_z, 1, 1, [0, 1])
    ret_model.set_hop(-1.0j * spin_orb * sigma_z, 1, 1, [1, 0])
    ret_model.set_hop(1.0j * spin_orb * sigma_z, 1, 1, [1, -1])

    # Rashba first-neighbor hoppings: (s_x)(dy)-(s_y)(d_x)
    r3h = np.sqrt(3.0) / 2.0
    # bond unit vectors are (r3h,half) then (0,-1) then (-r3h,half)
    ret_model.set_hop(
        1.0j * rashba * (0.5 * sigma_x - r3h * sigma_y), 0, 1, [0, 0], mode="add"
    )
    ret_model.set_hop(1.0j * rashba * (-1.0 * sigma_x), 0, 1, [0, -1], mode="add")
    ret_model.set_hop(
        1.0j * rashba * (0.5 * sigma_x + r3h * sigma_y), 0, 1, [-1, 0], mode="add"
    )

    return ret_model


def run():
    evals_list = []
    wan_cent_list = []
    for top_index in ["even", "odd"]:

        my_model = get_kane_mele(top_index)
        my_array = wf_array(my_model, [41, 41])
        my_array.solve_on_grid([-0.5, -0.5])
        
        # [Gamma, K, M, K', Gamma] path in the BZ
        path = [
            [0, 0],
            [2/3, 1/3],
            [1/2, 1/2],
            [1/3, 2/3],
            [0, 0],
        ]
        k_vec, _, _ = my_model.k_path(path, 101, report=False)

        evals = my_model.solve_all(k_vec)
        evals_list.append(evals)

        wan_cent = my_array.berry_phase([0, 1], dir=1, contin=False, berry_evals=True)
        wan_cent /= 2.0 * np.pi
        wan_cent_list.append(wan_cent)

    return np.array(evals_list), np.array(wan_cent_list)

