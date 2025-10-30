import numpy as np
from pythtb import WFArray
from pythtb.models import kane_mele

def get_kane_mele(topological):
    "Return a Kane-Mele model in the normal or topological phase."

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

    ret_model = kane_mele(esite, thop, spin_orb, rashba)

    return ret_model


def run():
    evals_list = []
    wan_cent_list = []
    for top_index in ["even", "odd"]:

        my_model = get_kane_mele(top_index)
        my_array = WFArray(my_model, [41, 41])
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

        evals = my_model.solve_ham(k_vec)
        evals = evals.T # transpose for v2
        evals_list.append(evals)

        wan_cent = my_array.berry_phase([0, 1], dir=1, contin=False, berry_evals=True) 
        # NOTE: the wan_cent must be sorted to match v1 output. This is not an intedended 
        # feature, because the bands of phases switches discontinuously, instead of simply
        # shifting by 2pi.
        wan_cent = np.sort(wan_cent)

        wan_cent /= 2.0 * np.pi
        wan_cent_list.append(wan_cent)

    return np.array(evals_list), np.array(wan_cent_list)

