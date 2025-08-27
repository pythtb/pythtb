import sys

sys.path.append("../")

from pythtb import tb_model, wf_array
from tests.test_tbmodel.test_dimr_dimk_different import generic_test_of_models


def test_answer():

    # these are three equivalent models that should ideally behave in an equivalent way

    lat = [[3.0, 0.1, 0.4], [0.1, 3.1, 1.2], [0.8, 0.2, 3.5]]
    orb = [
        [0.3, 0.1, 0.2],
        [0.3, 0.1, 0.2],
        [0.1, 0.8, 0.3],
        [0.1, 0.8, 0.3],
        [0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4],
    ]
    model0 = tb_model(2, 3, lat, orb, nspin=1, per=[0, 2])
    model0.set_onsite([-2.3, -2.3, 0.5, 0.5, 0.1, 0.1])
    #
    model0.set_hop(0.11 + 0.41, 0 * 2 + 0, 1 * 2 + 0, [1, 0, 2])
    model0.set_hop(0.11 - 0.41, 0 * 2 + 1, 1 * 2 + 1, [1, 0, 2])
    model0.set_hop(0.21 - 0.31j, 0 * 2 + 0, 1 * 2 + 1, [1, 0, 2])
    model0.set_hop(0.21 + 0.31j, 0 * 2 + 1, 1 * 2 + 0, [1, 0, 2])
    #
    model0.set_hop(0.42, 0 * 2 + 0, 1 * 2 + 0, [3, 0, 2])
    model0.set_hop(0.42, 0 * 2 + 1, 1 * 2 + 1, [3, 0, 2])
    #
    model0.set_hop(-0.12, 1 * 2 + 0, 2 * 2 + 0, [2, 0, 3])
    model0.set_hop(-0.12, 1 * 2 + 1, 2 * 2 + 1, [2, 0, 3])
    #
    model0.set_hop(-0.34 + 0.29, 2 * 2 + 0, 0 * 2 + 0, [-1, 0, 2])
    model0.set_hop(-0.34 - 0.29, 2 * 2 + 1, 0 * 2 + 1, [-1, 0, 2])
    model0.set_hop(0.21 + 0.14j, 2 * 2 + 0, 0 * 2 + 1, [-1, 0, 2])
    model0.set_hop(0.21 - 0.14j, 2 * 2 + 1, 0 * 2 + 0, [-1, 0, 2])

    lat = [[3.0, 0.4], [0.8, 3.5]]
    orb = [[0.3, 0.2], [0.1, 0.3], [0.2, 0.4]]
    model1 = tb_model(2, 2, lat, orb, nspin=2)
    model1.set_onsite([-2.3, 0.5, 0.1])
    model1.set_hop(
        [[0.11 + 0.41, 0.21 - 0.31j], [0.21 + 0.31j, 0.11 - 0.41]], 0, 1, [1, 2]
    )
    model1.set_hop(0.42, 0, 1, [3, 2])
    model1.set_hop(-0.12, 1, 2, [2, 3])
    model1.set_hop([-0.34, 0.21, -0.14, 0.29], 2, 0, [-1, 2])

    generic_test_of_models([model0, model1], use_dir=[2, 1], use_occ=[[0, 1], [0, 1]])
