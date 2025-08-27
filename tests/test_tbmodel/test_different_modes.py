import sys

sys.path.append("../")

from pythtb import tb_model

from tests.test_tbmodel.test_dimr_dimk_different import generic_test_of_models


def test_answer():

    lat = [[3.0, 0.1, 0.4], [0.1, 3.1, 1.2], [0.8, 0.2, 3.5]]
    orb = [[0.3, 0.1, 0.2], [0.1, 0.8, 0.3], [0.2, 0.3, 0.4]]
    model0 = tb_model(2, 3, lat, orb, per=[0, 2])
    model0.set_onsite([-2.3, 0.5, 0.1])
    model0.set_hop(0.24, 0, 1, [1, 0, 2], mode="set")
    model0.set_hop(0.42, 0, 1, [3, 0, 2])
    model0.set_hop(-0.12, 1, 2, [2, 0, 3])
    model0.set_hop(-0.34 + 0.3j, 2, 0, [-1, 0, 2])

    lat = [[3.0, 0.4], [0.8, 3.5]]
    orb = [[0.3, 0.2], [0.1, 0.3], [0.2, 0.4]]
    model1 = tb_model(2, 2, lat, orb)
    model1.set_onsite(-2.3, 0)
    model1.set_onsite(0.5, 1)
    model1.set_onsite(9.1, 2, mode="reset")
    model1.set_onsite(0.07, 2, mode="reset")
    model1.set_onsite(0.03, 2, mode="add")
    model1.set_hop(99.24, 0, 1, [1, 2], mode="set")
    model1.set_hop(0.04, 0, 1, [1, 2], mode="reset")
    model1.set_hop(0.08, 0, 1, [1, 2], mode="add")
    model1.set_hop(0.12, 0, 1, [1, 2], mode="add")
    model1.set_hop(0.42, 0, 1, [3, 2])
    model1.set_hop(-0.12, 1, 2, [2, 3])
    model1.set_hop(-0.34 + 0.3j, 2, 0, [-1, 2])

    lat = [[3.0, 0.1, 0.4], [0.8, 0.2, 3.5], [-0.1, -3.1, -1.2]]
    orb = [[0.3, 0.2, 0.1], [0.1, 0.3, 0.8], [0.2, 0.4, 0.3]]
    model2 = tb_model(2, 3, lat, orb, per=[0, 1])
    model2.set_onsite([-2.3, 0.5, 0.1])
    model2.set_hop(0.24, 0, 1, [1, 2, 0])
    model2.set_hop(99.42, 0, 1, [3, 2, 0], mode="reset")
    model2.set_hop(0.42, 0, 1, [3, 2, 0], mode="reset")
    model2.set_hop(-0.12, 1, 2, [2, 3, 0])
    model2.set_hop((-0.34 + 0.3j) * 0.7, 2, 0, [-1, 2, 0], allow_conjugate_pair=True)
    model2.set_hop((-0.34 - 0.3j) * 0.3, 0, 2, [1, -2, 0], allow_conjugate_pair=True)

    generic_test_of_models(
        [model0, model1, model2], use_dir=[2, 1, 1], use_occ=[[0], [0], [0]]
    )
