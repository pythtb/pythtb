import sys

sys.path.append("../")

from pythtb import tb_model, wf_array
import numpy as np

def test_answer():

    # these are three equivalent models that should ideally behave in an equivalent way

    lat = [[3.0, 0.1, 0.4], [0.1, 3.1, 1.2], [0.8, 0.2, 3.5]]
    orb = [[0.3, 0.1, 0.2], [0.1, 0.8, 0.3], [0.2, 0.3, 0.4]]
    model0 = tb_model(2, 3, lat, orb, per=[0, 2])
    model0.set_onsite([-2.3, 0.5, 0.1])
    model0.set_hop(0.24, 0, 1, [1, 0, 2])
    model0.set_hop(0.42, 0, 1, [3, 0, 2])
    model0.set_hop(-0.12, 1, 2, [2, 0, 3])
    model0.set_hop(-0.34, 2, 0, [-1, 0, 2])

    lat = [[3.0, 0.4], [0.8, 3.5]]
    orb = [[0.3, 0.2], [0.1, 0.3], [0.2, 0.4]]
    model1 = tb_model(2, 2, lat, orb)
    model1.set_onsite([-2.3, 0.5, 0.1])
    model1.set_hop(0.24, 0, 1, [1, 2])
    model1.set_hop(0.42, 0, 1, [3, 2])
    model1.set_hop(-0.12, 1, 2, [2, 3])
    model1.set_hop(-0.34, 2, 0, [-1, 2])

    lat = [[3.0, 0.1, 0.4], [0.8, 0.2, 3.5], [-0.1, -3.1, -1.2]]
    orb = [[0.3, 0.2, 0.1], [0.1, 0.3, 0.8], [0.2, 0.4, 0.3]]
    model2 = tb_model(2, 3, lat, orb, per=[0, 1])
    model2.set_onsite([-2.3, 0.5, 0.1])
    model2.set_hop(0.24, 0, 1, [1, 2, 0])
    model2.set_hop(0.42, 0, 1, [3, 2, 0])
    model2.set_hop(-0.12, 1, 2, [2, 3, 0])
    model2.set_hop(-0.34, 2, 0, [-1, 2, 0])

    generic_test_of_models(
        [model0, model1, model2], use_dir=[2, 1, 1], use_occ=[[0], [0], [0]]
    )


def generic_test_of_models(models: list[tb_model], use_dir, use_occ):

    # check that berry phases are the same
    val = []
    for ii, mod in enumerate(models):
        my_array = wf_array(mod, [11, 11])
        my_array.solve_on_grid([-0.5, -0.5])
        val.append(my_array.berry_phase(use_occ[ii], 1, contin=True))
    val = np.array(val)
    passed = []
    for i in range(1, val.shape[0]):
        passed.append(np.isclose(val[0], val[i]))
    passed = np.array(passed)
    assert np.all(passed)

    # check that energies are the same at some random point
    val = []
    for ii, mod in enumerate(models):
        val.append(mod.solve_one([0.123, 0.523]))
    val = np.array(val)
    passed = []
    for i in range(1, val.shape[0]):
        passed.append(np.isclose(val[0], val[i]))
    passed = np.array(passed)
    assert np.all(passed)

    # check finitely cut models
    val = []
    for ii, mod in enumerate(models):
        mod_cut = mod.cut_piece(4, use_dir[ii], glue_edgs=False)
        evalu, evec = mod_cut.solve_one([0.214], eig_vectors=True)
        print(evec.shape)
        val.append(mod_cut.position_expectation(evec, use_dir[ii]))
    val = np.array(val)
    passed = []
    for i in range(1, val.shape[0]):
        print(val[0], val[i])
        passed.append(np.isclose(val[0], val[i]))
    passed = np.array(passed)
    assert np.all(passed)
