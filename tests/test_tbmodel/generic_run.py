import numpy as np
from pythtb import TBModel, WFArray, Mesh


def generic_test_of_models(models: list[TBModel], use_dir, use_occ):
    mesh = Mesh(dim_k=2, axis_types=["k", "k"])
    mesh.build_grid([11, 11])

    # check that berry phases are the same
    val = []
    for ii, mod in enumerate(models):
        my_array = WFArray(mod.lattice, mesh, spinful=mod.spinful)
        my_array.solve_model(mod)
        val.append(my_array.berry_phase(1, use_occ[ii], contin=True))
    val = np.array(val)
    passed = []
    for i in range(1, val.shape[0]):
        passed.append(np.isclose(val[0], val[i]))
    passed = np.array(passed)
    assert np.all(passed)

    # check that energies are the same at some random point
    val = []
    for ii, mod in enumerate(models):
        val.append(mod.solve_ham([0.123, 0.523]))
    val = np.array(val)
    passed = []
    for i in range(1, val.shape[0]):
        passed.append(np.isclose(val[0], val[i]))
    passed = np.array(passed)
    assert np.all(passed)

    # check finitely cut models
    val = []
    H = []
    evecs = []
    for ii, mod in enumerate(models):
        mod_cut = mod.cut_piece(4, use_dir[ii], glue_edges=False)
        H.append(mod_cut.hamiltonian([0.214]))
        _, evec = mod_cut.solve_ham(
            [0.214], return_eigvecs=True, flatten_spin_axis=False
        )
        evecs.append(evec)
        print(evec.shape)
        val.append(mod_cut.position_expectation(evec, use_dir[ii]))

    val = np.array(val)
    passed = []
    for i in range(1, val.shape[0]):
        # only sum is multi-band gauge invariant (there are degeneracies)
        passed.append(np.isclose(np.sum(val[0]), np.sum(val[i])))

    passed = np.array(passed)
    assert np.all(passed)
