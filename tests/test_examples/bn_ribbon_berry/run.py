import numpy as np
from pythtb import Lattice, Mesh, TBModel, WFArray


def bn_model(t, delta):
    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    orb = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]
    lattice = Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=[0, 1])
    my_model = TBModel(lattice=lattice, spinful=False)
    my_model.set_onsite([-delta, delta])
    my_model.set_hop(t, 0, 1, [0, 0])
    my_model.set_hop(t, 1, 0, [1, 0])
    my_model.set_hop(t, 1, 0, [0, 1])
    return my_model


def run():
    t = -1.0
    delta = 0.4

    model_orig = bn_model(t, delta).cut_piece(
        num_cells=3, periodic_dir=1, glue_edges=False
    )

    nk = 40
    n_occ = model_orig.nstate // 2

    mesh = Mesh(["k"])
    mesh.build_grid([nk])

    wfa = WFArray(lattice=model_orig.lattice, mesh=mesh)
    wfa.solve_model(model_orig)
    berry_phase = wfa.berry_phase(axis_idx=0, state_idx=range(n_occ))

    model_orig.change_nonperiodic_vector(1)

    wfa2 = WFArray(lattice=model_orig.lattice, mesh=mesh)
    wfa2.solve_model(model_orig)
    berry_phase2 = wfa2.berry_phase(axis_idx=0, state_idx=range(n_occ))

    return berry_phase, berry_phase2
