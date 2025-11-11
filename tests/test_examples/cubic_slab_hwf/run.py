import numpy as np
from pythtb import TBModel, WFArray, Lattice, Mesh


def set_model(delta, ta, tb):
    lat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    orb = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    lattice = Lattice(lat, orb, periodic_dirs=[0, 1, 2])
    model = TBModel(lattice)
    model.set_onsite([-delta, delta])
    for lvec in ([-1, 0, 0], [0, 0, -1], [-1, -1, 0], [0, -1, -1]):
        model.set_hop(ta, 0, 1, lvec)
    for lvec in ([0, 0, 0], [0, -1, 0], [-1, -1, -1], [-1, 0, -1]):
        model.set_hop(tb, 0, 1, lvec)

    return model


def run():
    delta = 1.0
    ta = 0.4
    tb = 0.7
    bulk_model = set_model(delta, ta, tb)

    nl = 9
    slab_model = bulk_model.cut_piece(nl, 2, glue_edges=False)
    slab_model.remove_orb(2 * nl - 1)

    nk = 10
    k_1d = np.linspace(0.0, 1.0, nk, endpoint=False)
    kpts = []
    for kx in k_1d:
        for ky in k_1d:
            kpts.append([kx, ky])

    evals = slab_model.solve_ham(kpts)

    nk = 9
    mesh = Mesh(dim_k=2, axis_types=["k", "k"])
    mesh.build_grid(shape=(nk, nk), k_endpoints=True)
    bloch_arr = WFArray(slab_model.lattice, mesh)
    bloch_arr.solve_model(slab_model)

    hwf_arr = bloch_arr.empty_like(nstates=nl)
    hwfc = np.zeros([nk, nk, nl])

    for ix in range(nk):
        for iy in range(nk):
            (val, vec) = bloch_arr.position_hwf(
                mesh_idx=[ix, iy],
                state_idx=list(range(nl)),
                pos_dir=2,
                hwf_evec=True,
                basis="orbital",
            )
            hwfc[ix, iy] = val
            hwf_arr[ix, iy] = vec

    # compute and print layer contributions to polarization along x, then y
    px = np.zeros((nl, nk))
    for n in range(nl):
        px[n, :] = hwf_arr.berry_phase(axis_idx=0, state_idx=[n]) / (2.0 * np.pi)

    evals = evals.T  # transpose for v2

    return evals, hwfc, px
