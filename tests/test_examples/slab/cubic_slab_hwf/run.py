import numpy as np
from pythtb import tb_model, wf_array

def set_model(delta, ta, tb):
    lat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    orb = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    model = tb_model(3, 3, lat, orb)
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
    slab_model = bulk_model.cut_piece(nl, 2, glue_edgs=False)
    slab_model = slab_model.remove_orb(2 * nl - 1)

    nk = 10
    k_1d = np.linspace(0.0, 1.0, nk, endpoint=False)
    kpts = []
    for kx in k_1d:
        for ky in k_1d:
            kpts.append([kx, ky])

    evals = slab_model.solve_all(kpts)
    e_vb = evals[:, :nl]
    e_cb = evals[:, nl + 1 :]

    nk = 9
    bloch_arr = wf_array(slab_model, [nk, nk])
    bloch_arr.solve_on_grid([0.0, 0.0])
   
    hwf_arr = bloch_arr.empty_like(nsta_arr=nl)
    hwfc = np.zeros([nk, nk, nl])

    for ix in range(nk):
        for iy in range(nk):
            (val, vec) = bloch_arr.position_hwf(
                [ix, iy], occ=list(range(nl)), dir=2, hwf_evec=True, basis="orbital"
            )
            hwfc[ix, iy] = val
            hwf_arr[ix, iy] = vec
    hwf_arr.impose_pbc(0, 0)
    hwf_arr.impose_pbc(1, 1)

    # compute and print layer contributions to polarization along x, then y
    px = np.zeros((nl, nk))
    py = np.zeros((nl, nk))
    for n in range(nl):
        px[n, :] = hwf_arr.berry_phase(dir=0, occ=[n]) / (2.0 * np.pi)

    px_mean = np.mean(px[:, :-1], axis=1)

    nlh = nl // 2
    sum_top = np.sum(px_mean[:nlh])
    sum_bot = np.sum(px_mean[-nlh:])

    return evals, hwfc, px