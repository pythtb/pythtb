import numpy as np
from pythtb import TBModel, WFArray, Lattice, Mesh


def haldane_model():
    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    orb = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]
    lattice = Lattice(lat, orb, periodic_dirs=[0, 1])

    my_model = TBModel(lattice=lattice)

    delta = 0.0
    t = -1.0
    t2 = 0.15 * np.exp((1.0j) * np.pi / 2.0)
    t2c = t2.conjugate()

    my_model.set_onsite([-delta, delta])

    my_model.set_hop(t, 0, 1, [0, 0])
    my_model.set_hop(t, 1, 0, [1, 0])
    my_model.set_hop(t, 1, 0, [0, 1])
    my_model.set_hop(t2, 0, 0, [1, 0])
    my_model.set_hop(t2, 1, 1, [1, -1])
    my_model.set_hop(t2, 1, 1, [0, 1])
    my_model.set_hop(t2c, 1, 1, [1, 0])
    my_model.set_hop(t2c, 0, 0, [1, -1])
    my_model.set_hop(t2c, 0, 0, [0, 1])
    return my_model


def run():
    my_model = haldane_model()

    # approach #1
    mesh = Mesh(dim_k=2, axis_types=["k", "k"])
    mesh.build_grid(shape=(31, 31), k_endpoints=True, gamma_centered=True)
    my_array_1 = WFArray(my_model.lattice, mesh)
    my_array_1.solve_model(my_model)

    phi_a_1 = my_array_1.berry_phase(state_idx=[0], axis_idx=0, contin=True)
    phi_b_1 = my_array_1.berry_phase(state_idx=[1], axis_idx=0, contin=True)
    phi_c_1 = my_array_1.berry_phase(state_idx=[0, 1], axis_idx=0, contin=True)
    flux_a_1 = np.sum(my_array_1.berry_flux(state_idx=[0], plane=(0, 1)))

    # plot Berry phases
    ky = np.linspace(0.0, 1.0, len(phi_a_1))

    nkx = 31
    nky = 31
    kx = np.linspace(-0.5, 0.5, num=nkx)
    ky = np.linspace(-0.5, 0.5, num=nky)

    mesh = Mesh(dim_k=2, axis_types=["k", "k"])
    mesh.build_grid(shape=(nkx, nky), k_endpoints=True, gamma_centered=True)
    my_array_2 = WFArray(my_model.lattice, mesh)
    for i in range(nkx):
        for j in range(nky):
            (eval, evec) = my_model.solve_ham([kx[i], ky[j]], return_eigvecs=True)
            my_array_2[i, j] = evec
    # my_array_2.impose_pbc(0, 0)
    # my_array_2.impose_pbc(1, 1)

    flux_a_2 = np.sum(my_array_2.berry_flux(state_idx=[0], plane=(0, 1)))

    return phi_a_1, phi_b_1, phi_c_1, flux_a_1, flux_a_2
