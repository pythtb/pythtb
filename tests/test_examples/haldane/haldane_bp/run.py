import numpy as np
from pythtb import tb_model, wf_array

def haldane_model():

    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    orb = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]

    my_model = tb_model(2, 2, lat, orb)

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
    my_array_1 = wf_array(my_model, [31, 31])
    my_array_1.solve_on_grid([-0.5, -0.5])

    phi_a_1 = my_array_1.berry_phase([0], 0, contin=True)
    phi_b_1 = my_array_1.berry_phase([1], 0, contin=True)
    phi_c_1 = my_array_1.berry_phase([0, 1], 0, contin=True)
    flux_a_1 = my_array_1.berry_flux([0])

    # plot Berry phases
    ky = np.linspace(0.0, 1.0, len(phi_a_1))

    nkx = 31
    nky = 31
    kx = np.linspace(-0.5, 0.5, num=nkx)
    ky = np.linspace(-0.5, 0.5, num=nky)
    my_array_2 = wf_array(my_model, [nkx, nky])
    for i in range(nkx):
        for j in range(nky):
            (eval, evec) = my_model.solve_one([kx[i], ky[j]], eig_vectors=True)
            my_array_2[i, j] = evec
    my_array_2.impose_pbc(0, 0)
    my_array_2.impose_pbc(1, 1)

    flux_a_2 = my_array_2.berry_flux([0])

    return phi_a_1, phi_b_1, phi_c_1, flux_a_1, flux_a_2

