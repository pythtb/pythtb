import numpy as np
from pythtb import TBModel, WFArray, Lattice, Mesh


def three_site_chain(t, delta):
    lat = [[1.0]]
    orb = [[0.0], [1.0 / 3.0], [2.0 / 3.0]]
    lattice = Lattice(lat, orb, periodic_dirs=[0])
    model = TBModel(lattice=lattice, spinful=False)
    model.set_hop(t, 0, 1, [0])
    model.set_hop(t, 1, 2, [0])
    model.set_hop(t, 2, 0, [1])

    onsite = [
        lambda lam: delta * -np.cos(2 * np.pi * (lam - 0 / 3)),
        lambda lam: delta * -np.cos(2 * np.pi * (lam - 1 / 3)),
        lambda lam: delta * -np.cos(2 * np.pi * (lam - 2 / 3)),
    ]

    model.set_onsite(onsite)
    return model


def run(t, delta):
    mesh = Mesh(
        dim_k=1,
        axis_types=[
            "k",
            "l",
        ],  # first axis: crystal momentum; second: adiabatic parameter
        axis_names=["kx", "lam"],
    )

    mesh.build_grid(
        shape=(31, 21),
        gamma_centered=True,
        k_endpoints=True,
        lambda_start=0.0,
        lambda_stop=1.0,
        lambda_endpoints=True,
    )
    mesh.loop(
        axis_idx=1, component_idx=1, closed=True
    )  # form the lambda axis into a loop

    my_model = three_site_chain(t, delta)

    wfa = WFArray(my_model.lattice, mesh)
    wfa.solve_model(my_model)

    phase = wfa.berry_phase(axis_idx=0, state_idx=[0])
    wann_center = phase / (2.0 * np.pi)
    final = np.sum(wfa.berry_flux(state_idx=[0], plane=(0, 1)))

    return wann_center, final
