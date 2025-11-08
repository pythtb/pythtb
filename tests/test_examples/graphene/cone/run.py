import numpy as np
from pythtb import TBModel, WFArray, Lattice, Mesh


def graphene_model() -> TBModel:
    "Return a graphene-like model on a triangular lattice."

    lat = [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]]
    orb = [[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]
    lattice = Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=[0, 1])
    my_model = TBModel(lattice=lattice, spinful=False)

    return my_model


def run():
    my_model: TBModel = graphene_model()

    delta = -0.1
    t = -1.0

    my_model.set_onsite([-delta, delta])
    my_model.set_hop(t, 0, 1, [0, 0])
    my_model.set_hop(t, 1, 0, [1, 0])
    my_model.set_hop(t, 1, 0, [0, 1])

    circ_step = 31
    circ_center = np.array([1.0 / 3.0, 2.0 / 3.0])
    circ_radius = 0.05

    # construct k-point coordinate on the path
    kpts = []
    for i in range(circ_step):
        ang = 2 * np.pi * i / (circ_step - 1)
        kpt = np.array([np.cos(ang) * circ_radius, np.sin(ang) * circ_radius])
        kpt += circ_center
        kpts.append(kpt)
    kpts = np.array(kpts)

    mesh = Mesh(dim_k=2, axis_types=["k"])
    mesh.build_custom(kpts)
    w_circ = WFArray(my_model.lattice, mesh)
    w_circ.solve_model(my_model)

    berry_phase_circ_0 = w_circ.berry_phase(0, [0])
    berry_phase_circ_1 = w_circ.berry_phase(0, [1])
    berry_phase_circ_01 = w_circ.berry_phase(0, [0, 1])

    square_step = 31
    square_center = np.array([1.0 / 3.0, 2.0 / 3.0])
    square_length = 0.1

    all_kpt = np.zeros((square_step, square_step, 2))
    for i in range(square_step):
        for j in range(square_step):
            kpt = np.array(
                [
                    square_length * (-0.5 + i / (square_step - 1)),
                    square_length * (-0.5 + j / (square_step - 1)),
                ]
            )
            kpt += square_center
            all_kpt[i, j, :] = kpt

    mesh = Mesh(dim_k=2, axis_types=["k", "k"])
    mesh.build_custom(points=all_kpt)

    w_square = WFArray(my_model.lattice, mesh)
    w_square.solve_model(my_model)

    berr_flux_square_0 = np.sum(w_square.berry_flux(state_idx=[0], plane=(0, 1)))
    berr_flux_square_1 = np.sum(w_square.berry_flux(state_idx=[1], plane=(0, 1)))
    berr_flux_square_01 = np.sum(w_square.berry_flux(state_idx=[0, 1], plane=(0, 1)))
    plaq = w_square.berry_flux(state_idx=[0], plane=(0, 1))

    return (
        berry_phase_circ_0,
        berry_phase_circ_1,
        berry_phase_circ_01,
        berr_flux_square_0,
        berr_flux_square_1,
        berr_flux_square_01,
        plaq,
    )
