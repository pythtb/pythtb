from pythtb import Lattice, TBModel


def buckled_model():
    "Return a buckled layer model on a rectangular lattice."

    lat = [[1.0, 0.0, 0.0], [0.0, 1.25, 0.0], [0.0, 0.0, 3.0]]
    orb = [[0.0, 0.0, -0.15], [0.5, 0.5, 0.15]]

    lattice = Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=[0, 1])
    model = TBModel(lattice=lattice, spinful=False)

    return model


def run():
    my_model = buckled_model()

    delta = 1.1
    t = 0.6

    my_model.set_onsite([-delta, delta])
    my_model.set_hop(t, 1, 0, [0, 0, 0])
    my_model.set_hop(t, 1, 0, [1, 0, 0])
    my_model.set_hop(t, 1, 0, [0, 1, 0])
    my_model.set_hop(t, 1, 0, [1, 1, 0])

    # path: [Gamma, X, M, Gamma]
    path = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.5], [0.0, 0.0]]
    k_vec, _, _ = my_model.k_path(path, 81)

    evals = my_model.solve_ham(k_vec)

    return evals
