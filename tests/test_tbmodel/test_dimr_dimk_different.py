from pythtb import TBModel, Lattice
from generic_run import generic_test_of_models


def test_slab():
    # these are three equivalent models that should ideally behave in an equivalent way

    lat = [[3.0, 0.1, 0.4], [0.1, 3.1, 1.2], [0.8, 0.2, 3.5]]
    orb = [[0.3, 0.1, 0.2], [0.1, 0.8, 0.3], [0.2, 0.3, 0.4]]
    lattice = Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=[0, 2])
    model0 = TBModel(lattice=lattice, spinful=False)
    model0.set_onsite([-2.3, 0.5, 0.1])
    model0.set_hop(0.24, 0, 1, [1, 0, 2])
    model0.set_hop(0.42, 0, 1, [3, 0, 2])
    model0.set_hop(-0.12, 1, 2, [2, 0, 3])
    model0.set_hop(-0.34, 2, 0, [-1, 0, 2])

    lat = [[3.0, 0.4], [0.8, 3.5]]
    orb = [[0.3, 0.2], [0.1, 0.3], [0.2, 0.4]]
    lattice = Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=[0, 1])
    model1 = TBModel(lattice=lattice, spinful=False)
    model1.set_onsite([-2.3, 0.5, 0.1])
    model1.set_hop(0.24, 0, 1, [1, 2])
    model1.set_hop(0.42, 0, 1, [3, 2])
    model1.set_hop(-0.12, 1, 2, [2, 3])
    model1.set_hop(-0.34, 2, 0, [-1, 2])

    lat = [[3.0, 0.1, 0.4], [0.8, 0.2, 3.5], [-0.1, -3.1, -1.2]]
    orb = [[0.3, 0.2, 0.1], [0.1, 0.3, 0.8], [0.2, 0.4, 0.3]]
    lattice = Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=[0, 1])
    model2 = TBModel(lattice=lattice, spinful=False)
    model2.set_onsite([-2.3, 0.5, 0.1])
    model2.set_hop(0.24, 0, 1, [1, 2, 0])
    model2.set_hop(0.42, 0, 1, [3, 2, 0])
    model2.set_hop(-0.12, 1, 2, [2, 3, 0])
    model2.set_hop(-0.34, 2, 0, [-1, 2, 0])

    generic_test_of_models(
        [model0, model1, model2], use_dir=[2, 1, 1], use_occ=[[0], [0], [0]]
    )
