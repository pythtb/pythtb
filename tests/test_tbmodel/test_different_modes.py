from pythtb import TBModel, Lattice
from generic_run import generic_test_of_models


def test_modes():
    # 3D model, with 2D k-space along first and third dimension
    lat = [[3.0, 0.1, 0.4], [0.1, 3.1, 1.2], [0.8, 0.2, 3.5]]
    orb = [[0.3, 0.1, 0.2], [0.1, 0.8, 0.3], [0.2, 0.3, 0.4]]
    lattice = Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=[0, 2])
    model0 = TBModel(lattice=lattice, spinful=False)
    model0.set_onsite([-2.3, 0.5, 0.1])
    model0.set_hop(0.24, 0, 1, [1, 0, 2], mode="set")
    model0.set_hop(0.42, 0, 1, [3, 0, 2], mode="set")
    model0.set_hop(-0.12, 1, 2, [2, 0, 3], mode="set")
    model0.set_hop(-0.34 + 0.3j, 2, 0, [-1, 0, 2], mode="set")

    # 2D model with 2D k-space
    lat = [[3.0, 0.4], [0.8, 3.5]]
    orb = [[0.3, 0.2], [0.1, 0.3], [0.2, 0.4]]
    lattice = Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=[0, 1])
    model1 = TBModel(lattice=lattice, spinful=False)
    # set proper onsite
    model1.set_onsite(-2.3, 0)
    # set proper onsite
    model1.set_onsite(0.5, 1)
    # dummy set onsite
    model1.set_onsite(9.1, 2, mode="set")
    # reset onsite
    model1.set_onsite(0.07, 2, mode="set")
    # add to get proper value
    model1.set_onsite(0.03, 2, mode="add")
    # dummy set hop
    model1.set_hop(99.24, 0, 1, [1, 2], mode="set")
    # reset hop
    model1.set_hop(0.04, 0, 1, [1, 2], mode="set")
    # add to get proper value
    model1.set_hop(0.08, 0, 1, [1, 2], mode="add")
    # add to get proper value
    model1.set_hop(0.12, 0, 1, [1, 2], mode="add")
    # set proper value
    model1.set_hop(0.42, 0, 1, [3, 2])
    # set proper value
    model1.set_hop(-0.12, 1, 2, [2, 3])
    # set proper value
    model1.set_hop(-0.34 + 0.3j, 2, 0, [-1, 2])

    # 3D model with 2D k-space along first and second dimension
    lat = [[3.0, 0.1, 0.4], [0.8, 0.2, 3.5], [-0.1, -3.1, -1.2]]
    orb = [[0.3, 0.2, 0.1], [0.1, 0.3, 0.8], [0.2, 0.4, 0.3]]
    lattice = Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=[0, 1])
    model2 = TBModel(lattice=lattice, spinful=False)
    # set proper onsite
    model2.set_onsite([-2.3, 0.5, 0.1])
    # set proper hop
    model2.set_hop(0.24, 0, 1, [1, 2, 0])
    # dummy set hop
    model2.set_hop(99.42, 0, 1, [3, 2, 0], mode="set")
    # reset hop to proper value
    model2.set_hop(0.42, 0, 1, [3, 2, 0], mode="set")
    # set proper hop
    model2.set_hop(-0.12, 1, 2, [2, 3, 0])

    # set proper hop manually along reciprocal directions
    model2.set_hop((-0.34 + 0.3j) * 0.7, 2, 0, [-1, 2, 0], allow_conjugate_pair=True)
    model2.set_hop((-0.34 - 0.3j) * 0.3, 0, 2, [1, -2, 0], allow_conjugate_pair=True)

    print("model0:", model0)
    print("model1:", model1)
    print("model2:", model2)

    generic_test_of_models(
        [model0, model1, model2], use_dir=[2, 1, 1], use_occ=[[0], [0], [0]]
    )
