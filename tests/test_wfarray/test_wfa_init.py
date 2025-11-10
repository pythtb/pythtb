import pytest
from pythtb import TBModel, Mesh, WFArray, Lattice


def test_wfa_initialization():
    # define lattice vectors
    lat = [[1.0, 0.0], [0.0, 1.0]]
    # define coordinates of orbitals
    orb = [[0.0, 0.0], [0.5, 0.5]]
    # make two-dimensional model
    lattice = Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=[0, 1])
    bulk_model = TBModel(lattice=lattice, spinful=False)

    # sampling of Brillouin zone
    numk = 10  # number of k-points along each direction

    # create mesh
    mesh = Mesh(["k", "k"])
    mesh.build_grid([numk, numk])

    # initialize WFArray
    bulk_array = WFArray(bulk_model.lattice, mesh, spinful=bulk_model.spinful)

    # Check properties of the WFArray
    assert bulk_array.lattice == bulk_model.lattice, (
        "Lattice should be set correctly in WFArray"
    )
    assert bulk_array.mesh == mesh, "Mesh should be set correctly in WFArray"
    assert bulk_array.spinful == bulk_model.spinful, (
        "Spinful property should match the TBModel"
    )
    assert bulk_array.naxes == 2, "Number of k-space axes should be 2"
    assert bulk_array.norb == bulk_model.norb, (
        "Number of orbitals should match the TBModel"
    )

    # Check that the wavefunction array is initialized correctly
    expected_shape = (numk, numk, bulk_model.norb, bulk_model.norb)
    assert bulk_array.shape == expected_shape, (
        f"Wavefunction array shape should be {expected_shape}"
    )

    mesh_param = Mesh(["k", "k", "l"])
    mesh_param.build_grid([numk, numk, 5])  # 5 points along the lambda direction
    param_array = WFArray(bulk_model.lattice, mesh_param, spinful=bulk_model.spinful)

    # Check properties of the parameterized WFArray
    assert param_array.lattice == bulk_model.lattice, (
        "Lattice should be set correctly in parameterized WFArray"
    )
    assert param_array.mesh == mesh_param, (
        "Mesh should be set correctly in parameterized WFArray"
    )
    assert param_array.spinful == bulk_model.spinful, (
        "Spinful property should match the TBModel"
    )
    assert param_array.naxes == 3, "Number of k-space axes should be 2"
    assert param_array.norb == bulk_model.norb, (
        "Number of orbitals should match the TBModel"
    )
    expected_shape_param = (numk, numk, 5, bulk_model.norb, bulk_model.norb)
    assert param_array.shape == expected_shape_param, (
        f"Wavefunction array shape should be {expected_shape_param}"
    )


def test_wfa_initialization_spinful():
    # define lattice vectors
    lat = [[1.0, 0.0], [0.0, 1.0]]
    # define coordinates of orbitals
    orb = [[0.0, 0.0], [0.5, 0.5]]
    # make two-dimensional spinful model
    lattice = Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=[0, 1])
    bulk_model = TBModel(lattice=lattice, spinful=True)

    # sampling of Brillouin zone
    numk = 10  # number of k-points along each direction

    # create mesh
    mesh = Mesh(["k", "k"])
    mesh.build_grid([numk, numk])

    # initialize WFArray
    bulk_array = WFArray(bulk_model.lattice, mesh, spinful=bulk_model.spinful)

    # Check properties of the WFArray
    assert bulk_array.lattice == bulk_model.lattice, (
        "Lattice should be set correctly in WFArray"
    )
    assert bulk_array.mesh == mesh, "Mesh should be set correctly in WFArray"
    assert bulk_array.spinful == bulk_model.spinful, (
        "Spinful property should match the TBModel"
    )
    assert bulk_array.naxes == 2, "Number of k-space axes should be 2"
    assert bulk_array.norb == bulk_model.norb, (
        "Number of orbitals should match the TBModel"
    )

    # Check that the wavefunction array is initialized correctly
    expected_shape = (numk, numk, bulk_model.norb * 2, bulk_model.norb, 2)
    assert bulk_array.shape == expected_shape, (
        f"Wavefunction array shape should be {expected_shape}"
    )


def test_wfarray_allows_single_point_lambda_axis():
    lat = Lattice([[1.0]], [[0.0]], periodic_dirs=[0])
    mesh = Mesh(["k", "l"])
    mesh.build_grid(shape=(8, 1), lambda_start=[0.0], lambda_stop=[0.0])
    WFArray(lat, mesh, spinful=False)  # should not raise


def test_wfarray_rejects_loop_axis_with_single_point():
    lat = Lattice([[1.0]], [[0.0]], periodic_dirs=[0])
    mesh = Mesh(["k"])
    mesh.build_grid(shape=(1,), k_endpoints=False)
    with pytest.raises(
        ValueError, match="Looping mesh axes must have at least two samples"
    ):
        WFArray(lat, mesh)
