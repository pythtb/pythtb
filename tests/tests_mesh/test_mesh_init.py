from pythtb import Mesh


def test_mesh_init():
    # Create a 2D mesh with 10 points along each axis
    mesh = Mesh(dim_k=2, dim_lambda=0, axis_types=["k", "k"])
    mesh.build_grid((10, 10))

    # Check the shape of the mesh
    assert mesh.shape_mesh == (10, 10)

    # Check the number of dimensions
    assert mesh.naxes == 2
    assert mesh.nk_axes == 2

    # Check that k-axes are looped by default in build_grid
    for axis in range(mesh.naxes):
        assert mesh.is_axis_looped(axis)
        assert not mesh.is_axis_closed(axis)
        assert mesh.is_grid
        assert mesh.is_k_torus

    # Loop the first axis around the first k-space dimension
    mesh.loop(axis_idx=0, component_idx=0)

    # Check that the first axis is now looped
    assert mesh.is_axis_looped(0)

    # Loop the second axis around the second k-space dimension
    mesh.loop(axis_idx=1, component_idx=1)

    # Check that the second axis is now looped
    assert mesh.is_axis_looped(1)
