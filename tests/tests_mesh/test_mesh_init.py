from pythtb import Mesh
import numpy as np


def test_mesh_init():
    # Create a 2D mesh with 10 points along each axis
    mesh = Mesh(dim_k=2, axis_types=["k", "k"])
    mesh.build_grid((10, 10))

    # Check the shape of the mesh
    assert mesh.shape_axes == (10, 10)

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


def test_mesh_axis_type_ordering():
    mesh = Mesh(dim_k=2, axis_types=["l", "k", "k"])
    mesh.build_grid(
        shape=(3, 4, 5),
        gamma_centered=[True, False],
        k_endpoints=[False, True],
        lambda_start=[-np.pi],
        lambda_stop=[np.pi],
        lambda_endpoints=[True],
    )
    assert mesh.axis_names == ["l_0", "k_0", "k_1"]
    k_pts = mesh.get_k_points()
    assert np.isclose(k_pts[0, 0, 0], -0.5)
    assert np.isclose(k_pts[-1, 0, 0], 0.25)
    assert np.isclose(k_pts[0, 0, 1], 0.0)
    assert np.isclose(k_pts[0, -1, 1], 1.0)
    lam_points = mesh.get_param_points().ravel()
    assert lam_points[0] == -np.pi and lam_points[-1] == np.pi


def test_mesh_k_and_param_points_match_axis_order():
    mesh = Mesh(dim_k=1, axis_types=["l", "k"])
    mesh.build_grid(
        shape=(2, 3),
        gamma_centered=[False],
        k_endpoints=[True],
        lambda_start=[-0.2],
        lambda_stop=[0.4],
        lambda_endpoints=[True],
    )

    k_pts = mesh.get_k_points()
    param_pts = mesh.get_param_points()

    assert k_pts.shape == (3, 1)
    assert param_pts.shape == (2, 1)

    np.testing.assert_allclose(k_pts[:, 0], np.linspace(0, 1, 3))
    np.testing.assert_allclose(param_pts[:, 0], np.linspace(-0.2, 0.4, 2))
