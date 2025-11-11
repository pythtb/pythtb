from pythtb import Mesh
import numpy as np


def test_mesh_k_and_param_points_match_axis_order():
    mesh = Mesh(["l", "k"])
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


def test_mesh_axis_type_ordering():
    mesh = Mesh(["l", "k", "k"])
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
    # first k-axis gamma-centered -> starts at -0.5
    assert np.isclose(k_pts[0, 0, 0], -0.5)
    # first k-axis does not include endpoint -> last entry less than 0.5
    assert np.isclose(k_pts[-1, 0, 0], 0.25)
    # second k-axis not gamma-centered -> starts at 0.0
    assert np.isclose(k_pts[0, 0, 1], 0.0)
    # second k-axis includes endpoint -> last entry equals 1.0
    assert np.isclose(k_pts[0, -1, 1], 1.0)
    # lambda axis got the requested endpoints
    lam_points = mesh.get_param_points().ravel()
    assert lam_points[0] == -np.pi and lam_points[-1] == np.pi
