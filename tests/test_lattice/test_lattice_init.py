import numpy as np
import pytest

from pythtb import Lattice


def test_default_is_finite():
    lat = Lattice(lat_vecs=[[1.0]], orb_vecs=[[0.0]])
    assert lat.periodic_dirs == []
    assert lat.dim_r == 1
    assert lat.dim_k == 0


@pytest.mark.parametrize("shortcut", (..., "all"))
def test_all_shortcuts(shortcut):
    lat = Lattice(
        lat_vecs=[[1.0, 0.0], [0.0, 1.0]],
        orb_vecs=[[0.0, 0.0]],
        periodic_dirs=shortcut,
    )
    assert lat.periodic_dirs == [0, 1]
    assert lat.dim_k == 2
    np.testing.assert_array_equal(lat.recip_lat_vecs, 2 * np.pi * np.eye(2))


def test_explicit_empty_list_is_allowed():
    lat = Lattice(
        lat_vecs=[[1.0, 0.0], [0.0, 1.0]],
        orb_vecs=[[0.0, 0.0]],
        periodic_dirs=[],
    )
    assert lat.periodic_dirs == []
    assert lat.dim_k == 0
    with pytest.raises(ValueError):
        _ = lat.recip_lat_vecs  # no reciprocal lattice when dim_k == 0


def test_invalid_periodic_index_raises():
    with pytest.raises(ValueError, match="out of bounds"):
        Lattice(
            lat_vecs=[[1.0, 0.0], [0.0, 1.0]],
            orb_vecs=[[0.0, 0.0]],
            periodic_dirs=[2],
        )


def test_periodic_dirs_setter_recomputes_reciprocal():
    lat = Lattice(
        lat_vecs=[[2.0, 0.0], [0.0, 1.0]],
        orb_vecs=[[0.0, 0.0]],
        periodic_dirs=[0],
    )
    np.testing.assert_allclose(lat.recip_lat_vecs, 2 * np.pi * np.array([[0.5, 0.0]]))

    lat.periodic_dirs = [0, 1]
    np.testing.assert_allclose(
        lat.recip_lat_vecs, 2 * np.pi * np.array([[0.5, 0.0], [0.0, 1.0]])
    )
    assert lat.dim_k == 2
