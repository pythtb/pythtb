import numpy as np
from pythtb import TBModel, Lattice
import pytest


@pytest.mark.parametrize(
    "periodic_dirs, lat_vecs, orbital_pos, spinful",
    [
        ([0], [[1]], [[0]], False),  # 1D
        ([0], [[1]], [[0], [0.5]], False),  # 1D with two orbitals
        ([0, 1], [[1, 0], [0, 1]], [[0, 0], [0.5, 0.5]], False),  # 2D
        (
            [0],
            [[1, 0], [0, 1]],
            [[0, 0], [0.5, 0.5]],
            False,
        ),  # 1D k-space, 2D real space
        (
            [0, 1, 2],
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            [[0, 0, 0], [0.25, 0.25, 0.25]],
            False,
        ),  # 3D
        (
            [0, 1],
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            [[0, 0, 0], [0.25, 0.25, 0.25]],
            False,
        ),  # 2D k-space, 3D real space
        (
            [0],
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            [[0, 0, 0], [0.25, 0.25, 0.25]],
            False,
        ),  # 1D k-space, 3D real space
        ([0], [[1]], [[0]], True),  # 1D with nspin=2
        ([0], [[1]], [[0], [0.5]], True),  # 1D with two orbitals and nspin=2
        ([0, 1], [[1, 0], [0, 1]], [[0, 0], [0.5, 0.5]], True),  # 2D with nspin=2
        (
            [0],
            [[1, 0], [0, 1]],
            [[0, 0], [0.5, 0.5]],
            True,
        ),  # 1D k-space, 2D real space with nspin=2
        (
            [0, 1, 2],
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            [[0, 0, 0], [0.25, 0.25, 0.25]],
            True,
        ),  # 3D with nspin=2
        (
            [0, 1],
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            [[0, 0, 0], [0.25, 0.25, 0.25]],
            True,
        ),  # 2D k-space, 3D real space with nspin=2
        (
            [0],
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            [[0, 0, 0], [0.25, 0.25, 0.25]],
            True,
        ),  # 1D k-space, 3D real space with nspin=2
    ],
)
def test_tbmodel_initialization(periodic_dirs, lat_vecs, orbital_pos, spinful):
    """
    Test the TBModel initialization with various dimensions and lattice vectors.
    """
    # Create a TBModel instance
    lattice = Lattice(
        lat_vecs=lat_vecs, orb_vecs=orbital_pos, periodic_dirs=periodic_dirs
    )
    test_model = TBModel(lattice, spinful=spinful)

    assert test_model.lattice == lattice, "Lattice should be set correctly in TBModel"
    # Check lattice properties match
    assert test_model.periodic_dirs == periodic_dirs, "Periodic directions should match"
    assert test_model.lat_vecs.shape == (len(lat_vecs), len(lat_vecs[0])), (
        "Lattice vectors shape should match"
    )
    assert test_model.orb_vecs.shape == (len(orbital_pos), len(orbital_pos[0])), (
        "Orbital positions shape should match"
    )

    # Check if the dimensions are set correctly
    assert test_model.dim_k == len(periodic_dirs), (
        f"dim_k should be {len(periodic_dirs)}"
    )
    assert test_model.dim_r == len(lat_vecs[0]), f"dim_r should be {len(lat_vecs[0])}"
    assert test_model.nspin == 2 if spinful else 1, (
        f"nspin should be {2 if spinful else 1}"
    )

    # Check if the lattice vectors and orbital positions are set correctly
    np.testing.assert_array_equal(
        test_model.lat_vecs, lat_vecs, "lat_vecs should match"
    )
    np.testing.assert_array_equal(
        test_model.orb_vecs, orbital_pos, "orbital positions should match"
    )

    # Check if the number of orbitals is correct
    assert test_model.norb == len(orbital_pos), (
        "norb should match the number of orbital positions"
    )

    assert len(test_model.hoppings) == 0, "Hoppings should be empty upon initialization"
    if not spinful:
        np.testing.assert_array_equal(
            test_model.onsite,
            np.zeros(test_model.norb),
            "Onsite energies should be zero",
        )
    else:
        np.testing.assert_array_equal(
            test_model.onsite,
            np.zeros((test_model.norb, 2, 2), dtype=complex),
            "Site energies should be zero matrices for spinful model",
        )

    assert test_model.parameters == [], (
        "Parameters dictionary should be empty upon initialization"
    )


@pytest.mark.parametrize(
    "periodic_dirs, lat_vecs, orbital_pos, spinful",
    [
        ([0, 1, 2], [[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0, False),  # 3D
        (
            [0, 1],
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            5,
            False,
        ),  # 2D k-space, 3D real space
        ([0], [[0, 1, 1], [1, 0, 1], [1, 1, 0]], 4, False),  # 1D k-space, 3D real space
        ([0, 1, 2], [[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0, True),  # 3D with nspin=2
        (
            [0, 1],
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            10,
            True,
        ),  # 2D k-space, 3D real space with nspin=2
        (
            [0],
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
            1,
            True,
        ),  # 1D k-space, 3D real space with nspin=2
        ([0, 1], [[1, 0], [0, 1]], 1, 1),  # 2D with nspin=1
        ([0], [[1, 0], [0, 1]], 2, 1),  # 1D k-space, 2D real space with nspin=1
        (
            [0, 1, 2],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            4,
            False,
        ),  # Simple cubic lattice
        (
            [0, 1],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            2,
            False,
        ),  # Simple cubic lattice in k-space
    ],
)
def test_origin_orbs(periodic_dirs, lat_vecs, orbital_pos, spinful):
    """
    Test the TBModel's origin_orbs method.
    """
    lattice = Lattice(
        lat_vecs=lat_vecs, orb_vecs=orbital_pos, periodic_dirs=periodic_dirs
    )
    model = TBModel(lattice, spinful=spinful)

    assert model.norb == orbital_pos, (
        "norb should match the number of orbital positions"
    )
    # assert that lat_vecs are all at origin
    np.testing.assert_array_equal(
        model.orb_vecs,
        np.zeros_like(model.orb_vecs),
        "lattice vectors should be at the origin",
    )
    assert model.orb_vecs.shape[0] == orbital_pos, (
        "lat_vecs should have the same number of vectors as orbitals"
    )
