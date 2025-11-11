import sys
import os
import pytest
import numpy as np

# Ensure local pythtb.py is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
import pythtb


def test_version_exists_and_format():
    # __version__ should be defined and be a non-empty string
    assert hasattr(pythtb, '__version__'), "pythtb must define a __version__"
    version = pythtb.__version__
    assert isinstance(version, str) and version, "__version__ should be a non-empty string"


def test_single_orbital_onsite():
    # 0D model with one orbital: eigenvalue equals onsite energy
    # args: dim, n_orb, lattice, orbitals
    lat = [[1.0]]
    orb = [[0.0]]
    m = pythtb.tb_model(0, 1, lat, orb)
    m.set_onsite([2.5])
    evals = m.solve_all()
    assert evals.shape == (1,)
    assert np.allclose(evals, [2.5])


def test_two_orbital_hopping():
    # 0D model with two orbitals connected by hopping
    lat = [[1.0]]
    orb = [[0.0], [0.5]]
    m = pythtb.tb_model(0, 1, lat, orb)
    # zero onsite
    m.set_onsite([0.0, 0.0])
    # hopping t=3 between orbitals
    m.set_hop(3.0, 0, 1)
    evals = np.sort(m.solve_all())
    # eigenvalues of [[0,3],[3,0]] are [-3,3]
    assert evals.shape == (2,)
    assert np.allclose(evals, [-3.0, 3.0])


def test_k_path_shapes():
    # 1D periodic model (one orbital), trivial onsite
    lat = [[1.0]]
    orb = [[0.0]]
    m = pythtb.tb_model(1, 1, lat, orb)
    m.set_onsite([0.0])
    # path from k=0 to k=0.5 with 5 points
    path = [[0.0], [0.5]]
    k_vec, k_dist, k_node = m.k_path(path, 5)
    # k_vec should be 5x1, k_dist length 5, k_node length 2
    assert k_vec.shape == (5, 1)
    assert k_dist.shape == (5,)
    assert len(k_node) == 2
    # Check k_dist starts at 0 and ends at >0
    assert pytest.approx(k_dist[0]) == 0.0
    assert k_dist[-1] > 0.0


def test_solve_all_consistency_across_calls():
    # Calling solve_all twice yields same result
    lat = [[1.0]]
    orb = [[0.0]]
    m = pythtb.tb_model(0, 1, lat, orb)
    m.set_onsite([-1.0])
    evals1 = m.solve_all()
    evals2 = m.solve_all()
    assert np.array_equal(evals1, evals2)
