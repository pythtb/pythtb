"""Tests for reading the off-diagonal Wannier position matrix (write_tb).

These exercise:
  * ``pythtb.io.w90.read_tb`` / ``read_r`` parsing of ``prefix_tb.dat`` and
    ``prefix_r.dat``,
  * the ``W90`` interface (``pos_r``, ``has_position_matrix``,
    ``wannier_centers``, ``berry_connection_wann``),
  * propagation of the position matrix onto ``TBModel`` via ``W90.model``.

The fixtures are synthetic but internally consistent: the R=0 diagonal of the
position matrix equals the centers listed in ``prefix_centres.xyz``, and the
position blocks satisfy ``r(-R) = r(R)^dagger`` so that the Wannier-gauge Berry
connection is Hermitian.
"""

import numpy as np
import pytest

from pythtb import W90
from pythtb.io.w90 import read_tb, read_r


LAT = np.diag([5.0, 5.0, 5.0])
CENTERS = np.array([[0.10, 0.20, 0.30], [0.60, 0.70, 0.80]])  # (num_wan, 3)
NUM_WAN = 2
# Wigner-Seitz vectors and degeneracies (each non-zero R has its -R partner).
RVECS = [(0, 0, 0), (1, 0, 0), (-1, 0, 0)]
DEG = [1, 1, 1]


def _build_blocks():
    """Return Hermitian-consistent H(R) and r(R) dictionaries."""
    rng = np.random.default_rng(1234)

    # Hamiltonian: H(0) Hermitian with real diagonal; H(-R) = H(R)^dagger.
    H0 = np.array([[1.0 + 0j, 0.3 + 0.1j], [0.3 - 0.1j, 2.0 + 0j]])
    H1 = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    ham = {
        (0, 0, 0): H0,
        (1, 0, 0): H1,
        (-1, 0, 0): H1.conj().T,
    }

    # Position matrix: r(0)[a] Hermitian with diagonal = CENTERS[:, a];
    # r(-R)[a] = r(R)[a]^dagger.
    X0 = np.zeros((3, 2, 2), complex)
    X1 = np.zeros((3, 2, 2), complex)
    for a in range(3):
        offdiag = (0.05 + 0.02j) * (a + 1)
        X0[a] = np.array([[CENTERS[0, a], offdiag], [np.conj(offdiag), CENTERS[1, a]]])
        X1[a] = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    pos = {
        (0, 0, 0): X0,
        (1, 0, 0): X1,
        (-1, 0, 0): X1.conj().transpose(0, 2, 1),
    }
    return ham, pos


def _write_win(folder, prefix):
    (folder / f"{prefix}.win").write_text(
        "begin unit_cell_cart\n"
        "Ang\n"
        f"{LAT[0, 0]} {LAT[0, 1]} {LAT[0, 2]}\n"
        f"{LAT[1, 0]} {LAT[1, 1]} {LAT[1, 2]}\n"
        f"{LAT[2, 0]} {LAT[2, 1]} {LAT[2, 2]}\n"
        "end unit_cell_cart\n"
    )


def _write_centres(folder, prefix):
    lines = [f"{NUM_WAN}", "comment line"]
    for n in range(NUM_WAN):
        x, y, z = CENTERS[n]
        lines.append(f"X {x} {y} {z}")
    (folder / f"{prefix}_centres.xyz").write_text("\n".join(lines) + "\n")


def _write_tb(folder, prefix, ham, pos):
    out = ["written on test"]
    for i in range(3):
        out.append(f"{LAT[i, 0]:.8f} {LAT[i, 1]:.8f} {LAT[i, 2]:.8f}")
    out.append(f"{NUM_WAN}")
    out.append(f"{len(RVECS)}")
    out.append(" ".join(str(d) for d in DEG))
    # Hamiltonian blocks
    for R in RVECS:
        out.append("")
        out.append(f"{R[0]} {R[1]} {R[2]}")
        H = ham[R]
        for j in range(NUM_WAN):
            for i in range(NUM_WAN):
                out.append(f"{i + 1} {j + 1} {H[i, j].real:.8f} {H[i, j].imag:.8f}")
    # Position blocks
    for R in RVECS:
        out.append("")
        out.append(f"{R[0]} {R[1]} {R[2]}")
        X = pos[R]
        for j in range(NUM_WAN):
            for i in range(NUM_WAN):
                vals = []
                for a in range(3):
                    vals.append(f"{X[a, i, j].real:.8f}")
                    vals.append(f"{X[a, i, j].imag:.8f}")
                out.append(f"{i + 1} {j + 1} " + " ".join(vals))
    (folder / f"{prefix}_tb.dat").write_text("\n".join(out) + "\n")


def _write_hr(folder, prefix, ham):
    out = ["written on test", f"{NUM_WAN}", f"{len(RVECS)}"]
    out.append(" ".join(str(d) for d in DEG))
    for R in RVECS:
        H = ham[R]
        for j in range(NUM_WAN):
            for i in range(NUM_WAN):
                out.append(
                    f"{R[0]} {R[1]} {R[2]} {i + 1} {j + 1} "
                    f"{H[i, j].real:.8f} {H[i, j].imag:.8f}"
                )
    (folder / f"{prefix}_hr.dat").write_text("\n".join(out) + "\n")


def _write_r(folder, prefix, pos):
    out = ["written on test", f"{NUM_WAN}", f"{len(RVECS)}"]
    for R in RVECS:
        X = pos[R]
        for j in range(NUM_WAN):
            for i in range(NUM_WAN):
                vals = []
                for a in range(3):
                    vals.append(f"{X[a, i, j].real:.8f}")
                    vals.append(f"{X[a, i, j].imag:.8f}")
                out.append(f"{R[0]} {R[1]} {R[2]} {i + 1} {j + 1} " + " ".join(vals))
    (folder / f"{prefix}_r.dat").write_text("\n".join(out) + "\n")


@pytest.fixture
def tb_folder(tmp_path):
    prefix = "test"
    ham, pos = _build_blocks()
    _write_win(tmp_path, prefix)
    _write_centres(tmp_path, prefix)
    _write_tb(tmp_path, prefix, ham, pos)
    return tmp_path, prefix, ham, pos


# --------------------------------------------------------------------------- #
# Low-level reader
# --------------------------------------------------------------------------- #


def test_read_tb_roundtrip(tb_folder):
    folder, prefix, ham, pos = tb_folder
    num_wan, ham_r, pos_r, lat = read_tb(folder, prefix)

    assert num_wan == NUM_WAN
    assert np.allclose(lat, LAT)
    assert set(pos_r.keys()) == set(RVECS)
    for R in RVECS:
        assert pos_r[R].shape == (3, NUM_WAN, NUM_WAN)
        assert np.allclose(ham_r[R].h, ham[R])
        assert np.allclose(pos_r[R], pos[R])


def test_read_tb_missing(tmp_path):
    with pytest.raises(Exception):
        read_tb(tmp_path, "does_not_exist")


def test_read_r_matches_tb(tb_folder):
    folder, prefix, ham, pos = tb_folder
    _write_r(folder, prefix, pos)
    pos_r = read_r(folder, prefix, NUM_WAN)
    for R in RVECS:
        assert np.allclose(pos_r[R], pos[R])


# --------------------------------------------------------------------------- #
# W90 interface
# --------------------------------------------------------------------------- #


def test_w90_has_position_matrix(tb_folder):
    folder, prefix, _ham, _pos = tb_folder
    w90 = W90(str(folder), prefix)
    assert w90.has_position_matrix is True
    assert set(w90.pos_r.keys()) == set(RVECS)


def test_w90_falls_back_without_tb(tmp_path):
    """With only _hr.dat (no _tb.dat / _r.dat) there is no position matrix."""
    prefix = "test"
    ham, _pos = _build_blocks()
    _write_win(tmp_path, prefix)
    _write_centres(tmp_path, prefix)
    _write_hr(tmp_path, prefix, ham)
    w90 = W90(str(tmp_path), prefix)
    assert w90.has_position_matrix is False
    assert w90.pos_r is None


def test_position_matrix_accessor(tb_folder):
    folder, prefix, _ham, pos = tb_folder
    w90 = W90(str(folder), prefix)

    pm = w90.position_matrix()  # Cartesian <0n|r|Rm>
    assert set(pm.keys()) == set(RVECS)
    for R in RVECS:
        assert pm[R].shape == (3, NUM_WAN, NUM_WAN)
        assert np.allclose(pm[R], pos[R])

    # returns copies -- mutating the result must not corrupt the W90 object
    pm[(0, 0, 0)][0, 0, 0] = 999.0
    assert not np.isclose(w90.position_matrix()[(0, 0, 0)][0, 0, 0], 999.0)

    # reduced components: r_red_i = sum_a r_cart_a inv(lat)[a, i]
    pm_red = w90.position_matrix(cartesian=False)
    inv_lat = np.linalg.inv(LAT)
    for R in RVECS:
        assert np.allclose(pm_red[R], np.einsum("anm, ai -> inm", pos[R], inv_lat))


def test_position_matrix_requires_data(tmp_path):
    prefix = "test"
    ham, _pos = _build_blocks()
    _write_win(tmp_path, prefix)
    _write_centres(tmp_path, prefix)
    _write_hr(tmp_path, prefix, ham)
    w90 = W90(str(tmp_path), prefix)
    with pytest.raises(ValueError, match="No position matrix"):
        w90.position_matrix()


def test_wannier_centers_match_centres_file(tb_folder):
    folder, prefix, _ham, _pos = tb_folder
    w90 = W90(str(folder), prefix)
    # R=0 diagonal of the position matrix == centers from prefix_centres.xyz
    assert np.allclose(w90.wannier_centers(), CENTERS)
    assert np.allclose(w90.wannier_centers(), w90.xyz_cen)


def test_berry_connection_hermitian(tb_folder):
    folder, prefix, _ham, _pos = tb_folder
    w90 = W90(str(folder), prefix)
    kpts = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.37, -0.11, 0.5]])
    A = w90.berry_connection_wann(kpts)
    assert A.shape == (3, 3, NUM_WAN, NUM_WAN)
    for ik in range(kpts.shape[0]):
        for a in range(3):
            assert np.allclose(A[ik, a], A[ik, a].conj().T)


def test_berry_connection_matches_manual_sum(tb_folder):
    folder, prefix, _ham, pos = tb_folder
    w90 = W90(str(folder), prefix)
    k = np.array([0.13, -0.27, 0.4])
    A = w90.berry_connection_wann(k)[0]  # (3, nw, nw)
    manual = np.zeros((3, NUM_WAN, NUM_WAN), complex)
    for R in RVECS:
        phase = np.exp(2j * np.pi * np.dot(k, R))
        manual += phase * pos[R]  # deg == 1 here
    assert np.allclose(A, manual)


def test_berry_connection_requires_position(tmp_path):
    prefix = "test"
    ham, _pos = _build_blocks()
    _write_win(tmp_path, prefix)
    _write_centres(tmp_path, prefix)
    _write_hr(tmp_path, prefix, ham)
    w90 = W90(str(tmp_path), prefix)
    with pytest.raises(ValueError, match="No position matrix"):
        w90.berry_connection_wann(np.zeros(3))


# --------------------------------------------------------------------------- #
# Propagation to TBModel
# --------------------------------------------------------------------------- #


def test_model_carries_position_matrix(tb_folder):
    folder, prefix, _ham, _pos = tb_folder
    w90 = W90(str(folder), prefix)
    tb = w90.model()
    assert tb.has_wannier_position is True
    assert not tb.assume_position_operator_diagonal

    k = np.array([[0.13, -0.27, 0.4]])
    A_model = tb.wannier_berry_connection(k)
    A_w90 = w90.berry_connection_wann(k)
    assert np.allclose(A_model, A_w90)


def test_model_without_position_has_no_connection(tmp_path):
    prefix = "test"
    ham, _pos = _build_blocks()
    _write_win(tmp_path, prefix)
    _write_centres(tmp_path, prefix)
    _write_hr(tmp_path, prefix, ham)
    tb = W90(str(tmp_path), prefix).model()
    assert tb.has_wannier_position is False
    with pytest.raises(ValueError, match="no Wannier position matrix"):
        tb.wannier_berry_connection(np.zeros(3))


# --------------------------------------------------------------------------- #
# Berry curvature machinery (shape + complete-manifold property)
# --------------------------------------------------------------------------- #


def test_berry_curvature_correction_auto_and_complete_manifold(tb_folder):
    folder, prefix, _ham, _pos = tb_folder
    tb = W90(str(folder), prefix).model()
    assert tb.has_wannier_position
    k = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
    occ = [0, 1]  # both bands -> complete manifold

    # Correction is auto-applied for write_tb models; returns Omega_z (plane 0,1).
    om = tb.berry_curvature(k, occ_idxs=occ, plane=(0, 1), cartesian=True)
    assert om.shape == (2,)
    assert np.all(np.isreal(om))

    # Complete (all-occupied) manifold: the diagonal approximation must vanish.
    om0 = tb.berry_curvature(
        k, occ_idxs=occ, plane=(0, 1), cartesian=True, include_external=False
    )
    assert np.max(np.abs(om0)) < 1e-9

    # With the position correction the curl term is generally non-zero.
    assert np.max(np.abs(om)) > 0.0


def test_berry_curvature_full_tensor_antisymmetric(tb_folder):
    folder, prefix, _ham, _pos = tb_folder
    tb = W90(str(folder), prefix).model()
    k = np.array([[0.13, -0.2, 0.4]])
    om = tb.berry_curvature(k, occ_idxs=[0], cartesian=True)  # one band occupied
    assert om.shape == (3, 3, 1)
    assert np.allclose(om, -np.swapaxes(om, 0, 1))  # antisymmetric tensor
    assert np.all(np.isfinite(om))


def test_berry_curvature_non_abelian(tb_folder):
    folder, prefix, _ham, _pos = tb_folder
    tb = W90(str(folder), prefix).model()
    k = np.array([[0.1, 0.2, 0.3], [0.37, -0.11, 0.05]])
    occ = [0, 1]  # complete 2-band manifold

    # plane=(0,1): non-Abelian Omega_z matrix, shape (Nk, n_occ, n_occ)
    Bna = tb.berry_curvature(
        k, occ_idxs=occ, plane=(0, 1), non_abelian=True, cartesian=True
    )
    assert Bna.shape == (2, 2, 2)
    # Hermitian in the band indices
    assert np.allclose(Bna, np.conj(np.transpose(Bna, (0, 2, 1))))
    # its band trace equals the band-summed (abelian) curvature
    Bbs = tb.berry_curvature(k, occ_idxs=occ, plane=(0, 1), cartesian=True)
    assert np.allclose(np.einsum("kmm -> k", Bna).real, Bbs)

    # full tensor (plane=None) is antisymmetric in the Cartesian axes
    Bfull = tb.berry_curvature(k, occ_idxs=occ, non_abelian=True, cartesian=True)
    assert Bfull.shape == (3, 3, 2, 2, 2)
    assert np.allclose(Bfull, -np.swapaxes(Bfull, 0, 1))


def test_berry_curvature_fermi_occupation(tb_folder):
    folder, prefix, _ham, _pos = tb_folder
    tb = W90(str(folder), prefix).model()
    k = np.array([[0.1, 0.2, 0.3], [0.37, -0.11, 0.05]])

    # A Fermi level above both bands == complete manifold == occ_idxs=[0, 1].
    a = tb.berry_curvature(k, occ_idxs=[0, 1], plane=(0, 1), cartesian=True)
    b = tb.berry_curvature(k, fermi=100.0, plane=(0, 1), cartesian=True)
    assert np.allclose(a, b)

    # A per-k (metal-like) Fermi level runs and gives a real, finite tensor.
    om = tb.berry_curvature(k, fermi=1.5, cartesian=True)
    assert om.shape == (3, 3, 2)
    assert np.all(np.isreal(om)) and np.all(np.isfinite(om))


def test_berry_curvature_non_abelian_rejects_fermi(tb_folder):
    folder, prefix, _ham, _pos = tb_folder
    tb = W90(str(folder), prefix).model()
    # The non-Abelian matrix needs a fixed band group; a Fermi level gives a
    # k-dependent occupied dimension.
    with pytest.raises(ValueError, match="fixed band set"):
        tb.berry_curvature(np.zeros((1, 3)), fermi=1.5, non_abelian=True)


def test_berry_curvature_correction_requires_position(tmp_path):
    prefix = "test"
    ham, _pos = _build_blocks()
    _write_win(tmp_path, prefix)
    _write_centres(tmp_path, prefix)
    _write_hr(tmp_path, prefix, ham)
    tb = W90(str(tmp_path), prefix).model()
    assert not tb.has_wannier_position
    with pytest.raises(ValueError, match="requires a Wannier position matrix"):
        tb.berry_curvature(np.zeros((1, 3)), occ_idxs=[0], include_external=True)
