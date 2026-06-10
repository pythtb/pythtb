"""Regression test: PythTB Berry curvature vs a committed postw90 reference.

Uses the small monolayer-hBN fixtures under ``tests/w90_berry/hBN`` (the
position-matrix file ``hBN_tb.dat`` and the postw90 curvature ``hBN-curv.dat``).
The data were produced by the pipeline in ``tests/w90_berry/hBN/run.sh``; see
that directory's README for how to regenerate them (and for the Fe benchmark).

For hBN's complete 4-valence Wannier manifold the diagonal-position
approximation is identically zero, so the whole Berry curvature is the
off-diagonal position-matrix correction -- which must reproduce postw90.
"""

from pathlib import Path

import numpy as np
import pytest

from pythtb import W90

DATA = Path(__file__).resolve().parents[1] / "w90_berry" / "hBN"
NODES = [[0, 0, 0], [0.5, 0, 0], [1 / 3, 1 / 3, 0], [0, 0, 0]]  # G M K G
FERMI = -0.035

pytestmark = pytest.mark.skipif(
    not (DATA / "hBN_tb.dat").exists() or not (DATA / "hBN-curv.dat").exists(),
    reason="hBN reference data not present (run tests/w90_berry/hBN/run.sh)",
)


def _kpath_from_postw90(s, nodes_red, lat):
    nodes_red = np.asarray(nodes_red, float)
    B = 2 * np.pi * np.linalg.inv(np.asarray(lat, float)).T
    nodes_cart = nodes_red @ B
    seg = np.linalg.norm(np.diff(nodes_cart, axis=0), axis=1)
    s_nodes = np.concatenate([[0.0], np.cumsum(seg)])
    s_geo = s * (s_nodes[-1] / s[-1])
    k = np.zeros((s.size, 3))
    for i, sv in enumerate(s_geo):
        j = min(max(np.searchsorted(s_nodes, sv, "right") - 1, 0), len(seg) - 1)
        t = (sv - s_nodes[j]) / seg[j] if seg[j] > 0 else 0.0
        k[i] = nodes_red[j] + t * (nodes_red[j + 1] - nodes_red[j])
    return k


def test_hBN_berry_curvature_matches_postw90():
    w90 = W90(str(DATA), "hBN")
    assert w90.has_position_matrix
    tb = w90.model()  # correction lives on TBModel.berry_curvature

    ref = np.loadtxt(DATA / "hBN-curv.dat")
    k = _kpath_from_postw90(ref[:, 0], NODES, w90.lat)
    occ = [0, 1, 2, 3]  # complete valence manifold

    # plane=(0,1) -> Omega_z (Cartesian, Ang^2). Correction auto-applied.
    omega = tb.berry_curvature(k, occ_idxs=occ, plane=(0, 1), cartesian=True)
    omega0 = tb.berry_curvature(
        k, occ_idxs=occ, plane=(0, 1), cartesian=True, include_external=False
    )
    ref_z = -ref[:, 3]  # postw90 prints -Omega

    # diagonal approximation is exactly zero for this complete manifold
    assert np.max(np.abs(omega0)) < 1e-9

    # correction reproduces postw90
    corr = np.corrcoef(omega, ref_z)[0, 1]
    fit = np.sum(omega * ref_z) / np.sum(omega * omega)
    assert corr > 0.999
    assert abs(fit - 1.0) < 0.01


def test_hBN_wannier_centers_match_centres_file():
    w90 = W90(str(DATA), "hBN")
    # diagonal of the R=0 position block == prefix_centres.xyz
    assert np.allclose(w90.wannier_centers(), w90.xyz_cen, atol=1e-6)


def test_hBN_nonabelian_matches_wilson_loop():
    """Full non-Abelian Berry curvature matrix vs a finite-difference Wilson loop
    with position-corrected overlaps (the gauge-invariant check of the off-trace
    structure that the band trace cannot see)."""
    from pythtb.io.w90 import wannier_connection_ft

    w90 = W90(str(DATA), "hBN")
    tb = w90.model()
    occ = [0, 1, 2, 3]
    lat = w90.lat
    Brec = 2 * np.pi * np.linalg.inv(lat).T

    # Hamiltonian-gauge eigenvectors and the orbital-basis external connection.
    ham_h = {R: blk["h"] / blk["deg"] for R, blk in w90.ham_r.items()}

    def U_of(kk):
        H = sum(np.exp(2j * np.pi * np.dot(kk, R)) * H_R for R, H_R in ham_h.items())
        return np.linalg.eigh(H)[1]

    pos_eff = {R: w90.pos_r[R] / w90.ham_r[R]["deg"] for R in w90.pos_r}

    def overlap(k1, k2):
        U1, U2 = U_of(k1), U_of(k2)
        A_raw = wannier_connection_ft(pos_eff, (k1 + k2) / 2)[
            0
        ]  # (3, N, N) at midpoint
        # Hermitianize the connection, matching berry_curvature_wannier_matrix.
        A = 0.5 * (A_raw + np.conj(np.transpose(A_raw, (0, 2, 1))))
        dk = (k2 - k1) @ Brec  # Cartesian
        S = np.eye(U1.shape[0]) - 1j * np.einsum("a, aij -> ij", dk, A)
        return (U1.conj().T @ S @ U2)[np.ix_(occ, occ)]

    def logm(W):
        w, V = np.linalg.eig(W)
        return (V * np.log(w)) @ np.linalg.inv(V)

    k0 = np.array([1 / 3 + 0.02, 1 / 3 - 0.01, 0.0])  # near K, sizeable curvature
    B_an = tb.berry_curvature(
        k0[None], occ_idxs=occ, plane=(0, 1), non_abelian=True, cartesian=True
    )[0]

    prev = None
    for delta in (2e-3, 1e-3, 5e-4):
        dx, dy = np.array([delta, 0, 0]), np.array([0, delta, 0])
        W = (
            overlap(k0, k0 + dx)
            @ overlap(k0 + dx, k0 + dx + dy)
            @ overlap(k0 + dx + dy, k0 + dy)
            @ overlap(k0 + dy, k0)
        )
        dxc, dyc = dx @ Brec, dy @ Brec
        area = dxc[0] * dyc[1] - dxc[1] * dyc[0]
        L = logm(W)
        F = 1j * 0.5 * (L - L.conj().T) / area  # anti-Herm part = curvature
        err = np.linalg.norm(F - B_an)
        if prev is not None:
            assert err < prev  # converging to the analytic matrix
        prev = err
    assert prev < 5e-3  # agrees at the finest spacing
