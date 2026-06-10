"""Reference Wannier-interpolated Berry curvature (Wang-Yates-Souza-Vanderbilt).

This mirrors, in NumPy, the exact formula implemented in Wannier90's
``postw90`` (``berry.F90::berry_get_imfgh_klist``), so that PythTB's
position-matrix correction can be validated against ``postw90`` output.

The band-summed Berry curvature (the quantity ``postw90`` prints as
``-Omega``) is the sum of three terms:

    -Omega_gamma = J0 + J1 + J2

    J0 = Re Tr[ f . Omega_bar_gamma ]                         (position-matrix curl)
    J1 = -2 Im( Tr[A_alpha . JJp_beta] + Tr[JJm_alpha . A_beta] )
    J2 = -2 Im Tr[ JJm_alpha . JJp_beta ]                     (naive Kubo term)

with (alpha, beta) the axial partners of gamma. ``J2`` alone is the
"diagonal position operator" result PythTB computes today; ``J0`` and ``J1``
are the corrections supplied by the off-diagonal position matrix
``<0n|r|Rm>`` (``pos_r``).

All matrices are in the Wannier gauge; ``f`` is the occupied projector,
``A`` the Wannier-gauge Berry connection, ``JJp/JJm`` the energy-denominator
("D") matrices restricted to the empty-occupied / occupied-empty blocks.
"""

from __future__ import annotations

import numpy as np

# axial map gamma -> (alpha, beta):  Omega_x=(y,z), Omega_y=(z,x), Omega_z=(x,y)
ALPHA = (1, 2, 0)
BETA = (2, 0, 1)


def _assemble(ham_r, pos_r, lat):
    """Stack dict data into arrays for fast Fourier sums.

    Returns R_int (nR,3 int), Rc (nR,3 cart, Ang), Hs (nR,N,N),
    As (nR,3,N,N) or None, inv_deg (nR,).

    The position matrix is Hermitianized,
    ``A_R = 1/2 (pos_R + conj(pos_{-R}^T))``, so that the Berry connection
    ``A(k) = sum_R e^{ikR} A_R`` is Hermitian -- this matches the
    ``1/2 (A + A^dagger)`` symmetrization that postw90's ``get_AA_R`` applies.
    """
    Rs = list(ham_r.keys())
    R_int = np.array(Rs, dtype=float)
    Rc = R_int @ np.asarray(lat, float)  # Cartesian R (Ang): sum_i R_i a_i
    Hs = np.stack([ham_r[R]["h"] for R in Rs], axis=0)
    inv_deg = np.array([1.0 / float(ham_r[R]["deg"]) for R in Rs])
    if pos_r is not None:
        Rset = set(Rs)
        As = np.zeros((len(Rs), 3, Hs.shape[-1], Hs.shape[-1]), complex)
        for k, R in enumerate(Rs):
            negR = (-R[0], -R[1], -R[2])
            Pn = pos_r[negR] if negR in Rset else np.zeros_like(pos_r[R])
            As[k] = 0.5 * (pos_r[R] + np.conj(np.transpose(Pn, (0, 2, 1))))
    else:
        As = None
    return R_int, Rc, Hs, As, inv_deg


def berry_curvature_w90(
    ham_r, pos_r, lat, k_red, fermi=None, *, occ_idxs=None, include_external=True
):
    """Band-summed -Omega (3-vector, Ang^2) at each reduced k-point.

    Parameters
    ----------
    ham_r : dict R -> {"h": (N,N) complex, "deg": int}
    pos_r : dict R -> (3,N,N) complex   (<0|r|R>, Ang) or None
    lat   : (3,3) Cartesian lattice vectors (Ang)
    k_red : (Nk,3) reduced k-points
    fermi : float, Fermi energy (eV)
    include_external : bool
        If True, use the full J0+J1+J2 formula. If False (or pos_r is None),
        return only J2 -- the diagonal-position-operator result.

    Returns
    -------
    mOmega : (Nk, 3) real array of -Omega_{x,y,z} in Ang^2.
    """
    R_int, Rc, Hs, As, inv_deg = _assemble(ham_r, pos_r, lat)
    k_red = np.atleast_2d(np.asarray(k_red, float))
    N = Hs.shape[-1]
    use_pos = include_external and As is not None

    out = np.zeros((k_red.shape[0], 3))
    for ik, k in enumerate(k_red):
        ph = np.exp(2j * np.pi * (R_int @ k)) * inv_deg  # (nR,)

        Hk = np.einsum("r, rij -> ij", ph, Hs)
        # velocity dH/dk_a with Cartesian R (Ang):  i * Rc_a * H(R)
        delH = np.einsum("r, ra, rij -> aij", ph, 1j * Rc, Hs)  # (3,N,N)

        E, U = np.linalg.eigh(Hk)
        if occ_idxs is not None:
            occm = np.zeros(N, bool)
            occm[np.asarray(occ_idxs, int)] = True
        else:
            occm = E < fermi
        occ = occm.astype(float)
        Uh = U.conj().T

        if use_pos:
            A = np.einsum("r, raij -> aij", ph, As)  # (3,N,N) Wannier connection
            # Omega_bar_gamma = sum_R ph * i (Rc_alpha A_R[beta] - Rc_beta A_R[alpha])
            Omega = np.zeros((3, N, N), complex)
            for g in range(3):
                a, b = ALPHA[g], BETA[g]
                Omega[g] = np.einsum(
                    "r, rij -> ij",
                    ph,
                    1j
                    * (
                        Rc[:, a, None, None] * As[:, b]
                        - Rc[:, b, None, None] * As[:, a]
                    ),
                )
            f = (U * occ) @ Uh  # occupied projector in W gauge

        # JJp/JJm: rotate delH to eigenbasis, keep empty(n)-occ(m) block, rotate back
        JJp = np.zeros((3, N, N), complex)
        JJm = np.zeros((3, N, N), complex)
        empty = ~occm
        # energy denominators
        for a in range(3):
            dbar = Uh @ delH[a] @ U  # (N,N) in eigenbasis
            Pp = np.zeros((N, N), complex)
            Pm = np.zeros((N, N), complex)
            # n empty, m occ
            ne = np.where(empty)[0]
            mo = np.where(occm)[0]
            if ne.size and mo.size:
                dE = E[mo][None, :] - E[ne][:, None]  # (ne,mo) = E_m - E_n
                Pp[np.ix_(ne, mo)] = 1j * dbar[np.ix_(ne, mo)] / dE
                Pm[np.ix_(mo, ne)] = 1j * dbar[np.ix_(mo, ne)] / (-dE.T)
            JJp[a] = U @ Pp @ Uh
            JJm[a] = U @ Pm @ Uh

        for g in range(3):
            a, b = ALPHA[g], BETA[g]
            J2 = -2.0 * np.imag(np.trace(JJm[a] @ JJp[b]))
            val = J2
            if use_pos:
                J0 = np.real(np.trace(f @ Omega[g]))
                J1 = -2.0 * (
                    np.imag(np.trace(A[a] @ JJp[b])) + np.imag(np.trace(JJm[a] @ A[b]))
                )
                val = J0 + J1 + J2
            out[ik, g] = val
    return out


def kpath_from_postw90(curv_dat, nodes_red, lat):
    """Reconstruct reduced k-points for each row of a postw90 ``-curv.dat``.

    The path is piecewise-linear between ``nodes_red`` (G, M, K, G, ...). Column
    0 of the file is the cumulative Cartesian path length; we invert it.
    """
    s = np.asarray(curv_dat)[:, 0]
    nodes_red = np.asarray(nodes_red, float)
    B = 2 * np.pi * np.linalg.inv(np.asarray(lat, float)).T  # rows b_i (1/Ang)
    nodes_cart = nodes_red @ B
    seg = np.linalg.norm(np.diff(nodes_cart, axis=0), axis=1)
    s_nodes = np.concatenate([[0.0], np.cumsum(seg)])
    # scale geometry to the file's total length (handles 2pi/units convention)
    s_geo = s * (s_nodes[-1] / s[-1])
    k_red = np.zeros((s.size, 3))
    for i, sv in enumerate(s_geo):
        j = np.searchsorted(s_nodes, sv, side="right") - 1
        j = min(max(j, 0), len(seg) - 1)
        t = (sv - s_nodes[j]) / seg[j] if seg[j] > 0 else 0.0
        k_red[i] = nodes_red[j] + t * (nodes_red[j + 1] - nodes_red[j])
    return k_red
