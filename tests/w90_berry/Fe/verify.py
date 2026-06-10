"""Validate the Berry-curvature correction for bcc Fe (SOC) along G-H-P-N-G.

Unlike hBN, Fe is a metal: the occupied set varies with k, so the J1/J2 terms
of the Wang-Yates-Souza-Vanderbilt formula are active. This validates the
*full* position-matrix correction (not just the curl term) against postw90.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))

from pythtb import W90  # noqa: E402
from berry_wannier import kpath_from_postw90  # noqa: E402

FERMI = 17.4654
NODES = [
    [0.0, 0.0, 0.0],
    [0.5, -0.5, -0.5],
    [0.75, 0.25, -0.25],
    [0.5, 0.0, -0.5],
    [0.0, 0.0, 0.0],
]  # G H P N G


def main():
    w90 = W90(str(HERE), "Fe")
    assert w90.has_position_matrix
    tb = w90.model()  # the correction lives on TBModel.berry_curvature

    ref = np.loadtxt(HERE / "Fe-curv.dat")  # s, -Wx, -Wy, -Wz
    k_red = kpath_from_postw90(ref, NODES, w90.lat)

    # Metal -> per-k occupation set by the Fermi level. TBModel.berry_curvature
    # reuses the QGT with an occupation-weighted (Fermi) band sum; plane=(0,1)
    # -> Omega_z. include_external is auto-applied for write_tb models.
    yes = tb.berry_curvature(k_red, fermi=FERMI, plane=(0, 1), cartesian=True)
    no = tb.berry_curvature(
        k_red, fermi=FERMI, plane=(0, 1), cartesian=True, include_external=False
    )
    ref_z = -ref[:, 3]  # postw90 prints -Omega

    scale = np.max(np.abs(ref_z))
    corr = np.corrcoef(yes, ref_z)[0, 1]
    fit = np.sum(yes * ref_z) / np.sum(yes * yes)
    err = np.max(np.abs(yes - ref_z))
    corr_no = np.corrcoef(no, ref_z)[0, 1]
    print(f"  k-points on path        : {k_red.shape[0]}")
    print(f"  max|postw90 Omega_z|    : {scale:.2f} Ang^2")
    print(
        f"  WITH correction : corr={corr:.5f}  fit_scale={fit:.4f}  peak_relerr={err / scale:.2e}"
    )
    print(f"  WITHOUT (diag)  : corr={corr_no:.5f}  (J2 only - misses J0,J1)")

    # Integrated anomalous Hall conductivity on a uniform 25^3 BZ mesh
    # (matches postw90's berry_kmesh; densify + adaptive to converge to ~750).
    n = 25
    g = np.arange(n) / n
    KX, KY, KZ = np.meshgrid(g, g, g, indexing="ij")
    kmesh = np.stack([KX.ravel(), KY.ravel(), KZ.ravel()], axis=1)
    e_SI, hbar_SI = 1.602176634e-19, 1.054571817e-34
    Vc = abs(np.linalg.det(w90.lat))  # Ang^3
    fac = -1.0e8 * e_SI**2 / (hbar_SI * Vc)  # S/cm
    sig_yes = fac * np.mean(
        -tb.berry_curvature(kmesh, fermi=FERMI, plane=(0, 1), cartesian=True)
    )
    sig_no = fac * np.mean(
        -tb.berry_curvature(
            kmesh, fermi=FERMI, plane=(0, 1), cartesian=True, include_external=False
        )
    )
    print(
        f"  AHC sigma_xy (25^3): with corr = {sig_yes:8.2f} S/cm,  "
        f"diagonal = {sig_no:8.2f} S/cm"
    )
    print("    (postw90 plain 25^3 reference: |sigma_xy| = 1224.82 S/cm)")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        s = ref[:, 0]
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.plot(s, ref_z, "k-", lw=3, alpha=0.4, label="postw90 (reference)")
        ax.plot(s, yes, "r--", lw=1.2, label="PythTB + position correction")
        ax.plot(s, no, "b:", lw=1.2, label="PythTB, diagonal approx")
        ax.set_xlabel("k-path  G - H - P - N - G")
        ax.set_ylabel(r"$\Omega_z(k)$  [$\AA^2$]")
        ax.set_title("bcc Fe (SOC): Berry curvature, sum over occupied")
        ax.legend()
        fig.tight_layout()
        fig.savefig(HERE / "Fe_berry_curvature.png", dpi=130)
        print(f"  wrote {HERE / 'Fe_berry_curvature.png'}")
    except Exception as e:  # pragma: no cover
        print("  (plot skipped:", e, ")")

    ok = corr > 0.95 and abs(fit - 1.0) < 0.1
    print("  RESULT:", "PASS" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
