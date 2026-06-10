"""Validate the position-matrix Berry-curvature correction for hBN.

Compares, along the G-M-K-G path:
  * postw90 reference (-Omega_z, from hBN-curv.dat),
  * PythTB WITHOUT the correction (diagonal position operator),
  * PythTB WITH the correction (off-diagonal position matrix from write_tb).

For hBN's complete 4-valence Wannier manifold the uncorrected result is
identically zero, so the entire curve is the correction -- and it should match
postw90 to numerical precision.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # for berry_wannier
sys.path.insert(0, str(HERE.parents[2]))  # repo root for pythtb

from pythtb import W90  # noqa: E402
from berry_wannier import kpath_from_postw90  # noqa: E402

# All 4 valence bands occupied (a complete Wannier manifold).
OCC = [0, 1, 2, 3]
NODES = [[0, 0, 0], [0.5, 0, 0], [1 / 3, 1 / 3, 0], [0, 0, 0]]  # G M K G


def main():
    w90 = W90(str(HERE), "hBN")
    assert w90.has_position_matrix, "hBN_tb.dat not read -- did write_tb run?"
    tb = w90.model()  # the correction lives on TBModel.berry_curvature

    ref = np.loadtxt(HERE / "hBN-curv.dat")  # cols: s, -Wx, -Wy, -Wz
    k_red = kpath_from_postw90(ref, NODES, w90.lat)

    # plane=(0,1) -> Omega_z (Cartesian, Ang^2). Auto correction for write_tb models.
    yes_z = tb.berry_curvature(k_red, occ_idxs=OCC, plane=(0, 1), cartesian=True)
    no_z = tb.berry_curvature(
        k_red, occ_idxs=OCC, plane=(0, 1), cartesian=True, include_external=False
    )
    # postw90 prints -Omega_z; berry_curvature returns +Omega_z.
    ref_z = -ref[:, 3]

    err = np.max(np.abs(yes_z - ref_z))
    scale = np.max(np.abs(ref_z))
    corr = np.corrcoef(yes_z, ref_z)[0, 1]
    fit = np.sum(yes_z * ref_z) / np.sum(yes_z * yes_z)
    print(f"  k-points on path        : {k_red.shape[0]}")
    print(f"  max|postw90 Omega_z|    : {scale:.6f} Ang^2")
    print(f"  max|uncorrected|        : {np.max(np.abs(no_z)):.3e} Ang^2  (expect ~0)")
    print(f"  max|corrected - postw90|: {err:.3e} Ang^2")
    print(f"  peak relative error     : {err / scale:.3e}")
    print(f"  correlation             : {corr:.6f}")
    print(f"  best-fit scale           : {fit:.5f}  (expect ~1)")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        s = ref[:, 0]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(s, ref_z, "k-", lw=3, alpha=0.4, label="postw90 (reference)")
        ax.plot(s, yes_z, "r--", lw=1.5, label="PythTB + position correction")
        ax.plot(s, no_z, "b:", lw=1.5, label="PythTB, diagonal approx")
        ax.set_xlabel("k-path  G - M - K - G")
        ax.set_ylabel(r"$-\Omega_z(k)$  [$\AA^2$]")
        ax.legend()
        ax.set_title("Monolayer hBN: Berry curvature of the valence manifold")
        fig.tight_layout()
        fig.savefig(HERE / "hBN_berry_curvature.png", dpi=130)
        print(f"  wrote {HERE / 'hBN_berry_curvature.png'}")
    except Exception as e:  # pragma: no cover
        print("  (plot skipped:", e, ")")

    # Agreement is limited by the W90-internal discretization (~0.2%): the
    # _tb.dat values are a fixed-mesh discretized position matrix and postw90
    # symmetrizes/transl-invariants on the same mesh. The correction reproduces
    # the reference to <1%, while the diagonal approximation gives exactly 0.
    ok = (corr > 0.999) and (abs(fit - 1.0) < 0.01) and (np.max(np.abs(no_z)) < 1e-9)
    print("  RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
