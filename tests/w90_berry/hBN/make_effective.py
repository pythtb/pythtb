"""Convert hBN_tb.dat -> postw90 effective-model inputs (_HH_R.dat, _AA_R.dat).

In ``effective_model = .true.`` mode postw90 reads H(R) and <0|r|R> straight
from these files (ndegen assumed 1, plain  sum_R e^{ikR}), bypassing the
get_AA_R reconstruction from .mmn. That makes postw90 and PythTB use *identical*
data, so the Berry-curvature comparison tests the formula alone.

File formats (Fortran '(5I5,2F12.6)' / '(5I5,6F12.6)'):
    _HH_R.dat : R1 R2 R3  i j  Re(H_ij) Im(H_ij)
    _AA_R.dat : R1 R2 R3  i j  Re(x) Im(x) Re(y) Im(y) Re(z) Im(z)
Values are pre-divided by the Wigner-Seitz degeneracy (ndegen=1 assumed).
"""

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "eff"
sys.path.insert(0, str(HERE.parents[2]))
from pythtb.io.w90 import read_tb  # noqa: E402

PREFIX = "hBN"
FERMI = -0.035


def write_win(nw, lat):
    latxt = "\n".join(f" {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}" for v in lat)
    (OUT / f"{PREFIX}.win").write_text(
        f"""effective_model = true
num_wann = {nw}
num_bands = {nw}
fermi_energy = {FERMI}
use_ws_distance = false

begin unit_cell_cart
Ang
{latxt}
end unit_cell_cart

mp_grid = 6 6 1

begin kpoint_path
G 0.0000000 0.0000000 0.0000000  M 0.5000000 0.0000000 0.0000000
M 0.5000000 0.0000000 0.0000000  K 0.3333333 0.3333333 0.0000000
K 0.3333333 0.3333333 0.0000000  G 0.0000000 0.0000000 0.0000000
end kpoint_path

kpath = true
kpath_task = curv
kpath_num_points = 200
kpath_bands_colour = none
"""
    )


def main():
    OUT.mkdir(exist_ok=True)
    nw, ham, pos, lat = read_tb(HERE, PREFIX)
    Rs = list(ham.keys())
    nrpts = len(Rs)
    write_win(nw, lat)

    with (OUT / f"{PREFIX}_HH_R.dat").open("w") as f:
        f.write(" effective-model HH_R generated from _tb.dat\n")
        f.write(f"{nw:12d}\n{nrpts:12d}\n")
        for R in Rs:
            deg = float(ham[R].degeneracy)
            H = ham[R].h / deg
            for i in range(nw):
                for j in range(nw):
                    f.write(
                        "%5d%5d%5d%5d%5d%12.6f%12.6f\n"
                        % (R[0], R[1], R[2], i + 1, j + 1, H[i, j].real, H[i, j].imag)
                    )

    with (OUT / f"{PREFIX}_AA_R.dat").open("w") as f:
        f.write(" effective-model AA_R generated from _tb.dat\n")
        f.write(f"{nw:12d}\n{nrpts:12d}\n")
        for R in Rs:
            deg = float(ham[R].degeneracy)
            X = pos[R] / deg  # (3, nw, nw)
            for i in range(nw):
                for j in range(nw):
                    f.write(
                        "%5d%5d%5d%5d%5d%12.6f%12.6f%12.6f%12.6f%12.6f%12.6f\n"
                        % (
                            R[0],
                            R[1],
                            R[2],
                            i + 1,
                            j + 1,
                            X[0, i, j].real,
                            X[0, i, j].imag,
                            X[1, i, j].real,
                            X[1, i, j].imag,
                            X[2, i, j].real,
                            X[2, i, j].imag,
                        )
                    )
    print(
        f"wrote {PREFIX}_HH_R.dat and {PREFIX}_AA_R.dat  (num_wann={nw}, nrpts={nrpts})"
    )


if __name__ == "__main__":
    main()
