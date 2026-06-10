#!/usr/bin/env bash
# Full pipeline for the bcc-Fe (SOC) anomalous-Hall benchmark.
#
# Produces Fe_tb.dat (Hamiltonian + position matrix) and the postw90 reference
# (AHC in Fe.wpout, Berry curvature in Fe-curv.dat), then validates PythTB's
# position-matrix correction against them (verify.py).
#
# Heavier than hBN: noncollinear + spin-orbit, 18 spinor Wannier functions.
set -euo pipefail
cd "$(dirname "$0")"

export OMP_NUM_THREADS=1
SHIM="$(cd .. && pwd)/libdotfix.dylib"   # Apple Accelerate ZDOTC ABI fix
[[ -f "$SHIM" ]] || cc -O2 -dynamiclib -o "$SHIM" ../dotfix.c
export DYLD_INSERT_LIBRARIES="$SHIM"
export DYLD_FORCE_FLAT_NAMESPACE=1
NP="${NP:-6}"
PY="${PY:-../../../.venv/bin/python}"

$PY gen_Fe.py
mpirun -np "$NP" pw.x          -in scf.in   > scf.out
EF=$(awk '/the Fermi energy is/{print $(NF-1)}' scf.out)
mpirun -np "$NP" pw.x          -in nscf.in  > nscf.out
$PY -c "import gen_Fe as g; g.write_win(fermi_energy=$EF)"
wannier90.x -pp Fe
mpirun -np "$NP" pw2wannier90.x -in Fe.pw2wan.in > pw2wan.out
wannier90.x Fe                                 # -> Fe_tb.dat
postw90.x  Fe                                  # -> AHC (Fe.wpout), Fe-curv.dat

$PY verify.py
