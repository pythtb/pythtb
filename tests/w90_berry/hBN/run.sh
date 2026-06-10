#!/usr/bin/env bash
# Full pipeline for the monolayer-hBN Berry-curvature benchmark.
#
# Produces hBN_tb.dat (Hamiltonian + position matrix) and the postw90 reference
# Berry curvature hBN-curv.dat, then validates the PythTB position-matrix
# correction against it (verify.py).
set -euo pipefail
cd "$(dirname "$0")"

# --- toolchain ---------------------------------------------------------------
# This machine's wannier90.x/postw90.x crash in ZDOTC (Apple Accelerate <->
# gfortran complex-return ABI bug). The shim in ../libdotfix.dylib fixes it.
export OMP_NUM_THREADS=1
SHIM="$(cd .. && pwd)/libdotfix.dylib"
[[ -f "$SHIM" ]] || cc -O2 -dynamiclib -o "$SHIM" ../dotfix.c
export DYLD_INSERT_LIBRARIES="$SHIM"
export DYLD_FORCE_FLAT_NAMESPACE=1
NP="${NP:-4}"
PY="${PY:-../../../.venv/bin/python}"

$PY gen_hBN.py                                  # write QE/W90 inputs
mpirun -np "$NP" pw.x          -in scf.in   > scf.out
mpirun -np "$NP" pw.x          -in nscf.in  > nscf.out
EF=$(awk '/highest occupied/{print ($(NF-1)+$NF)/2}' nscf.out)   # mid-gap
$PY -c "import gen_hBN as g; g.write_win(fermi_energy=$EF)"
wannier90.x -pp hBN
mpirun -np "$NP" pw2wannier90.x -in hBN.pw2wan.in > pw2wan.out
wannier90.x hBN                                  # -> hBN_tb.dat, _hr.dat, _centres.xyz
postw90.x  hBN                                   # -> hBN-curv.dat (reference)

$PY verify.py                                    # PythTB (with/without) vs postw90
