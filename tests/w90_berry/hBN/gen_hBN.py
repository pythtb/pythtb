#!/usr/bin/env python3
"""Generate Quantum ESPRESSO + Wannier90 input files for monolayer hBN.

Monolayer hBN (hexagonal, broken inversion symmetry) has a nonzero, valley-
localized Berry curvature in its (isolated) 4-band valence manifold. We
Wannierize those 4 valence bands (num_bands == num_wann, no disentanglement),
write the position matrix via ``write_tb = .true.``, and use postw90's berry
module to produce a reference Berry curvature along a k-path.

Run order (see run.sh):
    pw.x scf -> pw.x nscf -> wannier90 -pp -> pw2wannier90 -> wannier90 -> postw90
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PSEUDO_DIR = "/Users/treycole/Codes/qe-7.4.1/pseudo"
PREFIX = "hBN"

# Geometry (Angstrom). a = in-plane lattice constant, c = vacuum.
A = 2.504
C = 10.0
ECUTWFC = 50.0
ECUTRHO = 400.0
NBND_NSCF = 8  # compute a few conduction bands; W90 excludes them
MP = (6, 6, 1)  # Wannier uniform k-grid
SCF_K = (12, 12, 1)  # denser grid for the self-consistent step


a1 = np.array([A, 0.0, 0.0])
a2 = np.array([-A / 2.0, A * np.sqrt(3.0) / 2.0, 0.0])
a3 = np.array([0.0, 0.0, C])

# Honeycomb sublattices in crystal coordinates.
ATOMS = [
    ("B", 0.0, 0.0, 0.5),
    ("N", 1.0 / 3.0, 2.0 / 3.0, 0.5),
]


def uniform_kgrid(mp):
    pts = []
    for i in range(mp[0]):
        for j in range(mp[1]):
            for k in range(mp[2]):
                pts.append((i / mp[0], j / mp[1], k / mp[2]))
    return pts


def write_scf():
    txt = (
        f"""&control
    calculation = 'scf'
    prefix      = '{PREFIX}'
    outdir      = './out'
    pseudo_dir  = '{PSEUDO_DIR}'
    verbosity   = 'high'
/
&system
    ibrav     = 4
    A         = {A}
    C         = {C}
    nat       = 2
    ntyp      = 2
    ecutwfc   = {ECUTWFC}
    ecutrho   = {ECUTRHO}
    occupations = 'fixed'
/
&electrons
    conv_thr     = 1.0d-10
    mixing_beta  = 0.7
/
ATOMIC_SPECIES
    B  10.811  B.pbe-n-rrkjus_psl.1.0.0.UPF
    N  14.007  N.pbe-n-rrkjus_psl.1.0.0.UPF
ATOMIC_POSITIONS crystal
"""
        + "".join(f"    {s}  {x:.10f}  {y:.10f}  {z:.10f}\n" for (s, x, y, z) in ATOMS)
        + f"""K_POINTS automatic
    {SCF_K[0]} {SCF_K[1]} {SCF_K[2]} 0 0 0
"""
    )
    (HERE / "scf.in").write_text(txt)


def write_nscf():
    kpts = uniform_kgrid(MP)
    w = 1.0 / len(kpts)
    head = f"""&control
    calculation = 'nscf'
    prefix      = '{PREFIX}'
    outdir      = './out'
    pseudo_dir  = '{PSEUDO_DIR}'
    verbosity   = 'high'
/
&system
    ibrav     = 4
    A         = {A}
    C         = {C}
    nat       = 2
    ntyp      = 2
    ecutwfc   = {ECUTWFC}
    ecutrho   = {ECUTRHO}
    nbnd      = {NBND_NSCF}
    occupations = 'fixed'
    nosym     = .true.
    noinv     = .true.
/
&electrons
    conv_thr     = 1.0d-10
    mixing_beta  = 0.7
/
ATOMIC_SPECIES
    B  10.811  B.pbe-n-rrkjus_psl.1.0.0.UPF
    N  14.007  N.pbe-n-rrkjus_psl.1.0.0.UPF
ATOMIC_POSITIONS crystal
""" + "".join(f"    {s}  {x:.10f}  {y:.10f}  {z:.10f}\n" for (s, x, y, z) in ATOMS)
    body = f"K_POINTS crystal\n    {len(kpts)}\n"
    for x, y, z in kpts:
        body += f"    {x:.10f}  {y:.10f}  {z:.10f}  {w:.10f}\n"
    (HERE / "nscf.in").write_text(head + body)


def write_pw2wan():
    txt = f"""&inputpp
    outdir     = './out'
    prefix     = '{PREFIX}'
    seedname   = '{PREFIX}'
    write_mmn  = .true.
    write_amn  = .true.
    write_unk  = .false.
/
"""
    (HERE / f"{PREFIX}.pw2wan.in").write_text(txt)


def write_win(fermi_energy=None):
    kpts = uniform_kgrid(MP)
    lat = "\n".join(f" {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}" for v in (a1, a2, a3))
    kblock = "\n".join(f" {x:.10f} {y:.10f} {z:.10f}" for (x, y, z) in kpts)
    fermi_line = (
        f"fermi_energy = {fermi_energy:.6f}\n" if fermi_energy is not None else ""
    )
    txt = (
        f"""! Monolayer hBN: 4 isolated valence bands -> 4 MLWFs.
num_wann  = 4
num_bands = 4
exclude_bands = 5-{NBND_NSCF}

! ---- outputs needed by the PythTB W90 reader ----
write_hr   = true
write_xyz  = true
write_tb   = true          ! <-- off-diagonal position matrix  <0n|r|Rm>
translate_home_cell = false

num_iter          = 400
num_print_cycles  = 40
conv_tol          = 1.0e-10
conv_window       = 5

begin atoms_frac
"""
        + "".join(f" {s}  {x:.10f} {y:.10f} {z:.10f}\n" for (s, x, y, z) in ATOMS)
        + f"""end atoms_frac

begin projections
N : sp2
N : pz
end projections

begin unit_cell_cart
Ang
{lat}
end unit_cell_cart

mp_grid = {MP[0]} {MP[1]} {MP[2]}

begin kpoints
{kblock}
end kpoints

begin kpoint_path
G 0.0000000 0.0000000 0.0000000  M 0.5000000 0.0000000 0.0000000
M 0.5000000 0.0000000 0.0000000  K 0.3333333 0.3333333 0.0000000
K 0.3333333 0.3333333 0.0000000  G 0.0000000 0.0000000 0.0000000
end kpoint_path

! ---- postw90 reference: Berry curvature along the k-path ----
! The kpath module (kpath_task = curv) reuses the kpoint_path block above and
! writes the band-summed Berry curvature; it needs the Fermi level to know
! which bands are occupied.
! use_ws_distance=false -> plain  sum_R exp(i k.R)  interpolation, matching the
! PythTB reader exactly (so the two implementations can be compared directly).
use_ws_distance = false
{fermi_line}kpath = true
kpath_task = curv
kpath_num_points = 200
kpath_bands_colour = none
"""
    )
    (HERE / f"{PREFIX}.win").write_text(txt)


if __name__ == "__main__":
    write_scf()
    write_nscf()
    write_pw2wan()
    write_win()
    print("Wrote scf.in, nscf.in, hBN.pw2wan.in, hBN.win in", HERE)
