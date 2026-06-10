#!/usr/bin/env python3
"""Generate QE + Wannier90 inputs for bcc Fe with spin-orbit coupling.

This is the classic anomalous-Hall benchmark (Wannier90 example 18): bcc Fe,
magnetized along [001], spin-orbit coupling on. The intrinsic anomalous Hall
conductivity is sigma_xy ~ 750 S/cm (Wang et al., PRB 74, 195118 (2006)).

Unlike hBN (a complete, insulating Wannier manifold where only the J0 position
term survives), Fe is a metal: the occupied set varies with k and the J1/J2
terms of the Berry-curvature formula are active. So this exercises the *full*
position-matrix correction, not just the curl term.

The Wannier manifold is the usual 18 spinor functions (s,p,d on Fe). Grids here
are moderate so the pipeline runs in minutes; densify mp_grid + the AHC mesh to
converge sigma_xy to the ~750 S/cm reference.

Run order (see run.sh):
    pw.x scf -> pw.x nscf -> wannier90 -pp -> pw2wannier90 -> wannier90 -> postw90
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PSEUDO_DIR = "/Users/treycole/Codes/qe-7.4.1/pseudo"
PREFIX = "Fe"

A = 2.8665  # bcc Fe lattice constant (Ang)
ECUTWFC = 60.0
ECUTRHO = 600.0
NBND_NSCF = 28  # > num_wann=18, room for disentanglement
MP = (4, 4, 4)  # Wannier uniform grid (coarse but tractable)
SCF_K = (8, 8, 8)


def uniform_kgrid(mp):
    return [
        (i / mp[0], j / mp[1], k / mp[2])
        for i in range(mp[0])
        for j in range(mp[1])
        for k in range(mp[2])
    ]


def _system_block(calc, extra=""):
    return f"""&control
    calculation = '{calc}'
    prefix      = '{PREFIX}'
    outdir      = './out'
    pseudo_dir  = '{PSEUDO_DIR}'
    verbosity   = 'high'
/
&system
    ibrav        = 3
    A            = {A}
    nat          = 1
    ntyp         = 1
    ecutwfc      = {ECUTWFC}
    ecutrho      = {ECUTRHO}
    occupations  = 'smearing'
    smearing     = 'cold'
    degauss      = 0.02
    noncolin     = .true.
    lspinorb     = .true.
    starting_magnetization(1) = 0.4
    angle1(1)    = 0.0
    angle2(1)    = 0.0
{extra}/
&electrons
    conv_thr     = 1.0d-9
    mixing_beta  = 0.3
/
ATOMIC_SPECIES
    Fe  55.845  Fe.rel-pbe-spn-rrkjus_psl.0.2.1.UPF
ATOMIC_POSITIONS crystal
    Fe  0.0  0.0  0.0
"""


def write_scf():
    txt = (
        _system_block("scf")
        + f"K_POINTS automatic\n    {SCF_K[0]} {SCF_K[1]} {SCF_K[2]} 0 0 0\n"
    )
    (HERE / "scf.in").write_text(txt)


def write_nscf():
    kpts = uniform_kgrid(MP)
    w = 1.0 / len(kpts)
    txt = _system_block(
        "nscf",
        extra=f"    nbnd = {NBND_NSCF}\n    nosym = .true.\n    noinv = .true.\n",
    )
    txt += f"K_POINTS crystal\n    {len(kpts)}\n"
    for x, y, z in kpts:
        txt += f"    {x:.10f}  {y:.10f}  {z:.10f}  {w:.10f}\n"
    (HERE / "nscf.in").write_text(txt)


def write_pw2wan():
    (HERE / f"{PREFIX}.pw2wan.in").write_text(
        f"""&inputpp
    outdir    = './out'
    prefix    = '{PREFIX}'
    seedname  = '{PREFIX}'
    write_mmn = .true.
    write_amn = .true.
    write_unk = .false.
    write_spn = .false.
/
"""
    )


def write_win(fermi_energy=None):
    kpts = uniform_kgrid(MP)
    kblock = "\n".join(f" {x:.10f} {y:.10f} {z:.10f}" for (x, y, z) in kpts)
    fermi_line = (
        f"fermi_energy = {fermi_energy:.6f}\n" if fermi_energy is not None else ""
    )
    # bcc primitive lattice vectors (Ang), matching QE ibrav=3 exactly:
    #   a1 = (A/2)(1,1,1), a2 = (A/2)(-1,1,1), a3 = (A/2)(-1,-1,1)
    a1 = np.array([A / 2, A / 2, A / 2])
    a2 = np.array([-A / 2, A / 2, A / 2])
    a3 = np.array([-A / 2, -A / 2, A / 2])
    latxt = "\n".join(f" {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}" for v in (a1, a2, a3))
    # The fully-relativistic pseudo carries semicore 3s,3p in valence (8 deep
    # spinor bands well below E_F). Exclude them so the s,p,d Wannier manifold
    # is the 18 bands around the Fermi level.
    txt = f"""! bcc Fe, spin-orbit, magnetised along [001]: 18 spinor MLWFs (s,p,d).
num_wann  = 18
num_bands = {NBND_NSCF - 8}
exclude_bands = 1-8
spinors   = .true.

write_hr  = true
write_xyz = true
write_tb  = true            ! off-diagonal position matrix <0n|r|Rm>
translate_home_cell = false
use_ws_distance = false
transl_inv = true

dis_win_max  = 45.0
dis_froz_max = 21.0
dis_num_iter = 200
num_iter     = 400
num_print_cycles = 40

begin atoms_frac
Fe  0.0  0.0  0.0
end atoms_frac

begin projections
Fe : s; p; d
end projections

begin unit_cell_cart
Ang
{latxt}
end unit_cell_cart

mp_grid = {MP[0]} {MP[1]} {MP[2]}

begin kpoints
{kblock}
end kpoints

begin kpoint_path
G 0.00 0.00 0.00  H 0.50 -0.50 -0.50
H 0.50 -0.50 -0.50  P 0.75 0.25 -0.25
P 0.75 0.25 -0.25  N 0.50 0.00 -0.50
N 0.50 0.00 -0.50  G 0.00 0.00 0.00
end kpoint_path

! ---- postw90 reference: anomalous Hall conductivity + curvature on path ----
{fermi_line}berry = true
berry_task = ahc
berry_kmesh = 25 25 25
berry_curv_adpt_kmesh = 5
berry_curv_adpt_kmesh_thresh = 100.0

kpath = true
kpath_task = curv
kpath_num_points = 200
kpath_bands_colour = none
"""
    (HERE / f"{PREFIX}.win").write_text(txt)


if __name__ == "__main__":
    write_scf()
    write_nscf()
    write_pw2wan()
    write_win()
    print("Wrote scf.in, nscf.in, Fe.pw2wan.in, Fe.win in", HERE)
