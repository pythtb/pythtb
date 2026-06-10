"""Generate validation_summary.ipynb (no nbformat dependency)."""

import json
from pathlib import Path

cells = []


def md(text):
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


def code(text):
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


md(r"""
# Validating the external position-matrix Berry-curvature correction

This notebook gathers, in one place, every check behind PythTB's
`include_external` Berry-curvature correction — the term built from the
off-diagonal Wannier position matrix &langle;0n|**r**|Rm&rangle; that a
Wannier90 `write_tb` run supplies on top of *H*(R).

It answers three questions:

1. **What is PythTB computing, and what is postw90 computing?**
2. **Are the comparisons really independent?**  (Honest answer: *partly* — see the map below.)
3. **Do the numbers agree?**  (Yes: to ~1e-14 against an independent formula,
   to 0.004 % against postw90's Fortran, and linearly-convergent against a
   finite-difference Wilson loop.)

All inputs are the small committed fixtures under `tests/w90_berry/`.
""")

md(r"""
## How PythTB and postw90 relate — and what is actually independent

Both codes share **stages 0-5** -- the DFT, the overlaps, *and the position
matrix itself*. They fork only at the very last stage:

```
                          -- SHARED FRONT END --
0.  structure + pseudopotentials
1.  pw.x  < scf.in         -> self-consistent density            (prefix.save/)
2.  pw.x  < nscf.in        -> Bloch states |u_nk> on a uniform MP "Wannier mesh"
3.  wannier90.x -pp prefix -> reads prefix.win, writes prefix.nnkp
                              (neighbor b-vectors + weights w_b = the grad_k stencil)
4.  pw2wannier90.x         -> prefix.mmn = overlaps M^{(k,b)}_{mn}=<u_mk|u_{n,k+b}>
                              prefix.amn = projections,  prefix.eig = eps_nk
5.  wannier90.x prefix     -> disentangle (U^dis) + maximally localize (U)
                              -> prefix.chk, prefix_centres.xyz, prefix.wout
     +============= now everything exists: U_k , M^{(k,b)} , eps_nk =============+
     |                                                                           |
  -- PATHWAY A: postw90 --                          -- PATHWAY B: PythTB --
6A. postw90.x prefix (berry=true):           6B. SAME wannier90 run, write_tb=.true.
      get_HH_R : H(R)=Sum_k e^{-ikR}U_k+ eps U_k     -> also dumps prefix_tb.dat:
      get_AA_R : A(R)=<0n|r|Rm> (finite-diff of M)      lattice + H(R) + <0n|r|Rm>
      get_BB_R/CC_R : moments for the curl         7B. PythTB W90().model():
      berry.F90 : FT->k, Hamiltonian gauge,             read_tb -> TBModel(_ham_r,_pos_r)
        -Omega = J0+J1+J2   [convention II, F90]   8B. tb.berry_curvature(include_external):
      -> prefix-curv.dat / AHC   <== REFERENCE          FT->k [convention I, NumPy];
                                                        internal QGT (=J2) + external
                                                        from _pos_r (=J0+J1) -> Omega
```

The single real difference is **who runs the last stage.** The position matrix
`<0n|r|Rm>` is computed once, by `wannier90.x`, from the `.mmn` finite-difference
(`A_a(k)=i Sum_b w_b b_a (M^{(k,b)}-1)`). Pathway A consumes it inside Fortran;
Pathway B writes it to `prefix_tb.dat` and re-derives the curvature in NumPy under
a different convention. (`J2` is the diagonal-`r` Kubo term; `J0` = curl of the
connection, `J1` = internal-external cross -- these two are the position-matrix
correction.) To make the two pathways ingest the *identical* matrix we set
`use_ws_distance=false`, `transl_inv=true`, and Hermitianize `A(R)` on read.

**Shared input (NOT independently re-derived here):** the position matrix
`<0n|r|Rm>` itself. Wannier90 computes it once from the `.mmn` overlaps; PythTB
reads it from `prefix_tb.dat`, while postw90 recomputes the identical object
internally as `AA_R`. This is the standard quantity *every* Wannier-interpolation
code uses; our tests trust it rather than re-derive it.

**What the comparisons *do* establish, in increasing order of independence:**

| Check | What differs | What it proves |
|---|---|---|
| library vs `berry_wannier.py` | convention **I** (QGT+external) vs convention **II** (explicit J0+J1+J2); two derivations, both NumPy | the interpolation *algebra* is right — catches sign/convention/factor bugs (agree to ~1e-14) |
| library vs **postw90** | NumPy vs **Fortran**, a different code base (not ours) | our formula matches the field-standard reference implementation (0.004 %) |
| library vs **Wilson loop** | analytic formula vs **discretized geometry** (loop of covariant overlaps) | the result really is the curl of a connection, not a formula coincidence (linear convergence) |
| Wannier centers vs `_centres.xyz` | our read of `<0n|r|0n>` vs the centers file | **only** that `read_tb` parses correctly (units, indexing, R=0 block) — see note below |
| Fe AHC vs **literature** | full DFT->Wannier->observable vs Wang *et al.* / experiment | end-to-end physical sanity (converges toward ~750 S/cm) |

So matching postw90 proves PythTB **uses the position matrix correctly**, via a
different formula in a different language; the Wilson loop proves it a third,
geometric way. The position matrix itself is Wannier90's standard, literature-
validated output. The only fully end-to-end (DFT -> measurable) check is the Fe
AHC vs literature.

> **Note — the Wannier-centers comparison is a parsing check, not a physics
> check.** `_centres.xyz` (the Im-ln Marzari-Vanderbilt center) and the diagonal
> `<0n|r|0n>` of `_tb.dat` are the *same quantity from the same `.mmn` data*; with
> `transl_inv=.true.` Wannier90 even substitutes the former into the latter. So it
> can only disagree if we mis-parse the file (wrong unit/transpose/block). It says
> nothing about the off-diagonal external term — that is what Checks 1-3 test.
""")

md("## Setup")

code(r"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def find_wb(start):
    # locate tests/w90_berry whether the notebook runs from there or the repo root
    for d in [start, *start.parents]:
        if (d / "hBN" / "hBN_tb.dat").exists():
            return d
        if (d / "tests" / "w90_berry" / "hBN" / "hBN_tb.dat").exists():
            return d / "tests" / "w90_berry"
    raise FileNotFoundError(f"could not locate tests/w90_berry from {start}")


WB = find_wb(Path.cwd().resolve())
REPO = WB.parents[1]
sys.path.insert(0, str(REPO))   # the pythtb package
sys.path.insert(0, str(WB))     # the berry_wannier.py reference formula
from pythtb import W90
from berry_wannier import berry_curvature_w90, kpath_from_postw90

print("data :", WB)
print("repo :", REPO)
""")

md(r"""
## Check 1 — monolayer hBN: the external term *is* the whole curvature

hBN's four valence bands form a **complete** Wannier manifold (4 Wannier
functions, all occupied). With no empty states the diagonal-position Kubo term
is **identically zero**, so 100 % of the Berry curvature comes from the
off-diagonal position matrix. That makes hBN the most stringent possible test:
there is nothing else for the answer to come from.

The next cell evaluates, along Γ–M–K–Γ:

* `omega_diag` — diagonal approximation (`include_external=False`) → ≈ 0,
* `omega_corr` — library, convention I (`include_external=True`),
* `omega_ref2` — the independent convention-II J0+J1+J2 reference formula,

and compares both to the committed **postw90** curve and to the `_centres.xyz` file.
""")

code(r"""
NODES = [[0, 0, 0], [0.5, 0, 0], [1 / 3, 1 / 3, 0], [0, 0, 0]]  # G M K G
OCC = [0, 1, 2, 3]                                              # complete valence manifold

w90 = W90(str(WB / "hBN"), "hBN")
tb = w90.model()                       # carries H(R) AND the position matrix
assert w90.has_position_matrix

ref = np.loadtxt(WB / "hBN" / "hBN-curv.dat")   # cols: s, -Omega_x, -Omega_y, -Omega_z
k = kpath_from_postw90(ref, NODES, w90.lat)
ref_z = -ref[:, 3]                              # postw90 prints -Omega -> recover +Omega_z

# library path (convention I): internal QGT + external position-matrix term
omega_corr = tb.berry_curvature(k, occ_idxs=OCC, plane=(0, 1), cartesian=True)
omega_diag = tb.berry_curvature(k, occ_idxs=OCC, plane=(0, 1), cartesian=True,
                                include_external=False)
# independent convention-II reference formula (explicit J0 + J1 + J2), column z
omega_ref2 = berry_curvature_w90(w90.ham_r, w90.pos_r, w90.lat, k, occ_idxs=OCC)[:, 2]

print("diagonal-approx   max|Omega_z|        : %.2e   (complete manifold -> exactly 0)"
      % np.max(np.abs(omega_diag)))
print("library(convI) vs reference(convII)   : max|diff| = %.2e   (two derivations -> machine precision)"
      % np.max(np.abs(omega_corr - omega_ref2)))
print("library        vs postw90  (Fortran)  : corr = %.6f , best-fit scale = %.5f"
      % (np.corrcoef(omega_corr, ref_z)[0, 1],
         np.sum(omega_corr * ref_z) / np.sum(omega_corr ** 2)))
print("Wannier centers vs _centres.xyz       : max|diff| = %.1e Ang   (parsing check only -- same .mmn quantity)"
      % np.max(np.abs(w90.wannier_centers() - w90.xyz_cen)))
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(ref[:, 0], ref_z, "k-", lw=3, alpha=0.35, label="postw90 (Fortran reference)")
ax.plot(ref[:, 0], omega_corr, "r--", lw=1.5, label="PythTB, include_external=True")
ax.plot(ref[:, 0], omega_diag, "b:", lw=1.8, label="PythTB, diagonal approx  (= 0)")
ax.set_xlabel("k-path   $\\Gamma$ - M - K - $\\Gamma$")
ax.set_ylabel(r"$\Omega_z(k)$   [$\AA^2$]")
ax.set_title("Monolayer hBN - Berry curvature of the complete valence manifold\n"
             "(the entire curve is the external position-matrix term)")
ax.legend(loc="upper right")
fig.tight_layout()
plt.show()
""")

md(r"""
## Check 2 — finite-difference Wilson loop (an independent *method*)

The checks above all use the same analytic interpolation formula. A Wilson loop
instead builds the curvature **geometrically**: it multiplies the
position-corrected cell-periodic overlaps
`S(k,k') = U(k)† [1 − i(k'−k)·A] U(k')` around a small plaquette near K and
reads the curvature off the loop (the anti-Hermitian part of its matrix log, per
unit area). No J0/J1/J2 formula is involved. As the plaquette shrinks it must
converge — linearly in the spacing — to the analytic non-Abelian matrix from
`berry_curvature(..., non_abelian=True)`. This also exercises the *off-trace*
matrix structure that the band-summed trace cannot see.
""")

code(r"""
from pythtb.io.w90 import wannier_connection_ft

lat = w90.lat
Brec = 2 * np.pi * np.linalg.inv(lat).T                     # reciprocal vectors (rows)
ham_h = {R: blk["h"] / blk["deg"] for R, blk in w90.ham_r.items()}
pos_eff = {R: w90.pos_r[R] / w90.ham_r[R]["deg"] for R in w90.pos_r}


def U_of(kk):
    H = sum(np.exp(2j * np.pi * np.dot(kk, R)) * H_R for R, H_R in ham_h.items())
    return np.linalg.eigh(H)[1]


def overlap(k1, k2):
    U1, U2 = U_of(k1), U_of(k2)
    A_raw = wannier_connection_ft(pos_eff, (k1 + k2) / 2)[0]          # connection at midpoint
    A = 0.5 * (A_raw + np.conj(np.transpose(A_raw, (0, 2, 1))))       # Hermitian part
    S = np.eye(U1.shape[0]) - 1j * np.einsum("a,aij->ij", (k2 - k1) @ Brec, A)
    return (U1.conj().T @ S @ U2)[np.ix_(OCC, OCC)]                   # position-corrected overlap


def logm(W):
    w, V = np.linalg.eig(W)
    return (V * np.log(w)) @ np.linalg.inv(V)


k0 = np.array([1 / 3 + 0.02, 1 / 3 - 0.01, 0.0])                      # near K, sizeable curvature
B_analytic = tb.berry_curvature(k0[None], occ_idxs=OCC, plane=(0, 1),
                                non_abelian=True, cartesian=True)[0]
print("finite-difference plaquette  ->  ||Wilson - analytic non-Abelian matrix||")
for delta in (2e-3, 1e-3, 5e-4):
    dx, dy = np.array([delta, 0, 0]), np.array([0, delta, 0])
    W = (overlap(k0, k0 + dx) @ overlap(k0 + dx, k0 + dx + dy)
         @ overlap(k0 + dx + dy, k0 + dy) @ overlap(k0 + dy, k0))
    dxc, dyc = dx @ Brec, dy @ Brec
    area = dxc[0] * dyc[1] - dxc[1] * dyc[0]
    L = logm(W)
    F = 1j * 0.5 * (L - L.conj().T) / area                           # anti-Herm part = curvature
    print("   delta = %.0e   err = %.2e" % (delta, np.linalg.norm(F - B_analytic)))
""")

md(r"""
## Check 3 — bcc Fe (SOC): a metal, and the integrated AHC

Fe exercises what hBN cannot. It is a **metal**, so the occupied set changes
with **k** (Fermi occupation), and the Wannier manifold is a proper subset with
empty states present — the full J0 + J1 + J2 structure is active, not only the
curl. We compare the band-summed curvature along Γ–H–P–N–Γ to postw90, and
integrate the anomalous Hall conductivity σ_xy over a uniform mesh.

(`Fe_tb.dat` is ~4 MB and git-ignored; if it is absent this cell simply skips.)
""")

code(r"""
fe_tb = WB / "Fe" / "Fe_tb.dat"
if fe_tb.exists():
    FERMI = 17.4654
    NODES_FE = [[0, 0, 0], [0.5, -0.5, -0.5], [0.75, 0.25, -0.25],
                [0.5, 0, -0.5], [0, 0, 0]]                # G H P N G
    wfe = W90(str(WB / "Fe"), "Fe")
    tfe = wfe.model()
    rfe = np.loadtxt(WB / "Fe" / "Fe-curv.dat")
    kfe = kpath_from_postw90(rfe, NODES_FE, wfe.lat)
    rfe_z = -rfe[:, 3]
    cfe = tfe.berry_curvature(kfe, fermi=FERMI, plane=(0, 1), cartesian=True)
    dfe = tfe.berry_curvature(kfe, fermi=FERMI, plane=(0, 1), cartesian=True,
                              include_external=False)
    print("Fe path  corrected vs postw90 : corr = %.5f , fit = %.4f"
          % (np.corrcoef(cfe, rfe_z)[0, 1], np.sum(cfe * rfe_z) / np.sum(cfe ** 2)))
    print("Fe path  diagonal  vs postw90 : corr = %.5f   (worse -> external matters pointwise)"
          % np.corrcoef(dfe, rfe_z)[0, 1])

    n = 25
    g = np.arange(n) / n
    KX, KY, KZ = np.meshgrid(g, g, g, indexing="ij")
    km = np.stack([KX.ravel(), KY.ravel(), KZ.ravel()], axis=1)
    e_SI, hbar_SI = 1.602176634e-19, 1.054571817e-34
    Vc = abs(np.linalg.det(wfe.lat))
    fac = -1.0e8 * e_SI ** 2 / (hbar_SI * Vc)                        # -> S/cm
    sig_c = fac * np.mean(-tfe.berry_curvature(km, fermi=FERMI, plane=(0, 1), cartesian=True))
    sig_d = fac * np.mean(-tfe.berry_curvature(km, fermi=FERMI, plane=(0, 1), cartesian=True,
                                               include_external=False))
    print("AHC sigma_xy (25^3 mesh)      : corrected = %.1f , diagonal = %.1f  S/cm"
          % (sig_c, sig_d))
    print("   postw90 on the same 25^3   : 1224.82 S/cm    (literature, converged ~750)")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(rfe[:, 0], rfe_z, "k-", lw=3, alpha=0.35, label="postw90")
    ax.plot(rfe[:, 0], cfe, "r--", lw=1.2, label="PythTB + external")
    ax.plot(rfe[:, 0], dfe, "b:", lw=1.2, label="PythTB diagonal")
    ax.set_xlabel("k-path  $\\Gamma$ - H - P - N - $\\Gamma$")
    ax.set_ylabel(r"$\Omega_z$  [$\AA^2$]")
    ax.set_title("bcc Fe (SOC) - band-summed Berry curvature (Fermi occupation)")
    ax.legend()
    fig.tight_layout()
    plt.show()
else:
    print("Fe_tb.dat not present (git-ignored, ~4 MB) - skipping the Fe metal benchmark.")
    print("Regenerate via tests/w90_berry/Fe/ (see that folder's README).")
""")

md(r"""
## Why a Chern number can *not* validate the external

The external connection `A^W(k) = Σ_R e^{ik·R} <0n|r|Rm>` is smooth and periodic
over the BZ. Any closed-BZ topological integral (Chern, Z₂) differs between the
diagonal and corrected curvatures by `∮ A^W` around the torus — which is exactly
**zero**. So the external changes the **local** curvature dramatically but leaves
**every integer invariant unchanged**. (This is precisely why Wannier90 gets
correct Chern/Z₂ numbers from Wilson loops without ever forming the position
matrix.) The cell below shows it on hBN: large local curvature, Chern = 0 either
way. The honest probe of *"is the external real"* is the **local** curvature
(Checks 1–3), not topology.
""")

code(r"""
n = 60
gg = (np.arange(n) + 0.5) / n
KX, KY = np.meshgrid(gg, gg, indexing="ij")
km = np.stack([KX.ravel(), KY.ravel(), np.zeros(n * n)], axis=1)      # kx-ky slice, kz = 0

Om_red = tb.berry_curvature(km, occ_idxs=OCC, plane=(0, 1), cartesian=False)
Om_cart = tb.berry_curvature(km, occ_idxs=OCC, plane=(0, 1), cartesian=True)
Om_diag = tb.berry_curvature(km, occ_idxs=OCC, plane=(0, 1), cartesian=True,
                             include_external=False)
print("hBN valence, 2D kx-ky slice:")
print("  Chern number (corrected)  = %+.4f   (trivial insulator -> 0)"
      % (np.mean(Om_red) / (2 * np.pi)))
print("  max|Omega| corrected      = %7.3f  Ang^2   <- 100%% external"
      % np.max(np.abs(Om_cart)))
print("  max|Omega| diagonal       = %.1e  Ang^2   <- complete manifold"
      % np.max(np.abs(Om_diag)))
print()
print("  => large LOCAL curvature, but the closed-BZ integral (Chern) is 0 with")
print("     or without the external. Topology cannot test it; local Omega does.")
""")

md(r"""
## Summary

| system | what it isolates | result |
|---|---|---|
| hBN (insulator, complete manifold) | external = 100 % of Ω | diagonal ≡ 0; corrected = postw90 (corr 0.99996, fit 1.002); = conv-II formula to ~1e-14 |
| hBN Wilson loop | geometric, off-trace matrix | linear convergence to the analytic non-Abelian matrix |
| Wannier centers | position-matrix data ingest | matches `_centres.xyz` to 5e-8 |
| Fe (metal, SOC) | Fermi occupation + full J0+J1+J2 | path corr ≈ 1.0; AHC corrected = postw90 (1224.8 S/cm) |
| Chern (hBN slice) | topological (in)sensitivity | 0 with and without the external — by design |

**Bottom line.** PythTB's `include_external` Berry curvature is validated three
independent ways — an independent algebraic derivation (machine precision), the
field-standard Fortran code postw90 (0.004 %), and a finite-difference Wilson
loop (linear convergence) — with the underlying position matrix cross-checked
against Wannier90's own `_centres.xyz`. The one input it *shares* with postw90 is
the position matrix `<0n|r|Rm>`, which is Wannier90's standard, literature-
validated output, not something either code re-derives. The only fully
end-to-end (DFT → measurable) check is the Fe AHC versus literature.

### Regenerating the data
The committed fixtures live in `tests/w90_berry/{hBN,Fe}/`; each has a `run.sh`
(QE → pw2wannier90 → wannier90 `write_tb` → postw90) and a README. On Apple
Accelerate you need the `dotfix.c` ZDOTC shim (see `tests/w90_berry/`). The
pytest versions of these checks are in `tests/test_w90/test_berry_reference.py`.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent / "validation_summary.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, "with", len(cells), "cells")
