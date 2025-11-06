"""IO utilities for Wannier90 output files."""

from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np

BOHRTOANG = 0.52917721092

__all__ = [
    "HRBlock",
    "W90Dataset",
    "read_win",
    "parse_unit_cell_cart",
    "read_centres",
    "read_hr",
    "read_kpoint_path",
    "load_w90_dataset",
    "read_bands_w90",
]


class W90ParseError(RuntimeError):
    """Raised when a Wannier90 file is missing or cannot be parsed."""


class W90ConsistencyError(RuntimeError):
    """Raised when Wannier90 data are internally inconsistent."""


@dataclass(frozen=True)
class HRBlock:
    """
    Dataclass representing real-space Hamiltonian block :math:`H(R)` with degeneracy.

    Attributes
    ----------
    h : numpy.ndarray
        Complex matrix ``(num_wan, num_wan)`` containing the tight-binding
        amplitudes for lattice vector ``R``.
    degeneracy : int
        Wigner-Seitz multiplicity associated with the shell of ``R``.
    """

    h: np.ndarray
    degeneracy: int


@dataclass(frozen=True)
class W90Dataset:
    """Dataclass for Wannier90 data.

    Attributes
    ----------
    prefix : str
        Wannier90 run prefix.
    root : pathlib.Path
        Directory containing the output files.
    lat_cart : numpy.ndarray
        Cartesian lattice vectors with shape ``(3, 3)`` in angstroms.
    centres_xyz : numpy.ndarray
        Wannier centres in Cartesian coordinates ``(num_wan, 3)``.
    centres_red : numpy.ndarray
        Wannier centres in reduced coordinates ``(num_wan, 3)``.
    num_wan : int
        Number of Wannier functions in the dataset.
    ham_r : dict[tuple[int, int, int], HRBlock]
        Mapping from lattice vectors ``R`` to their Hamiltonian blocks.
    kpath_nodes_red : numpy.ndarray | None
        Reduced coordinates of the ``kpoint_path`` nodes, if present.
    kpath_labels : list[str] | None
        Labels corresponding to ``kpath_nodes_red``.
    bands_k_red : numpy.ndarray | None
        Reduced k-points from Wannier90 band interpolation.
    bands_ene_ev : numpy.ndarray | None
        Interpolated band energies (eV) matching ``bands_k_red``.
    meta : dict | None
        Additional metadata such as spreads or window definitions.
    """

    prefix: str
    root: Path
    lat_cart: np.ndarray  # (3,3) Angstrom
    centres_xyz: np.ndarray  # (num_wan,3) Angstrom
    centres_red: np.ndarray  # (num_wan,3) reduced
    num_wan: int
    ham_r: Dict[Tuple[int, int, int], HRBlock]
    # optional extras
    kpath_nodes_red: Optional[np.ndarray] = None
    kpath_labels: Optional[List[str]] = None
    bands_k_red: Optional[np.ndarray] = None
    bands_ene_ev: Optional[np.ndarray] = None
    meta: Optional[dict] = None  # spreads, windows, etc.


# ---------- low-level readers ----------


def _read_text(path: Path) -> List[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return f.readlines()
    except FileNotFoundError as e:
        raise W90ParseError(f"Missing file: {path}") from e


def _extract_block(lines: List[str], name: str) -> List[str]:
    begin, end = f"begin {name}".lower(), f"end {name}".lower()
    in_block, out = False, []
    for raw in lines:
        s = raw.strip()
        t = s.lower()
        if not in_block and t.startswith(begin):
            in_block = True
            continue
        if in_block:
            if t.startswith(end):
                break
            out.append(s.replace(",", " "))  # tolerate commas
    return out


def read_win(root: Path, prefix: str) -> List[str]:
    """Return the raw lines from ``prefix.win``."""
    return _read_text((root / f"{prefix}.win").expanduser())


def parse_unit_cell_cart(win_lines: List[str]) -> np.ndarray:
    """Parse ``unit_cell_cart`` into a Cartesian (3×3) lattice matrix."""
    block = _extract_block(win_lines, "unit_cell_cart")
    if not block or len(block) < 3:
        raise W90ParseError("unit_cell_cart block missing or too short.")
    scale = 1.0
    if block[0].lower() in {"bohr", "ang", "angstrom"}:
        if block[0].lower() == "bohr":
            scale = BOHRTOANG
        block = block[1:]
    lat = np.zeros((3, 3), float)
    for i in range(3):
        parts = block[i].split()
        if len(parts) < 3:
            raise W90ParseError("unit_cell_cart rows need 3 components.")
        lat[i] = [float(parts[j]) * scale for j in range(3)]
    return lat


def read_centres(root: Path, prefix: str, num_wan: int) -> np.ndarray:
    """Read ``prefix_centres.xyz`` and return Wannier centres in Cartesian coords."""
    lines = _read_text(root / f"{prefix}_centres.xyz")
    start = 2
    coords = []
    for idx in range(num_wan):
        try:
            tag, x, y, z, *_ = lines[start + idx].split()
        except Exception as e:
            raise W90ParseError("Centres file shorter than expected.") from e
        if tag != "X":
            raise W90ParseError("Centres file format error (expected 'X').")
        coords.append([float(x), float(y), float(z)])
    return np.asarray(coords, float)


def _cart_to_red(a1, a2, a3, xyz):
    # Here a1..a3 are direct lattice vectors; reduced = xyz @ inv(lat)
    Lat = np.vstack([a1, a2, a3])
    return np.asarray(xyz) @ np.linalg.inv(Lat)


def read_hr(root: Path, prefix: str) -> Tuple[int, Dict[Tuple[int, int, int], HRBlock]]:
    """Read ``prefix_hr.dat`` returning ``(num_wan, {R: HRBlock})``.

    Parameters
    ----------
    root : Path or str
        Directory containing the Wannier90 hr file.
    prefix : str
        Prefix used in the Wannier90 hr file name: "{prefix}_hr.dat".

    Returns
    -------
    num_wan : int
        Number of Wannier functions.
    ham_r : dict
        Mapping from lattice vector triplet R (tuple of ints) to :class:`HRBlock`.
    """
    p = root / f"{prefix}_hr.dat"
    with p.open("r", encoding="utf-8", errors="ignore") as fh:
        _ = fh.readline()
        try:
            num_wan = int(fh.readline())
            num_ws = int(fh.readline())
        except Exception as e:
            raise W90ParseError("Cannot read num_wan/num_ws in _hr.dat") from e
        # degeneracies (can span multiple lines)
        deg = []
        while len(deg) < num_ws:
            line = fh.readline()
            if not line:
                raise W90ParseError("Unexpected EOF while reading degeneracies.")
            deg.extend(int(x) for x in line.split())
        deg = np.asarray(deg[:num_ws], int)
        # remainder numeric table
        data = np.loadtxt(fh)  # shape (N,7)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != 7:
        raise W90ParseError("_hr.dat must have 7 columns.")
    R = data[:, :3].astype(int)  # Triplets (R1, R2, R3)
    i = data[:, 3].astype(int) - 1  # Wannier function index i
    j = data[:, 4].astype(int) - 1  # Wannier function index j
    v = data[:, 5] + 1j * data[:, 6]  # Hamiltonian matrix element H_{ij}(R)
    # unique shells in encounter order
    _, first_idx, inv = np.unique(R, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first_idx)
    remap = np.empty_like(order)
    remap[order] = np.arange(order.size)
    inv = remap[inv]
    unique_R = np.unique(R, axis=0)[order]
    if deg.size < unique_R.shape[0]:
        raise W90ConsistencyError("Degeneracy list shorter than number of shells.")
    blocks = np.zeros((unique_R.shape[0], num_wan, num_wan), complex)
    np.add.at(blocks, (inv, i, j), v)
    ham_r = {
        tuple(map(int, unique_R[k])): HRBlock(h=blocks[k], degeneracy=int(deg[k]))
        for k in range(unique_R.shape[0])
    }
    return num_wan, ham_r


_KPOINT_LABEL_PATTERN = re.compile(r"^(?P<base>[^\d]+?)(?P<suffix>\d+)?$", re.UNICODE)


def _format_k_label(label: str) -> str:
    special = {
        "g": r"\Gamma",
        "gamma": r"\Gamma",
        "Γ": r"\Gamma",
        "delta": r"\Delta",
        "Δ": r"\Delta",
        "theta": r"\Theta",
        "Θ": r"\Theta",
        "lambda": r"\Lambda",
        "λ": r"\Lambda",
        "xi": r"\Xi",
        "ξ": r"\Xi",
        "pi": r"\Pi",
        "π": r"\Pi",
        "sigma": r"\Sigma",
        "σ": r"\Sigma",
        "upsilon": r"\Upsilon",
        "υ": r"\Upsilon",
        "phi": r"\Phi",
        "ϕ": r"\Phi",
        "psi": r"\Psi",
        "ψ": r"\Psi",
        "omega": r"\Omega",
        "ω": r"\Omega",
    }
    raw = label.strip()
    if not raw:
        return "$$"
    m = _KPOINT_LABEL_PATTERN.match(raw)
    base, suf = (m.group("base"), m.group("suffix")) if m else (raw, None)
    key = base.lower()
    latex = special.get(key) or (
        base if (len(base) == 1 and base.isalpha()) else rf"\mathrm{{{base}}}"
    )
    return rf"${latex}_{{{suf}}}$" if suf else rf"${latex}$"


def read_kpoint_path(win_lines: List[str], *, latex=True):
    """
    Return the reduced-coordinate nodes declared in the ``kpoint_path`` block.

    Parameters
    ----------
    latex : bool, optional
        When True (default) convert labels into LaTeX-friendly strings,
        e.g. ``"G" -> r"$\\Gamma$"``.

    Returns
    -------
    coords : numpy.ndarray
        Array with shape ``(n_nodes, 3)`` containing the reduced coordinates.
    labels : list[str]
        Labels for each node, optionally formatted for LaTeX rendering.
    """
    block = _extract_block(win_lines, "kpoint_path")
    if not block:
        return None, None
    nodes, labels = [], []
    last = None
    for line in block:
        toks = line.split()
        if not toks:
            continue
        if len(toks) % 4:
            raise W90ParseError("kpoint_path entries must be label + 3 coords.")
        for o in range(0, len(toks), 4):
            lbl = toks[o]
            coord = np.array(
                [float(toks[o + 1]), float(toks[o + 2]), float(toks[o + 3])]
            )
            if last is not None and np.allclose(coord, last[1]) and lbl == last[0]:
                continue
            nodes.append(coord)
            labels.append(_format_k_label(lbl) if latex else lbl)
            last = (lbl, coord)
    return np.vstack(nodes), labels


# convenience: assemble dataset
def load_w90_dataset(root: Path | str, prefix: str) -> W90Dataset:
    """Gather lattice, centre, and Hamiltonian data into a :class:`W90Dataset`.

    Parameters
    ----------
    root : Path or str
        Directory containing the Wannier90 files.
    prefix : str
        Prefix used in the Wannier90 file names.

    Returns
    -------
    dataset : W90Dataset
        Container with all relevant data from the Wannier90 output files.
    """
    root = Path(root).expanduser()
    win = read_win(root, prefix)
    lat = parse_unit_cell_cart(win)
    num_wan, ham_r = read_hr(root, prefix)
    centres_xyz = read_centres(root, prefix, num_wan)
    centres_red = _cart_to_red(lat[0], lat[1], lat[2], centres_xyz)
    k_nodes, k_labels = read_kpoint_path(win, latex=True)
    # bands are optional
    bands_k, bands_ene = None, None
    try:
        bands_k, bands_ene = read_bands_w90(root, prefix, num_wan)
    except Exception:
        pass
    return W90Dataset(
        prefix=prefix,
        root=root,
        lat_cart=lat,
        centres_xyz=centres_xyz,
        centres_red=centres_red,
        num_wan=num_wan,
        ham_r=ham_r,
        kpath_nodes_red=k_nodes,
        kpath_labels=k_labels,
        bands_k_red=bands_k,
        bands_ene_ev=bands_ene,
        meta={},
    )


def read_bands_w90(
    root: Path | str, prefix: str, num_wan: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read Wannier90-interpolated band structure.

    Parameters
    ----------
    root : Path or str
        Directory containing the Wannier90 bands files.
    prefix : str
        Prefix used in the Wannier90 bands file names:
        "{prefix}_band.kpt" and "{prefix}_band.dat".
    num_wan : int
        Number of Wannier functions / bands expected.

    Returns
    -------
    kpts_red : (N_k, 3) reduced k-points
    energies_ev : (N_k, num_wan) energies in eV
    """
    root = Path(root).expanduser()
    kpts_path = root / f"{prefix}_band.kpt"
    ene_path = root / f"{prefix}_band.dat"

    if not kpts_path.exists() or not ene_path.exists():
        raise W90ParseError(f"Missing W90 bands files: {kpts_path} or {ene_path}")

    kpts_red = np.loadtxt(kpts_path, skiprows=1)[:, :3]
    ene_raw = np.loadtxt(ene_path)
    if ene_raw.ndim == 1:
        ene_raw = ene_raw[None, :]
    # column 0 is k-index, column 1 is energy; reshape like W90 writes it
    try:
        energies_ev = ene_raw[:, 1].reshape((num_wan, kpts_red.shape[0])).T
    except ValueError as e:
        raise W90ParseError(
            f"Cannot reshape bands: expected {num_wan} bands; "
            f"got {ene_raw.shape} rows for {kpts_red.shape[0]} k-points"
        ) from e
    return kpts_red, energies_ev
