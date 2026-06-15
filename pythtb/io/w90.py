"""IO utilities for Wannier90 output files."""

from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np

from pythtb.constants import BOHRTOANG

__all__ = [
    "HRBlock",
    "W90Dataset",
    "read_win",
    "parse_unit_cell_cart",
    "read_centres",
    "read_hr",
    "read_tb",
    "read_r",
    "read_kpoint_path",
    "load_w90_dataset",
    "read_bands_w90",
    "wannier_connection_ft",
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
    r"""Dataclass for Wannier90 data.

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
    pos_r : dict[tuple[int, int, int], numpy.ndarray] | None
        Mapping from lattice vectors ``R`` to the off-diagonal position
        matrix :math:`\langle 0n | r_\alpha | Rm \rangle`, stored as a complex
        array of shape ``(3, num_wan, num_wan)`` in Cartesian coordinates
        (angstroms). Populated only when ``prefix_tb.dat`` (``write_tb``) or
        ``prefix_r.dat`` is available; otherwise ``None``.

        .. versionadded:: 2.1.0

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
    win_lines : list[str] | None
        Raw lines from ``prefix.win`` when requested by the loader.
    """

    prefix: str
    root: Path
    lat_cart: np.ndarray  # (3,3) Angstrom
    centres_xyz: np.ndarray  # (num_wan,3) Angstrom
    centres_red: np.ndarray  # (num_wan,3) reduced
    num_wan: int
    ham_r: Dict[Tuple[int, int, int], HRBlock]
    # optional extras
    pos_r: Optional[Dict[Tuple[int, int, int], np.ndarray]] = None
    kpath_nodes_red: Optional[np.ndarray] = None
    kpath_labels: Optional[List[str]] = None
    bands_k_red: Optional[np.ndarray] = None
    bands_ene_ev: Optional[np.ndarray] = None
    meta: Optional[dict] = None  # spreads, windows, etc.
    win_lines: Optional[List[str]] = None


# ---------- low-level readers ----------


def _read_text(path: Path) -> List[str]:
    """Read a text file into a list of lines (raises if missing)."""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return f.readlines()
    except FileNotFoundError as e:
        raise W90ParseError(f"Missing file: {path}") from e


def _extract_block(lines: List[str], name: str) -> List[str]:
    """Return the lines inside a ``begin <name> ... end <name>`` block of a .win file."""
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
    """Convert Cartesian coordinates to reduced coordinates of the given lattice."""
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
    unique_R = R[first_idx[order]]
    if deg.size < unique_R.shape[0]:
        raise W90ConsistencyError("Degeneracy list shorter than number of shells.")
    blocks = np.zeros((unique_R.shape[0], num_wan, num_wan), complex)
    np.add.at(blocks, (inv, i, j), v)
    ham_r = {
        tuple(map(int, unique_R[k])): HRBlock(h=blocks[k], degeneracy=int(deg[k]))
        for k in range(unique_R.shape[0])
    }
    return num_wan, ham_r


_INT_RE = re.compile(r"^[+-]?\d+$")


def _all_ints(tokens: List[str]) -> bool:
    """True if every whitespace token parses as an integer."""
    return all(_INT_RE.match(t) for t in tokens)


def read_tb(
    root: Path | str, prefix: str
) -> Tuple[
    int,
    Dict[Tuple[int, int, int], HRBlock],
    Dict[Tuple[int, int, int], np.ndarray],
    np.ndarray,
]:
    r"""Read ``prefix_tb.dat`` produced by Wannier90 ``write_tb = .true.``.

    Unlike ``prefix_hr.dat`` (which only stores the Hamiltonian), the
    ``prefix_tb.dat`` file additionally contains the **off-diagonal position
    matrix elements** :math:`\langle 0n | r_\alpha | Rm \rangle`. These encode
    the true (non-diagonal) position operator of the maximally localized
    Wannier functions and are required to compute Berry-phase-like quantities
    that match the underlying first-principles result.

    .. versionadded:: 2.1.0

    Parameters
    ----------
    root : Path or str
        Directory containing ``prefix_tb.dat``.
    prefix : str
        Prefix used by Wannier90: the file read is ``{prefix}_tb.dat``.

    Returns
    -------
    num_wan : int
        Number of Wannier functions.
    ham_r : dict
        Mapping from lattice vector triplet ``R`` to :class:`HRBlock` (eV).
    pos_r : dict
        Mapping from lattice vector triplet ``R`` to a complex array of shape
        ``(3, num_wan, num_wan)`` holding
        :math:`\langle 0n | r_\alpha | Rm \rangle` for the three Cartesian
        directions :math:`\alpha \in \{x, y, z\}` (angstroms).
    lat_cart : numpy.ndarray
        Cartesian lattice vectors ``(3, 3)`` in angstroms (row ``i`` is
        :math:`\mathbf{a}_i`).

    Notes
    -----
    The Wannier90 file layout is: a comment line, three lattice-vector rows,
    ``num_wann``, ``nrpts``, the Wigner-Seitz degeneracy list, then ``nrpts``
    Hamiltonian blocks (rows ``i j Re(H) Im(H)``) followed by ``nrpts``
    position blocks (rows ``i j Re(x) Im(x) Re(y) Im(y) Re(z) Im(z)``). Each
    block is preceded by its ``R`` vector on its own line.
    """
    p = Path(root).expanduser() / f"{prefix}_tb.dat"
    if not p.exists():
        raise W90ParseError(f"Missing file: {p}")
    lines = _read_text(p)
    if len(lines) < 6:
        raise W90ParseError(f"{p} is too short to be a valid _tb.dat file.")

    # Line 0 is a free-form comment/date; lines 1-3 are the lattice vectors.
    lat = np.zeros((3, 3), float)
    for i in range(3):
        parts = lines[1 + i].split()
        if len(parts) < 3:
            raise W90ParseError("_tb.dat lattice rows need 3 components.")
        lat[i] = [float(parts[j]) for j in range(3)]

    try:
        num_wan = int(lines[4].split()[0])
        num_ws = int(lines[5].split()[0])
    except Exception as e:
        raise W90ParseError("Cannot read num_wann/nrpts in _tb.dat") from e

    # Wigner-Seitz degeneracies (may span multiple lines, 15 per line).
    idx = 6
    deg: List[int] = []
    while len(deg) < num_ws:
        if idx >= len(lines):
            raise W90ParseError("Unexpected EOF while reading degeneracies in _tb.dat.")
        deg.extend(int(x) for x in lines[idx].split())
        idx += 1
    deg = np.asarray(deg[:num_ws], int)

    # Group the remaining lines into blocks. A line with exactly three integer
    # tokens starts a new R block; every data row has >= 4 tokens.
    blocks: List[Tuple[Tuple[int, int, int], List[List[float]]]] = []
    cur: Optional[Tuple[Tuple[int, int, int], List[List[float]]]] = None
    for raw in lines[idx:]:
        toks = raw.split()
        if not toks:
            continue
        if len(toks) == 3 and _all_ints(toks):
            cur = ((int(toks[0]), int(toks[1]), int(toks[2])), [])
            blocks.append(cur)
        else:
            if cur is None:
                raise W90ParseError(
                    "Data row encountered before any R header in _tb.dat."
                )
            cur[1].append([float(t) for t in toks])

    if len(blocks) != 2 * num_ws:
        raise W90ConsistencyError(
            f"_tb.dat: expected {2 * num_ws} R-blocks "
            f"({num_ws} Hamiltonian + {num_ws} position), found {len(blocks)}."
        )

    nw2 = num_wan * num_wan

    def _check_rows(R, rows, kind):
        """Validate that an R-block has exactly num_wan**2 rows."""
        if len(rows) != nw2:
            raise W90ConsistencyError(
                f"_tb.dat {kind} block for R={R} has {len(rows)} rows; expected {nw2}."
            )

    # First num_ws blocks: Hamiltonian H(R).
    ham_r: Dict[Tuple[int, int, int], HRBlock] = {}
    for k in range(num_ws):
        R, rows = blocks[k]
        _check_rows(R, rows, "Hamiltonian")
        H = np.zeros((num_wan, num_wan), complex)
        for row in rows:
            i = int(row[0]) - 1
            j = int(row[1]) - 1
            H[i, j] = row[2] + 1j * row[3]
        ham_r[R] = HRBlock(h=H, degeneracy=int(deg[k]))

    # Next num_ws blocks: position matrix r(R).
    pos_r: Dict[Tuple[int, int, int], np.ndarray] = {}
    for k in range(num_ws):
        R, rows = blocks[num_ws + k]
        _check_rows(R, rows, "position")
        X = np.zeros((3, num_wan, num_wan), complex)
        for row in rows:
            if len(row) < 8:
                raise W90ConsistencyError(
                    f"_tb.dat position row for R={R} needs 8 numbers, got {len(row)}."
                )
            i = int(row[0]) - 1
            j = int(row[1]) - 1
            X[0, i, j] = row[2] + 1j * row[3]
            X[1, i, j] = row[4] + 1j * row[5]
            X[2, i, j] = row[6] + 1j * row[7]
        pos_r[R] = X

    return num_wan, ham_r, pos_r, lat


def read_r(
    root: Path | str, prefix: str, num_wan: int
) -> Dict[Tuple[int, int, int], np.ndarray]:
    r"""Read the legacy ``prefix_r.dat`` position-matrix file.

    This is the position-only counterpart to :func:`read_hr`, written by older
    Wannier90 runs (e.g. with ``transl_inv`` / ``write_rmn``). Prefer
    :func:`read_tb` when ``prefix_tb.dat`` is available.

    .. versionadded:: 2.1.0

    Parameters
    ----------
    root : Path or str
        Directory containing ``prefix_r.dat``.
    prefix : str
        Prefix used by Wannier90: the file read is ``{prefix}_r.dat``.
    num_wan : int
        Number of Wannier functions (used to validate the table).

    Returns
    -------
    pos_r : dict
        Mapping from lattice vector triplet ``R`` to a complex array of shape
        ``(3, num_wan, num_wan)`` holding
        :math:`\langle 0n | r_\alpha | Rm \rangle` (angstroms).
    """
    p = Path(root).expanduser() / f"{prefix}_r.dat"
    if not p.exists():
        raise W90ParseError(f"Missing file: {p}")
    with p.open("r", encoding="utf-8", errors="ignore") as fh:
        _ = fh.readline()  # comment/date
        try:
            file_num_wan = int(fh.readline())
            _num_ws = int(fh.readline())
        except Exception as e:
            raise W90ParseError("Cannot read num_wann/nrpts in _r.dat") from e
        data = np.loadtxt(fh)
    if file_num_wan != num_wan:
        raise W90ConsistencyError(
            f"_r.dat reports num_wann={file_num_wan}, expected {num_wan}."
        )
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != 11:
        raise W90ParseError("_r.dat must have 11 columns.")
    R = data[:, :3].astype(int)
    i = data[:, 3].astype(int) - 1
    j = data[:, 4].astype(int) - 1
    rx = data[:, 5] + 1j * data[:, 6]
    ry = data[:, 7] + 1j * data[:, 8]
    rz = data[:, 9] + 1j * data[:, 10]
    pos_r: Dict[Tuple[int, int, int], np.ndarray] = {}
    for row in range(data.shape[0]):
        key = (int(R[row, 0]), int(R[row, 1]), int(R[row, 2]))
        if key not in pos_r:
            pos_r[key] = np.zeros((3, num_wan, num_wan), complex)
        pos_r[key][0, i[row], j[row]] = rx[row]
        pos_r[key][1, i[row], j[row]] = ry[row]
        pos_r[key][2, i[row], j[row]] = rz[row]
    return pos_r


_KPOINT_LABEL_PATTERN = re.compile(r"^(?P<base>[^\d]+?)(?P<suffix>\d+)?$", re.UNICODE)


def _format_k_label(label: str) -> str:
    """Map a Wannier90 k-point label to its LaTeX form (e.g. 'G' -> r'\\Gamma')."""
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
# Bump when the cache layout changes so stale files are ignored, not misread.
_CACHE_VERSION = 1


def _hr_pos_sources(root: Path, prefix: str) -> list:
    """(name, mtime_ns, size) for each Hamiltonian/position source present."""
    sig = []
    for name in (f"{prefix}_tb.dat", f"{prefix}_hr.dat", f"{prefix}_r.dat"):
        p = root / name
        if p.exists():
            st = p.stat()
            sig.append((name, st.st_mtime_ns, st.st_size))
    return sig


def _read_hr_pos(root: Path, prefix: str):
    """Parse ``H(R)`` and, when available, the position matrix from disk.

    Prefers ``prefix_tb.dat`` (``write_tb``): it carries both ``H(R)`` and the
    position matrix ``<0n|r|Rm>``. Falls back to ``prefix_hr.dat``, optionally
    augmented by a legacy ``prefix_r.dat`` for the position matrix.
    """
    if (root / f"{prefix}_tb.dat").exists():
        num_wan, ham_r, pos_r, _lat_tb = read_tb(root, prefix)
        return num_wan, ham_r, pos_r
    num_wan, ham_r = read_hr(root, prefix)
    pos_r: Optional[Dict[Tuple[int, int, int], np.ndarray]] = None
    if (root / f"{prefix}_r.dat").exists():
        try:
            pos_r = read_r(root, prefix, num_wan)
        except (W90ParseError, W90ConsistencyError):
            pos_r = None
    return num_wan, ham_r, pos_r


def _load_hr_pos_cached(root: Path, prefix: str, *, cache: bool = True):
    """Cached wrapper around :func:`_read_hr_pos`.

    The parsed arrays are stored in ``{prefix}_pythtb_cache.npz`` next to the
    Wannier90 files, keyed on the source files' modification times and sizes.
    Reading and writing the cache is best-effort: any failure (unreadable or
    stale file, read-only directory) falls back to a fresh parse.
    """
    if not cache:
        return _read_hr_pos(root, prefix)

    cache_path = root / f"{prefix}_pythtb_cache.npz"
    signature = repr((_CACHE_VERSION, _hr_pos_sources(root, prefix)))

    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                if str(data["signature"]) == signature:
                    R_list = data["R_list"]
                    deg = data["deg"]
                    H = data["H"]
                    ham_r = {
                        tuple(map(int, R_list[k])): HRBlock(
                            h=H[k], degeneracy=int(deg[k])
                        )
                        for k in range(R_list.shape[0])
                    }
                    pos_r = None
                    if bool(data["has_pos"]):
                        pos_R = data["pos_R_list"]
                        pos = data["pos"]
                        pos_r = {
                            tuple(map(int, pos_R[k])): pos[k]
                            for k in range(pos_R.shape[0])
                        }
                    return int(data["num_wan"]), ham_r, pos_r
        except Exception:
            pass  # unreadable or stale cache: re-parse below

    num_wan, ham_r, pos_r = _read_hr_pos(root, prefix)

    try:
        Rs = list(ham_r)
        payload = {
            "signature": np.array(signature),
            "num_wan": np.int64(num_wan),
            "R_list": np.array(Rs, dtype=np.int64),
            "deg": np.array([ham_r[R].degeneracy for R in Rs], dtype=np.int64),
            "H": np.stack([ham_r[R].h for R in Rs]),
            "has_pos": np.bool_(pos_r is not None),
        }
        if pos_r is not None:
            pos_Rs = list(pos_r)
            payload["pos_R_list"] = np.array(pos_Rs, dtype=np.int64)
            payload["pos"] = np.stack([pos_r[R] for R in pos_Rs])
        np.savez(cache_path, **payload)
    except Exception:
        pass  # cache is an optimization only; never fail the load over it

    return num_wan, ham_r, pos_r


def load_w90_dataset(
    root: Path | str,
    prefix: str,
    *,
    include_bands: bool = True,
    include_win_lines: bool = False,
    cache: bool = True,
) -> W90Dataset:
    """Gather lattice, centre, and Hamiltonian data into a :class:`W90Dataset`.

    .. versionchanged:: 2.1.0
        Also loads the Wannier position matrix into :attr:`W90Dataset.pos_r`
        when ``prefix_tb.dat`` or ``prefix_r.dat`` is available, and caches the
        parsed Hamiltonian/position arrays (see ``cache``).

    Parameters
    ----------
    root : Path or str
        Directory containing the Wannier90 files.
    prefix : str
        Prefix used in the Wannier90 file names.
    cache : bool, optional
        If True (default), store the parsed ``H(R)``/position arrays in
        ``{prefix}_pythtb_cache.npz`` next to the Wannier90 files and reuse
        them while the source files are unchanged (keyed on their
        modification time and size). The text files are large (one row per
        matrix element), so this turns minutes of re-parsing into a
        sub-second load. The cache is best-effort: if the directory is not
        writable the parse result is simply not cached.

        .. versionadded:: 2.1.0

    Returns
    -------
    dataset : W90Dataset
        Container with all relevant data from the Wannier90 output files.
    """
    root = Path(root).expanduser()
    win = read_win(root, prefix)
    lat = parse_unit_cell_cart(win)

    num_wan, ham_r, pos_r = _load_hr_pos_cached(root, prefix, cache=cache)
    centres_xyz = read_centres(root, prefix, num_wan)
    centres_red = _cart_to_red(lat[0], lat[1], lat[2], centres_xyz)
    k_nodes, k_labels = read_kpoint_path(win, latex=True)
    # bands are optional
    bands_k, bands_ene = None, None
    if include_bands:
        try:
            bands_k, bands_ene = read_bands_w90(root, prefix, num_wan)
        except Exception:
            pass
    win_lines = win if include_win_lines else None
    return W90Dataset(
        prefix=prefix,
        root=root,
        lat_cart=lat,
        centres_xyz=centres_xyz,
        centres_red=centres_red,
        num_wan=num_wan,
        ham_r=ham_r,
        pos_r=pos_r,
        kpath_nodes_red=k_nodes,
        kpath_labels=k_labels,
        bands_k_red=bands_k,
        bands_ene_ev=bands_ene,
        meta={},
        win_lines=win_lines,
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


def wannier_connection_ft(
    pos_r_eff: Dict[Tuple[int, int, int], np.ndarray],
    k_red: np.ndarray,
    *,
    lat: Optional[np.ndarray] = None,
    cartesian: bool = True,
) -> np.ndarray:
    r"""Bloch sum of the Wannier position matrix (Wannier-gauge connection).

    Computes

    .. math::

        \mathcal{A}^{(\mathrm{W})}_{\alpha}(\mathbf{k})_{nm}
        = \sum_{\mathbf{R}} e^{i 2\pi \mathbf{k}\cdot\mathbf{R}}
          \langle 0n | r_\alpha | \mathbf{R}m \rangle ,

    i.e. the Fourier transform of the off-diagonal position matrix onto a set of
    reduced k-points. In the theory of Wannier interpolation [Wang2006]_ this is
    the Berry connection in the **Wannier gauge**; rotating it by the
    eigenvectors of :math:`H(\mathbf{k})` (and adding the usual gauge-covariant
    term) yields the smooth Berry connection used for Berry curvature and the
    anomalous Hall conductivity.

    .. versionadded:: 2.1.0

    Parameters
    ----------
    pos_r_eff : dict
        Mapping ``R -> (3, num_wan, num_wan)`` of position matrix elements,
        already divided by the Wigner-Seitz degeneracy (so the Bloch sum is a
        plain sum over ``R``).
    k_red : numpy.ndarray
        Reduced k-points of shape ``(Nk, 3)``.
    lat : numpy.ndarray, optional
        Cartesian lattice vectors ``(3, 3)``, required only when
        ``cartesian=False`` to project the Cartesian operator index onto the
        reduced lattice directions.
    cartesian : bool, optional
        If True (default) the leading operator index :math:`\alpha` is Cartesian
        (x, y, z). If False, it is expressed in reduced lattice components.

    Returns
    -------
    A : numpy.ndarray
        Array of shape ``(Nk, 3, num_wan, num_wan)``. For a Hermitian input
        (``r(-R) = r(R)^\dagger``) each ``A[k, alpha]`` is Hermitian.

    References
    ----------
    .. [Wang2006] X. Wang, J. R. Yates, I. Souza, D. Vanderbilt,
       "Ab initio calculation of the anomalous Hall conductivity by Wannier
       interpolation", Phys. Rev. B 74, 195118 (2006).
    """
    if not pos_r_eff:
        raise ValueError("pos_r_eff is empty; no position matrix to transform.")
    items = list(pos_r_eff.items())
    Rs = np.array([R for R, _ in items], dtype=float)  # (nR, 3)
    X = np.stack([V for _, V in items], axis=0)  # (nR, 3, nw, nw)

    k_red = np.atleast_2d(np.asarray(k_red, dtype=float))  # (Nk, 3)
    phases = np.exp(2j * np.pi * (k_red @ Rs.T))  # (Nk, nR)
    A = np.einsum("kr, ranm -> kanm", phases, X, optimize=True)  # (Nk, 3, nw, nw)

    if not cartesian:
        if lat is None:
            raise ValueError("lat must be provided when cartesian=False.")
        # r_red_i = sum_alpha r_cart_alpha * inv(lat)[alpha, i]
        inv_lat = np.linalg.inv(np.asarray(lat, dtype=float))
        A = np.einsum("kanm, ai -> kinm", A, inv_lat, optimize=True)

    return A
