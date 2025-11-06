"""IO utilities for Quantum ESPRESSO output files."""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np

BOHRTOANG = 0.52917721092


class QEParseError(RuntimeError):
    """Raised when a Quantum ESPRESSO bands file cannot be parsed."""


class QEConsistencyError(RuntimeError):
    """Raised when Quantum ESPRESSO band data fail internal sanity checks."""


_QE_HDR_RE = re.compile(r"nbnd\s*=\s*(\d+).+nks\s*=\s*(\d+)", re.I | re.S)

__all__ = [
    "BOHRTOANG",
    "QEParseError",
    "QEConsistencyError",
    "read_bands_qe",
]


def _qe_is_k_marker(s: str) -> bool:
    # line with exactly three floats → k marker
    try:
        vals = [float(x) for x in s.split()]
        return len(vals) == 3
    except ValueError:
        return False


def read_bands_qe(
    root: Path | str, prefix: str
) -> Tuple[np.ndarray, List[List[float]], Dict[str, int]]:
    """Read QE bands file lines, returning the raw (unscaled) k-markers and energies.

    Parameters
    ----------
    root : Path or str
        Directory containing the QE bands file.
    prefix : str
        Prefix used in the QE bands file name: "{prefix}_bands.dat".

    Returns
    -------
    k_markers : (N_k, 3)
        Cartesian k-vectors exactly as printed by `bands.x`, i.e. in
        units of ``2π/alat``. Callers may rescale them after the fact
        if they want k in reciprocal-lattice or Cartesian Å⁻¹ units.
    energies_rows : list[list[float]]
        energies per k-point
    meta : dict with possible keys 'nbnd', 'nks'

    Notes
    -----
    - Quantum ESPRESSO's `bands.x` module writes k-points in units of
      ``2pi/alat``, where `alat` is the lattice parameter specified in
      the QE input file. This function does not perform any rescaling
      of the k-points; it is the caller's responsibility to do so if
      needed. The :class:`pythtb.W90` class provides utilities for
      rescaling k-points to reciprocal-lattice or Cartesian units.
    """
    root = Path(root).expanduser()
    bands_path = root / f"{prefix}_bands.dat"
    if not bands_path.exists():
        raise QEParseError(f"Missing QE bands file: {bands_path}")

    with bands_path.open("r", encoding="utf-8", errors="ignore") as f:
        head = f.read(5000)
        f.seek(0)
        meta: Dict[str, int] = {}
        m = _QE_HDR_RE.search(head)
        if m:
            meta["nbnd"] = int(m.group(1))
            meta["nks"] = int(m.group(2))

        klist, energies_rows, ebuf = [], [], []
        for line in f:
            s = line.strip()
            if not s:
                continue
            if _qe_is_k_marker(s):
                # flush previous energies (if any) and start a new k-point
                if ebuf:
                    energies_rows.append(ebuf)
                    ebuf = []
                kx, ky, kz = (float(x) for x in s.split())
                klist.append([kx, ky, kz])
            else:
                try:
                    vals = [float(x) for x in s.split()]
                except ValueError:
                    continue
                ebuf.extend(vals)
        if ebuf:
            energies_rows.append(ebuf)

    return np.asarray(klist, float), energies_rows, meta
