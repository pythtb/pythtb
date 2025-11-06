# pythtb/io/wannier90.py
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np

BOHRTOANG = 0.52917721092

class QEParseError(RuntimeError): ...
class QEConsistencyError(RuntimeError): ...

_QE_HDR_RE = re.compile(r"nbnd\s*=\s*(\d+).+nks\s*=\s*(\d+)", re.I | re.S)

def _qe_is_k_marker(s: str) -> bool:
    # line with exactly three floats → k marker
    try:
        vals = [float(x) for x in s.split()]
        return len(vals) == 3
    except ValueError:
        return False

def read_bands_qe(root: Path | str, prefix: str) -> Tuple[np.ndarray, List[List[float]], Dict[str, int]]:
    """
    Read raw QE bands file lines, returning unscaled k-markers and ragged energy rows.

    Returns
    -------
    k_markers : (N_k, 3) floats as written by QE (units handled by caller)
    energies_rows : list[list[float]] energies per k (ragged OK)
    meta : dict with possible keys 'nbnd', 'nks'
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