import numpy as np
from ..utils import pauli_decompose


def _require_mpl():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception as e:
        raise ImportError(
            "Plotting requires matplotlib. Install with `pip install matplotlib`."
        ) from e


def _require_plotly():
    try:
        import plotly.graph_objects as go

        return go
    except Exception as e:
        raise ImportError(
            "Plotting requires plotly. Install with `pip install plotly`."
        ) from e


def _fmt_num(x, precision=3) -> str:
    """Format a complex number for display.

    Parameters
    ----------
    x : complex
        The complex number to format.
    precision : int
        Number of significant digits.

    Returns
    -------
    str
        Formatted string.
    """
    # If the imaginary part is negligible, print as a real number.
    if abs(x.imag) < 1e-10:
        if x.real == 1:
            return ""
        elif x.real == -1:
            return "-"
        else:
            return f"{x.real:.{precision}g}"
    elif abs(x.real) < 1e-10:
        if x.imag == 1:
            return "i"
        elif x.imag == -1:
            return "-i"
        else:
            return f"{x.imag:.{precision}g}i"
    else:
        return f"({x:.{precision}g})"


def _pauli_decompose_str(M, precision=3, use_unicode=False) -> str:
    """
    Decompose a 2x2 matrix M in terms of the Pauli matrices and return
    a string representation.

    That is, find coefficients a0, a1, a2, a3 such that:

        M = a0 * I + a1 * sigma_x + a2 * sigma_y + a3 * sigma_z

    Parameters
    ----------
    M : array-like shape (2, 2)
        A 2x2 complex matrix to decompose.
    precision : int
        Number of significant digits for the coefficients.
    use_unicode : bool
        Whether to use Unicode symbols for the Pauli matrices.

    Returns
    -------
    str
        A string representing the decomposition.
    """
    a0, a1, a2, a3 = pauli_decompose(M)

    # Build a list of terms, including only those with non-negligible coefficients.
    terms = []
    latex = [r"\sigma_0", r"\sigma_x", r"\sigma_y", r"\sigma_z"]
    unicode = [r"𝟙", r"σ_x", r"σ_y", r"σ_z"]
    latex_or_unicode = unicode if use_unicode else latex
    if abs(a0) > 1e-10:
        terms.append(rf"{_fmt_num(a0, precision=precision)} {latex_or_unicode[0]}")
    if abs(a1) > 1e-10:
        terms.append(rf"{_fmt_num(a1, precision=precision)} {latex_or_unicode[1]}")
    if abs(a2) > 1e-10:
        terms.append(rf"{_fmt_num(a2, precision=precision)} {latex_or_unicode[2]}")
    if abs(a3) > 1e-10:
        terms.append(rf"{_fmt_num(a3, precision=precision)} {latex_or_unicode[3]}")

    # If all coefficients are zero, return "0".
    if not terms:
        return "0"

    return " + ".join(terms).replace("+ -", "- ")


def _proj(v, proj_plane=None):
    v = np.array(v, dtype=float)
    if v.ndim != 1:
        raise ValueError("Input vector must be 1D.")
    if v.shape[0] <= 1:
        coord_x = v[0]
        coord_y = 0
    elif v.shape[0] == 2:
        coord_x = v[0]
        coord_y = v[1]
    elif v.shape[0] == 3:
        if proj_plane is not None:
            coord_x = v[proj_plane[0]]
            coord_y = v[proj_plane[1]]
        else:
            coord_x = v[0]
            coord_y = v[1]
    else:
        raise ValueError("Input vector must have 1, 2, or 3 elements.")
    return [coord_x, coord_y]
