import numpy as np
import logging
import copy
from itertools import product

from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.info("Lattice module loaded.")

__all__ = ["Lattice", "SymmetryOperation"]


@dataclass
class SymmetryOperation:
    """Symmetry operation acting in *reduced coordinates* of real space.

    R acts on reduced coordinates r -> R @ r + t. Periodic components will
    be wrapped modulo 1 when applied to orbitals. For k-space reduced coords,
    the action is k -> (R^{-T}) @ k.

    Attributes
    ----------
    R : np.ndarray
        Square matrix of shape (dim_r, dim_r) acting on reduced real-space coords.
    t : np.ndarray
        Translation vector in reduced coordinates, shape (dim_r,).
    label : str
        Human-readable label (e.g., 'C3z', 'Mx').
    time_reversal : bool
        If True, indicates the operation includes time reversal.
    """
    R: np.ndarray
    t: np.ndarray
    label: str = ""
    time_reversal: bool = False


class Lattice():
    r"""Class for lattice structure.

    ..  versionadded:: 2.0.0
    
    Parameters
    ----------
    lat_vecs : array_like
        Array of shape (dim_r, dim_r) containing the real-space lattice vectors as rows
        in Cartesian coordinates.
    orb_vecs : array_like, int
        Array of shape (norb, dim_r) containing the orbital positions as rows
        in reduced coordinates (fractions of the lattice vectors). If ``orb_vecs`` 
        is an integer, it specifies the number of orbitals at the origin.
    periodic_dirs : list of int, optional
        Indices of real-space lattice directions that are periodic. The order of
        entries defines the basis/order used for ``dim_k`` quantities (e.g., k-vectors
        and reciprocal vectors). If None (default), no directions are periodic.
    
    Notes
    -----
    - The dimension of the real-space lattice, `dim_r`, is inferred from the shape of `lat_vecs`.
    - The dimension of the k-space, `dim_k`, is inferred from the number of True entries in `periodic_dirs`.
    - The lattice vectors must form a right-handed system with non-zero volume.
    - Orbital positions are given in reduced coordinates, i.e., fractions of the lattice vectors.
    - Works for 0D, 1D, 2D, and 3D lattices. For 0D, use empty arrays for `lat_vecs` and an
        integer for `orb_vecs`.
    
    """

    def __init__(
        self,
        lat_vecs: np.ndarray,
        orb_vecs: np.ndarray,        
        periodic_dirs = None,
        symmetry_ops: Optional[List[SymmetryOperation]] = None,
    ):
        if periodic_dirs is None:
            logger.info("All lattice directions are considered open (non-periodic).")
            periodic_dirs = []
        elif isinstance(periodic_dirs, (list, tuple, np.ndarray)):
            periodic_dirs = list(periodic_dirs)
            if len(periodic_dirs) > len(lat_vecs):
                raise ValueError(
                    "Wrong periodic_dirs length. Must be of length <= dim_r."
                )
        else:
            raise TypeError("periodic_dirs must be a list of integers.")

        self._periodic_dirs = periodic_dirs

        self._set_lat_vecs(lat_vecs)
        self._set_orb_vecs(orb_vecs)

        # Initialize symmetry operations (stored in reduced-coordinate representation)
        self._sym_ops: List[SymmetryOperation] = []
        if symmetry_ops is not None:
            for op in symmetry_ops:
                self.add_symmetry_operation(op.R, op.t, label=op.label, time_reversal=op.time_reversal)


    def _set_orb_vecs(self, orb_vecs):

        if isinstance(orb_vecs, int):
            if orb_vecs < 1:
                raise ValueError("Number of orbitals must be positive.")
            orb_vecs = np.zeros((orb_vecs, self.dim_r), dtype=float)
        elif isinstance(orb_vecs, (list, np.ndarray)):
            orb_vecs = np.array(orb_vecs, dtype=float)
            if orb_vecs.ndim != 2 or orb_vecs.shape[1] != self.dim_r:
                raise ValueError(
                    "Wrong orb array dimensions. Must have shape (norb, dim_r)."
                )
        else:
            raise TypeError("Orbital vectors must be an integer, list, or numpy array.")

        if orb_vecs.shape[1] != self.dim_r:
            raise ValueError(
                "Orbital vectors have wrong shape. Must have shape (norb, dim_r)."
            )

        self._orb_vectors = orb_vecs

        if hasattr(self, '_lat_vectors'):
            self._orb_vecs_cart = orb_vecs @ self._lat_vectors


    def _set_lat_vecs(self, lat_vecs):
        if isinstance(lat_vecs, (list, np.ndarray)):
            lat_vecs = np.array(lat_vecs, dtype=float)
        else:
            raise TypeError("Lattice vectors must be a list or numpy array.")

        if lat_vecs.shape[0] == 0:
            lat_vecs = np.identity(0, dtype=float)

        if lat_vecs.shape[1] != lat_vecs.shape[0]:
            raise ValueError(
                "Wrong lat array dimensions. Must have shape (dim_r, dim_r)."
            )

        if lat_vecs.shape[0] > 3:
            raise ValueError("Argument dim_r must be from 0 to 3.")
        if lat_vecs.shape[0] > 0:
            det_lat = np.linalg.det(lat_vecs)
            if det_lat < 0:
                raise ValueError("Lattice vectors need to form right handed system.")
            elif det_lat < 1e-10:
                raise ValueError("Volume of unit cell is zero.")
        
        self._lat_vectors = lat_vecs

        # Cell volume
        if self.dim_r == 0:
            self._cell_vol = 0.0
        else:
            vol = np.sqrt(np.linalg.det(lat_vecs @ lat_vecs.T))
            self._cell_vol = vol

        # Reciprocal lattice
        self._recip_lat = self._get_recip_lat() if self.dim_k > 0 else None
        if self.dim_k == 0:
            self._recip_vol = 0.0
        else:
            self._recip_vol = np.sqrt(np.linalg.det(self._recip_lat @ self._recip_lat.T))

        if hasattr(self, '_orb_vectors'):
            self._orb_vecs_cart = self._orb_vectors @ self.lat_vecs


    # Mutable properties with getters and setters
    # Use copy to prevent external modification of internal arrays
    # Setters validate the new values
    # and update dependent properties accordingly
    # (e.g., updating reciprocal lattice when periodic_dirs change)
    # This ensures consistency of the lattice representation

    @property
    def symmetry_ops(self) -> List[SymmetryOperation]:
        """Return a copy of the list of symmetry operations.
        """
        # Deep copy of arrays to protect internal state
        copied: List[SymmetryOperation] = []
        for op in self._sym_ops:
            copied.append(
                SymmetryOperation(
                    R=op.R.copy(), t=op.t.copy(), label=str(op.label), time_reversal=bool(op.time_reversal)
                )
            )
        return copied

    @property
    def periodic_dirs(self) -> list[int]:
        """List of periodic directions."""
        return self._periodic_dirs
    
    @property
    def orb_vecs(self) -> np.ndarray:
        """Orbital vectors in reduced coordinates with shape ``(norb, dim_r)``.

        .. versionadded:: 2.0.0
        """
        return self._orb_vectors.copy()
    
    @property
    def lat_vecs(self) -> np.ndarray:
        """Lattice vectors in Cartesian coordinates with shape ``(dim_r, dim_r)``.

        .. versionadded:: 2.0.0
        """
        return self._lat_vectors.copy()

    @lat_vecs.setter
    def lat_vecs(self, new_lat_vecs: np.ndarray):
        self._set_lat_vecs(new_lat_vecs)
    @orb_vecs.setter
    def orb_vecs(self, new_orb_vecs: np.ndarray):
        self._set_orb_vecs(new_orb_vecs)
    @periodic_dirs.setter
    def periodic_dirs(self, new_per: list[int]):
        if not isinstance(new_per, (list, tuple, np.ndarray)):
            raise TypeError("periodic_dirs must be a list of integers.")
        new_per = list(new_per)
        if hasattr(self, '_lat_vectors') and len(new_per) > len(self._lat_vectors):
            raise ValueError(
                "Wrong periodic_dirs length. Must be of length <= dim_r."
            )

        self._periodic_dirs = new_per

        if hasattr(self, '_lat_vectors'):
            # Update reciprocal lattice since periodic directions may have changed
            self._recip_lat = self._get_recip_lat() if self.dim_k > 0 else None
            if self.dim_k == 0:
                self._recip_vol = 0.0
            else:
                self._recip_vol = np.sqrt(np.linalg.det(self._recip_lat @ self._recip_lat.T))

    # Read-only properties inferred from mutable attributes
    @property
    def dim_r(self) -> int:
        """The dimensionality of real space."""
        return self._lat_vectors.shape[0]

    @property
    def dim_k(self) -> int:
        """The dimensionality of reciprocal space (periodic directions)."""
        return len(self._periodic_dirs)

    @property
    def norb(self) -> int:
        """The number of orbitals in the lattice."""
        return self._orb_vectors.shape[0]

    @property
    def recip_lat_vecs(self) -> np.ndarray:
        """Reciprocal lattice vectors in Cartesian coordinates with shape ``(dim_k, dim_r)``.

        .. versionadded:: 2.0.0
        """
        if self._recip_lat is None:
            raise ValueError("Reciprocal lattice vectors are not defined for zero-dimensional k-space.")
        return self._recip_lat.copy()
    
    @property   
    def recip_volume(self) -> float:
        """Volume of the reciprocal unit cell in Cartesian coordinates.

        .. versionadded:: 2.0.0
        """
        return self._recip_vol
    
    @property
    def cell_volume(self) -> float:
        """Volume of the real-space unit cell in Cartesian coordinates.

        .. versionadded:: 2.0.0
        """
        return self._cell_vol
    
    def __str__(self) -> str:
        return self.report(show=False)
    
    def _report_list(self) -> list:
        output = []
        header = (
            "----------------------------------------\n"
            "       Lattice structure report         \n"
            "----------------------------------------"
        )
        output.append(header)
        output.append(f"r-space dimension           = {self.dim_r}")
        output.append(f"k-space dimension           = {self.dim_k}")
        output.append(f"periodic directions         = {self.periodic_dirs}")
        output.append(f"number of orbitals          = {self.norb}")

        formatter = {
                "float_kind": lambda x: f"{0:6.3f}" if abs(x) < 1e-10 else f"{x:6.3f}"
            }
        
        output.append("\nLattice vectors (Cartesian):")
        for i, vec in enumerate(self.lat_vecs):
            output.append(
                f"  # {i} ===> {np.array2string(vec, formatter=formatter, separator=', ')}"
            )

        output.append(f"Volume of unit cell (Cartesian) = {self.cell_volume:5.3f} [A^d]\n")

        if self.dim_k > 0:
            output.append("Reciprocal lattice vectors (Cartesian):")
            for i, vec in enumerate(self.recip_lat_vecs):
                output.append(
                    f"  # {i} ===> {np.array2string(vec, formatter=formatter, separator=', ')}"
                )
            output.append(f"Volume of reciprocal unit cell = {self.recip_volume:5.3f} [A^-d]\n")
        
        output.append("Orbital vectors (Cartesian):")
        for i, orb in enumerate(self._orb_vecs_cart):
            output.append(
                f"  # {i} ===> {np.array2string(orb, formatter=formatter, separator=', ')}"
            )

        output.append("Orbital vectors (fractional):")
        for i, orb in enumerate(self.orb_vecs):
            output.append(
                f"  # {i} ===> {np.array2string(orb, formatter=formatter, separator=', ')}"
            )

        # Symmetry summary
        if hasattr(self, "_sym_ops") and self._sym_ops:
            output.append("Symmetry operations (reduced-coord representation):")
            for i, op in enumerate(self._sym_ops):
                R_str = np.array2string(op.R, formatter=formatter, separator=", ")
                t_str = np.array2string(op.t, formatter=formatter, separator=", ")
                tr = ", TR" if op.time_reversal else ""
                label = f"[{op.label}]" if op.label else ""
                output.append(f"  # {i} {label}{tr}  R=\n{R_str}")
                output.append(f"           t={t_str}")
            output.append("")
        else:
            output.append("No symmetry operations registered.")
            output.append("")


        output.append("----------------------------------------")
    
        return output

    def report(self, show: bool = True) -> str:
        """Generate a report of the lattice properties.

        Parameters
        ----------
        show : bool, optional
            If True, prints the report to standard output (default).
            If False, only returns the report string.

        Returns
        -------
        str
            The report string.
        """
        output = self._report_list()
        if show:
            print("\n".join(output))
        else:
            return "\n".join(output)
        

    def get_orb_vecs(self, cartesian=False):
        """Return orbital positions.

        Parameters
        ----------
        cartesian : bool, optional
            If True, returns orbital positions in Cartesian coordinates.
            If False, returns reduced coordinates (default).

        Returns
        -------
        np.ndarray
            Array of orbital positions, shape (norb, dim_r).
        """
        if cartesian:
            return self._orb_vecs_cart.copy()
        else:
            return self.orb_vecs

    def get_lat_vecs(self):
        """Return lattice vectors in Cartesian coordinates.

        Returns
        -------
        np.ndarray
            Lattice vectors, shape ``(dim_r, dim_r)``.
        """
        return self.lat_vecs
    
    def get_recip_lat_vecs(self):
        """Return reciprocal lattice vectors in Cartesian coordinates.

        Returns
        -------
        np.ndarray
            Reciprocal lattice vectors, shape ``(dim_k, dim_r)``.

        Raises
        ------
        ValueError
            If the k-space dimension ``dim_k`` is zero (no periodic directions).
        """
        return self.recip_lat_vecs

    def _get_recip_lat(self):
        r"""Reciprocal lattice vectors in inverse Cartesian coordinates.

        Returns
        -------
        np.ndarray
            Array of shape (dim_k, dim_r): rows are the reciprocal vectors :math:`\mathbf{b}_i` 
            in :math:`\mathbb{R}^{\texttt{dim_r}}`
            satisfying :math:`\mathbf{a}_i \cdot \mathbf{b}_j = 2\pi \delta_{ij}`, 
            where :math:`\mathbf{a}_i` are the periodic real-space lattice vectors that 
            define k-space.

        Notes
        -----
        - Works for ``dim_k <= dim_r``. When ``dim_k < dim_r``, returns the minimum-norm solution.
        - Requires the periodic real-space vectors (rows of A_sub) to be linearly independent.
        """
        if self.dim_k == 0:
            logger.warning("Reciprocal lattice vectors are not defined for zero-dimensional k-space.")
            return None

        # Select the real-space lattice vectors that generate k-space.
        # Prefer an explicit list (e.g. self.per holds indices of periodic directions).
        # Fallback: take the first dim_k lattice vectors.
        lat = np.asarray(self.lat_vecs)            # shape (dim_r, dim_r) in Cartesian coords
        per = np.asarray(self.periodic_dirs, dtype=int)
    
        if per.size != self.dim_k:
            raise ValueError(f"'per' must list exactly dim_k={self.dim_k} periodic directions.")
        A_sub = lat[per, :]                    # (dim_k, dim_r)

        # Check linear independence of the chosen periodic vectors
        if np.linalg.matrix_rank(A_sub) != self.dim_k:
            raise ValueError(
                "Periodic real-space vectors are not linearly independent; "
                "cannot construct reciprocal lattice for k-subspace."
            )

        # Minimum-norm reciprocal set in the embedding R^{dim_r}:
        # rows b_i satisfy A_sub @ B^T = 2pi I_{dim_k}
        G = A_sub @ A_sub.T             # (dim_k, dim_k) Gram matrix
        X = np.linalg.solve(G, A_sub)   # (dim_k, dim_r)
        B = (2 * np.pi) * X             # (dim_k, dim_r)

        return B
    
    # -------------------------
    # Symmetry helper utilities
    # -------------------------
    def _wrap_reduced(self, r: np.ndarray) -> np.ndarray:
        """Wrap reduced coordinates on periodic components to [0,1).
        Non-periodic components are left unchanged.
        """
        r = np.array(r, float)
        out = r.copy()
        if self.dim_r == 0:
            return out
        if len(self.periodic_dirs) == 0:
            return out
        per = np.array(self.periodic_dirs, int)
        out[..., per] = np.mod(out[..., per], 1.0)
        return out

    def _reduced_allclose(self, a: np.ndarray, b: np.ndarray, tol: float = 1e-8) -> bool:
        """Compare two reduced-coordinate vectors up to lattice translations on periodic axes."""
        a = np.array(a, float)
        b = np.array(b, float)
        if a.shape != b.shape:
            return False
        if len(self.periodic_dirs) == 0:
            return np.allclose(a, b, atol=tol, rtol=0)
        per = np.array(self.periodic_dirs, int)
        # match periodic components modulo 1
        da = a.copy(); db = b.copy()
        da[..., per] = np.mod(da[..., per], 1.0)
        db[..., per] = np.mod(db[..., per], 1.0)
        return np.allclose(da, db, atol=tol, rtol=0)

    def add_symmetry_operation(self, R: np.ndarray, t: Optional[np.ndarray] = None, *, label: str = "", time_reversal: bool = False):
        """Register a symmetry operation in reduced coordinates.

        Parameters
        ----------
        R : array_like, shape (dim_r, dim_r)
            Matrix acting on reduced real-space coordinates.
        t : array_like, shape (dim_r,), optional
            Translation in reduced coordinates. Defaults to zero.
        label : str, optional
            Human-readable label.
        time_reversal : bool, optional
            Whether the operation includes time reversal.
        """
        R = np.array(R, float)
        if R.shape != (self.dim_r, self.dim_r):
            raise ValueError("R must have shape (dim_r, dim_r)")
        t = np.zeros(self.dim_r, float) if t is None else np.array(t, float)
        if t.shape != (self.dim_r,):
            raise ValueError("t must have shape (dim_r,)")
        # Basic validation: periodic blocks approximately integer for crystal symmetry
        if self.dim_r > 0 and len(self.periodic_dirs) > 0:
            per = np.array(self.periodic_dirs, int)
            Rpp = R[np.ix_(per, per)]
            if not np.allclose(Rpp, np.rint(Rpp), atol=1e-8):
                logger.warning("Symmetry R block on periodic subspace is not integer-like; ensure definition is in reduced basis.")
        self._sym_ops.append(SymmetryOperation(R=R, t=t, label=label, time_reversal=bool(time_reversal)))

    def clear_symmetry(self):
        """Remove all registered symmetry operations."""
        self._sym_ops.clear()

    def apply_symmetry_to_orbital(self, op: SymmetryOperation, r_red: np.ndarray) -> np.ndarray:
        """Apply a symmetry op to a single orbital position in reduced coords and wrap periodic components."""
        r_red = np.array(r_red, float)
        rr = op.R @ r_red + op.t
        return self._wrap_reduced(rr)

    def check_symmetry(self, tol: float = 1e-8) -> dict:
        """Validate that registered symmetries map lattice and orbitals onto themselves.

        Returns a dict with keys 'lattice_ok', 'orbitals_ok', and lists of offending operations.
        """
        result = {"lattice_ok": True, "orbitals_ok": True, "bad_ops_lattice": [], "bad_ops_orbitals": []}
        if not self._sym_ops:
            return result
        # Lattice check: R must map periodic subspace to itself via integer matrix
        if len(self.periodic_dirs) > 0:
            per = np.array(self.periodic_dirs, int)
            nonper = np.array([i for i in range(self.dim_r) if i not in per], int)
            for i, op in enumerate(self._sym_ops):
                Rpp = op.R[np.ix_(per, per)]
                # integer-like block on periodic subspace
                if not np.allclose(Rpp, np.rint(Rpp), atol=1e-8):
                    result["lattice_ok"] = False
                    result["bad_ops_lattice"].append(i)
                    continue
                # no mixing between periodic and nonperiodic directions
                if nonper.size > 0:
                    if np.any(np.abs(op.R[np.ix_(per, nonper)]) > 1e-10) or np.any(
                        np.abs(op.R[np.ix_(nonper, per)]) > 1e-10
                    ):
                        result["lattice_ok"] = False
                        result["bad_ops_lattice"].append(i)
                        continue
                # optional: unimodular in periodic subspace (det ±1)
                det = np.linalg.det(Rpp)
                if not np.isclose(abs(round(det)), 1.0, atol=1e-8):
                    # not strictly required in all contexts; warn via lattice_ok but keep info
                    result["lattice_ok"] = False
                    if i not in result["bad_ops_lattice"]:
                        result["bad_ops_lattice"].append(i)
        # Orbitals check: each orbital position should be carried to some orbital modulo periodic translations
        orbs = self.orb_vecs
        for i, op in enumerate(self._sym_ops):
            for r in orbs:
                r_map = self.apply_symmetry_to_orbital(op, r)
                if not np.any([self._reduced_allclose(r_map, q, tol) for q in orbs]):
                    result["orbitals_ok"] = False
                    if i not in result["bad_ops_orbitals"]:
                        result["bad_ops_orbitals"].append(i)
                    break
        return result

    def little_group(self, k_red: np.ndarray, tol: float = 1e-8) -> List[int]:
        """Return indices of symmetry operations that leave reduced k invariant modulo G.

        Notes
        -----
        - ``k_red`` is in the periodic subspace with length ``dim_k`` and component
          order matching ``self.periodic_dirs``.
        - For reduced coordinates in real space, the corresponding k transform is
          ``k -> s (R_pp^{-T}) k`` where ``s = -1`` if the operation includes time reversal.
        """
        if self.dim_k == 0:
            return list(range(len(self._sym_ops)))
        k = np.array(k_red, float).reshape(-1)
        if k.shape[0] != self.dim_k:
            raise ValueError(f"k_red must have length dim_k={self.dim_k}")
        per = np.array(self.periodic_dirs, int)
        nonper = np.array([i for i in range(self.dim_r) if i not in per], int)
        keepers = []
        for idx, op in enumerate(self._sym_ops):
            # Disallow mixing periodic with nonperiodic directions
            if nonper.size > 0:
                if np.any(np.abs(op.R[np.ix_(per, nonper)]) > 1e-10) or np.any(
                    np.abs(op.R[np.ix_(nonper, per)]) > 1e-10
                ):
                    continue
            try:
                Rpp = op.R[np.ix_(per, per)]
                Rinvt_pp = np.linalg.inv(Rpp).T
            except np.linalg.LinAlgError:
                continue
            s = -1.0 if op.time_reversal else 1.0
            k_map = s * (Rinvt_pp @ k)
            # compare modulo integers in the k-subspace
            if np.allclose(np.mod(k_map, 1.0), np.mod(k, 1.0), atol=tol, rtol=0):
                keepers.append(idx)
        return keepers

    def star_of_k(self, k_red: np.ndarray, tol: float = 1e-8) -> np.ndarray:
        """Generate the star of k (unique modulo G) in the k-subspace basis.

        ``k_red`` and the returned array live in a ``dim_k``-dimensional space
        ordered as ``self.periodic_dirs``.
        """
        if self.dim_k == 0:
            return np.zeros((1, 0))
        k = np.array(k_red, float).reshape(-1)
        if k.shape[0] != self.dim_k:
            raise ValueError(f"k_red must have length dim_k={self.dim_k}")
        per = np.array(self.periodic_dirs, int)
        nonper = np.array([i for i in range(self.dim_r) if i not in per], int)
        images = []
        for op in self._sym_ops:
            # Disallow mixing periodic with nonperiodic directions
            if nonper.size > 0:
                if np.any(np.abs(op.R[np.ix_(per, nonper)]) > 1e-10) or np.any(
                    np.abs(op.R[np.ix_(nonper, per)]) > 1e-10
                ):
                    continue
            try:
                Rpp = op.R[np.ix_(per, per)]
                Rinvt_pp = np.linalg.inv(Rpp).T
            except np.linalg.LinAlgError:
                continue
            s = -1.0 if op.time_reversal else 1.0
            k_map = s * (Rinvt_pp @ k)
            images.append(np.mod(k_map, 1.0))
        # unique modulo integers in k-subspace
        uniq = []
        for v in images:
            if not any(np.allclose(np.mod(v, 1.0), np.mod(u, 1.0), atol=tol, rtol=0) for u in uniq):
                uniq.append(v)
        return np.stack(uniq, axis=0)

    def reduce_kmesh(self, kpoints: np.ndarray, tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reduce a k-mesh to the irreducible wedge using the registered symmetry.

        Parameters
        ----------
        kpoints : array_like, shape (Nk, dim_k)
            Reduced k-points on the BZ torus in the basis/order matching ``periodic_dirs``.

        Returns
        -------
        k_irred : (Ni, dim_k)
            Representatives of symmetry orbits.
        weights : (Ni,)
            Orbit sizes divided by total Nk (sum to 1).
        map_full_to_irred : (Nk,)
            Indices mapping each original k to its representative index in ``k_irred``.
        """
        if self.dim_k == 0 or not self._sym_ops:
            Nk = len(kpoints)
            return np.array(kpoints, float), np.ones(Nk) / Nk, np.arange(Nk)
        pts = np.array(kpoints, float)
        if pts.ndim != 2 or pts.shape[1] != self.dim_k:
            raise ValueError(f"kpoints must have shape (Nk, dim_k={self.dim_k})")

        per = np.array(self.periodic_dirs, int)
        nonper = np.array([i for i in range(self.dim_r) if i not in per], int)

        used = np.zeros(len(pts), dtype=bool)
        reps = []
        weights = []
        mapping = -np.ones(len(pts), dtype=int)

        # Precompute allowed k-space transforms for efficiency
        transforms = []  # list of (Rinvt_pp, s)
        for op in self._sym_ops:
            if nonper.size > 0:
                if np.any(np.abs(op.R[np.ix_(per, nonper)]) > 1e-10) or np.any(
                    np.abs(op.R[np.ix_(nonper, per)]) > 1e-10
                ):
                    continue
            try:
                Rpp = op.R[np.ix_(per, per)]
                Rinvt_pp = np.linalg.inv(Rpp).T
            except np.linalg.LinAlgError:
                continue
            s = -1.0 if op.time_reversal else 1.0
            transforms.append((Rinvt_pp, s))

        for i, k in enumerate(pts):
            if used[i]:
                continue
            orbit = []
            idxs = []
            for j, k2 in enumerate(pts):
                if used[j]:
                    continue
                # Check if k2 is symmetry-equivalent to k
                equivalent = False
                for Rinvt_pp, s in transforms:
                    k_map = s * (Rinvt_pp @ k)
                    if np.allclose(np.mod(k_map, 1.0), np.mod(k2, 1.0), atol=tol, rtol=0):
                        equivalent = True
                        break
                if equivalent:
                    orbit.append(k2)
                    idxs.append(j)
            # choose representative (lexicographic on wrapped coords)
            orb_arr = np.array(orbit)
            wrapped = np.mod(orb_arr, 1.0)
            rep_idx_local = int(np.lexsort(wrapped.T)[0])
            rep = orbit[rep_idx_local]
            reps.append(rep)
            for j in idxs:
                used[j] = True
                mapping[j] = len(reps) - 1
            weights.append(len(idxs))
        reps = np.array(reps, float)
        weights = np.array(weights, float)
        weights /= weights.sum()
        return reps, weights, mapping


    def cut_piece(self, num, fin_dir) -> "Lattice":
        """Cut a (d-1)-dimensional piece out of a d-dimensional Lattice.
        
        Constructs a (d-1)-dimensional Lattice out of a
        d-dimensional one by repeating the unit cell a given number of
        times along one of the periodic lattice vectors. 
        
        Parameters
        ----------
        num : int
            How many times to repeat the unit cell.
        fin_dir : int
            Index of the real space lattice vector along
            which you no longer wish to maintain periodicity.

        Returns
        -------
        fin_lat : Lattice
            Object of type :class:`pythtb.Lattice` representing a cutout
            lattice.

        See Also
        ---------
        :ref:`haldane-fin-nb` : For an example
        :ref:`haldane-edge-nb` : For an example

        Notes
        -----
        - Orbitals in `fin_lat` are numbered so that the `i`-th orbital of the `n`-th unit
          cell has index ``i + norb * n`` (here `norb` is the number of orbitals in the original model).
        - The real-space lattice vectors of the returned model are the same as those of
          the original model; only the dimensionality of reciprocal space
          is reduced.

        Examples
        --------
        Construct two-dimensional model B out of three-dimensional model A

        >>> A = TBModel(3, 3, ...)

        model A by repeating model along second lattice vector ten times

        >>> B = A.cut_piece(10, 1)

        Further cut two-dimensional model B into one-dimensional model
        A by repeating unit cell twenty times along third lattice
        vector and allow hoppings from one edge to the other

        >>> C = B.cut_piece(20, 2, glue_edgs=True)

        """
        if self.dim_k == 0:
            raise Exception("Lattice is already finite")
        if not isinstance(num, int):
            raise TypeError("Parameter `num` is not an integer")

        # check value of num
        if num < 1:
            raise ValueError("Argument num must be positive!")

        # generate orbitals of a finite model
        fin_orb = []
        for i in range(num):  # go over all cells in finite direction
            for j in range(self.norb):  # go over all orbitals in one cell
                orb_tmp = np.copy(self.orb_vecs[j, :])
                # change coordinate along finite direction
                orb_tmp[fin_dir] += float(i)
                fin_orb.append(orb_tmp)
        fin_orb = np.array(fin_orb)

        fin_per = copy.deepcopy(self.periodic_dirs) # copy list of periodic directions
        # check if you can make model finite along this direction
        if fin_dir not in fin_per:
            raise Exception("Can not make model finite along this direction!")
        
        # remove index which is no longer periodic
        fin_per.remove(fin_dir)

        fin_lat = Lattice(self.lat_vecs, fin_orb, periodic_dirs=fin_per)
        return fin_lat
    
    def add_orb(self, orb_pos):
        """Add an orbital to the lattice.

        Parameters
        ----------
        orb_pos : array_like
            Position of the new orbital in reduced coordinates (fractions of the lattice vectors).
            Must be of length ``dim_r``.

        Returns
        -------
        None

        Notes
        -----
        - The new orbital is added at the end of the list of orbitals.
        - The number of orbitals ``norb`` is updated accordingly.
        """
        if isinstance(orb_pos, (float, int)):
            orb_pos = np.array([orb_pos], float)
        elif isinstance(orb_pos, list):
            orb_pos = np.array(orb_pos, float)
        elif not isinstance(orb_pos, np.ndarray):
            raise TypeError(f"Expected array_like or float, got {type(orb_pos)}")
        
        if orb_pos.ndim != 1 or orb_pos.shape[0] != self.dim_r:
            raise ValueError(f"Orbital position must be of length {self.dim_r}.")
        
        self._orb_vectors = np.vstack([self._orb_vectors, orb_pos])
        self._orb_vecs_cart = self._orb_vectors @ self.lat_vecs

    def remove_orb(self, to_remove):
        """Remove an orbital from the lattice.

        Parameters
        ----------
        to_remove : array-like or int
            List of orbital indices to be removed, or index of single orbital to be removed

        Returns
        -------
        None

        Notes
        -----
        - The number of orbitals ``norb`` is updated accordingly.
        - Raises an error if the index is out of bounds.
        """
        if isinstance(to_remove, int):
            indices = [to_remove]
        elif isinstance(to_remove, (list, np.ndarray)):
            indices = list(to_remove)
        else:
            raise TypeError("to_remove must be an integer or a list of integers.")

        for index in indices:
            if not isinstance(index, int):
                raise TypeError("All indices in to_remove must be integers.")
            if index < 0 or index >= self.norb:
                raise ValueError("Index out of bounds.")
            
        # check that all indices are unique
        if len(indices) != len(set(indices)):
            raise ValueError("All indices in to_remove must be unique.")

        # put the orbitals to be removed in descending order
        orb_index = sorted(indices, reverse=True)

        # remove indices one by one
        for i, orb_ind in enumerate(orb_index):
            # adjust variables
            self._orb_vectors = np.delete(self._orb_vectors, orb_ind, 0)

        self._orb_vecs_cart = self._orb_vectors @ self.lat_vecs


    def change_nonperiodic_vector(
        self, 
        np_dir: int, 
        new_lat_vec=None
    ) -> "Lattice":
        """Change non-periodic lattice vector 

        .. versionchanged:: 2.0.0
            Parameter `to_home_supress_warning` has been renamed to `to_home_warning`.
            Note: this change inverts the meaning of the boolean parameter.
        
        Returns tight-binding model :class:`pythtb.TBModel` in which one of
        the non-periodic "lattice vectors" is changed.  Non-periodic vectors are those 
        elements of `lat` that are not listed as periodic with the `per` parameter.

        The returned object has modified reduced coordinates of orbitals, 
        consistent with the new choice of `lat`. Therefore, the actual 
        (Cartesian) coordinates of orbitals in original and new :class:`Lattice`
        are the same.

        Parameters
        ----------
        np_dir : int
            Index of non-periodic lattice vector to change.

        new_lat_vec : array_like, optional
            The new non-periodic lattice vector. If None (default), the new
            non-periodic lattice vector is the same as the original one except
            that all components in the periodic space have been projected out
            (so that the new non-periodic vector is perpendicular to all
            periodic vectors).

        See Also
        --------
        per
        :ref:`boron-nitride-nb` : For an example.

        Notes
        -----
        - This function is especially useful after using function cut_piece to create slabs, rods, or ribbons.
        - By default, the new non-periodic vector is constructed
          from the original by removing all components in the periodic
          space. This ensures that the Berry phases computed in the
          periodic space correspond to the usual expectations.
        - For example, after this change, the Berry phase computed for a
          ribbon depends only on the location of the Wannier center
          in the extended direction, not on its location in the
          transverse direction. Alternatively, the new nonperiodic
          vector can be set explicitly via the `new_latt_vec` parameter.

        Examples
        --------
        Modify slab model so that nonperiodic third vector is perpendicular to the slab

        >>> nnp_tb = tb.change_nonperiodic_vector(2)

        """
        if not isinstance(np_dir, int):
            raise TypeError("Argument np_dir must be an integer")
        if np_dir in self.periodic_dirs:
            raise ValueError(f"Selected direction {np_dir} is not nonperiodic")

        if new_lat_vec is None:
            # construct new nonperiodic lattice vector
            per_temp = np.zeros_like(self.lat_vecs)
            for direc in self.periodic_dirs:
                per_temp[direc] = self.lat_vecs[direc]
            # find projection coefficients onto space of periodic vectors
            coeffs = np.linalg.lstsq(per_temp.T, self.lat_vecs[np_dir], rcond=None)[0]
            projec = np.dot(self.lat_vecs.T, coeffs)
            # subtract off to get new nonperiodic vector
            np_lattice_vec = self.lat_vecs[np_dir] - projec

            if np.linalg.norm(np_lattice_vec) < 1.0e-10:
                raise ValueError(
                    """New nonperiodic vector has zero length!?"""
                )
            
            # normalize new nonperiodic vector to have same length as original
            np_lattice_vec /= np.linalg.norm(np_lattice_vec)
            np_lattice_vec *= np.linalg.norm(self.lat_vecs[np_dir])

            # check that new nonperiodic vector is perpendicular to all periodic vectors
            for i in self.periodic_dirs:
                if np.abs(np.dot(self.lat_vecs[i], np_lattice_vec)) > 1.0e-6:
                    raise ValueError(
                        """This shouldn't happen. New nonperiodic vector
                        is not perpendicular to periodic vectors!?"""
                    )
        else:
            # new_latt_vec is passed as argument
            np_lattice_vec = np.array(new_lat_vec)

            # check shape
            if np_lattice_vec.shape != (self.dim_r,):
                raise ValueError("Non-periodic vector has wrong shape.")
            if np.linalg.norm(np_lattice_vec) < 1e-10:
                raise ValueError("New non-periodic vector has zero length.")

        og_orb_cart = copy.deepcopy(self._orb_vecs_cart)

        # Define new set of lattice vectors
        new_lat = copy.deepcopy(self.lat_vecs)
        new_lat[np_dir] = np_lattice_vec

        # Update reduced orb vecs 
        new_orb = []
        for orb_cart in og_orb_cart:  
            # convert to reduced coordinates
            new_orb.append(np.linalg.solve(new_lat.T, orb_cart))

        # update lattice vectors and orbitals
        self.lat_vecs = np.array(new_lat, dtype=float)
        self.orb_vecs = np.array(new_orb, dtype=float)

        # Are cartesian coordinates of orbitals the same in two cases?
        for idx, orb_cart in enumerate(og_orb_cart):
            cart_old = orb_cart
            cart_new = self._orb_vecs_cart[idx]
            if np.max(np.abs(cart_old - cart_new)) > 1e-6:
                raise ValueError(
                    """This shouldn't happen. New choice of nonperiodic vector
                        somehow changed Cartesian coordinates of orbitals."""
                )

      
    def _shift_orb_to_home(self, to_home_warning: bool=True):
        """Shifts orbital coordinates (along periodic directions) to the home
        unit cell. 
        
        After this function is called reduced coordinates
        (along periodic directions) of orbitals will be between 0 and
        1.

        Version of pythtb 1.7.2 (and earlier) was shifting orbitals to
        home along even nonperiodic directions. In the later versions
        of the code (this present version, and future versions) we
        don't allow this anymore, as this feature might produce
        counterintuitive results.  Shifting orbitals along nonperiodic
        directions changes physical nature of the tight-binding model.
        This behavior might be especially non-intuitive for
        tight-binding models that came from the *cut_piece* function.

        Parameters
        ----------
        to_home_warning: bool, optional
            Default value is ``True``. If ``True`` prints warning messages
            about orbitals being outside the home cell (reduced coordinate larger
            than 1 or smaller than 0 along non-periodic direction). 

            Note that setting this parameter to *True* or *False* has no effect on 
            resulting coordinates of the model. 
        """

        orb_vecs_new = copy.deepcopy(self.orb_vecs)

        # go over all orbitals
        for i in range(self.norb):
            # find displacement vector needed to bring back to home cell
            disp_vec = np.zeros(self.dim_r, dtype=int)
            for k in range(self.dim_r):
                shift = np.floor(self.orb_vecs[i, k]).astype(int)

                # shift only in periodic directions
                if k in self.periodic_dirs:
                    disp_vec[k] = shift
                elif k not in self.periodic_dirs and shift != 0 and to_home_warning:  # check for shift in non-periodic directions
                    logger.warning(
                        f"Orbital {i} has reduced coordinate {self.orb_vecs[i, k]:.3f} "
                        f"along non-periodic direction {k}, which is outside the home cell."
                    )
           
            orb_vecs_new[i] -= disp_vec

        self.orb_vecs = orb_vecs_new


    def nn_orb_shell(self, n_shell: int, report: bool = False):
        """Generates shells of nearest neighbor vectors connecting orbitals in real space.

        Returns array of vectors connecting the origin to nearest 
        neighboring orbitals in the lattice. The function

        Parameters
        ----------
        n_shell : int
            Number of nearest neighbor shells to include.
        report : bool
            If True, prints a summary of the nn-shell.

        Returns
        -------
        nn_shell : list[np.ndarray[float]]
            List of :math:`\mathbf{R}` vectors in units of lattice vectors
            connecting nearest neighbor orbitals. Length is `n_shell`.
        idx_shell : list[np.ndarray[int]]
            Each entry is an array of integer shifts that takes an orbital 
            index to its nearest neighbors.
            Length is `n_shell`.
        """
        if not isinstance(n_shell, int) or n_shell < 1:  
            raise ValueError("Invalid n_shell: must be a positive integer.")

        lat_vecs = self.lat_vecs
        dim_r = self.dim_r
        orb_cart = self._orb_vecs_cart
        norb = self.norb

        # Enumerate candidate neighbors up to a reasonable window of lattice shifts
        # R in Z^{dim_r} with components in [-n_shell, n_shell]
        from itertools import product as _product

        d2_list = []          # squared distances
        R_cart_list = []      # Cartesian displacement vectors Δr
        idx_list = []         # integer meta: [i, j, R_0, ..., R_{dim_r-1}]

        shifts = list(_product(range(-n_shell-1, n_shell + 2), repeat=dim_r))
        if (0,) * dim_r in shifts:
            pass  # keep; we filter self-pairs below

        for i in range(norb):
            ri = orb_cart[i]
            for R in shifts:
                R = np.asarray(R, dtype=int)
                # precompute lattice translation vector for this R
                T_R = R @ lat_vecs  # (dim_r,)
                for j in range(norb):
                    if i == j and not np.any(R):
                        continue
                    dr = (orb_cart[j] + T_R) - ri  # Δr (Cartesian)
                    d2 = float(dr @ dr)
                    d2_list.append(d2)
                    R_cart_list.append(dr)
                    idx_list.append(np.concatenate(([i, j], R)))

        if not d2_list:
            # No neighbors (e.g., single-orbital 0D with no shifts requested)
            return [[] for _ in range(n_shell)], [[] for _ in range(n_shell)]

        R_cart_arr = np.vstack(R_cart_list)                 # (N, dim_r)
        idx_arr = np.vstack(idx_list).astype(int)           # (N, 2+dim_r)
        d2_arr = np.asarray(d2_list)

        # Numerical stability: round squared norms to cluster nearly-equal distances
        d2_rounded = np.round(d2_arr, 12)
        order = np.argsort(d2_rounded)
        d2_sorted = d2_rounded[order]
        R_sorted = R_cart_arr[order]
        idx_sorted = idx_arr[order]

        # First n_shell unique radii
        unique_d2 = []
        for val in d2_sorted:
            if not unique_d2 or val > unique_d2[-1]:
                unique_d2.append(val)
            if len(unique_d2) == n_shell:
                break
        # If fewer unique shells exist, truncate gracefully
        n_take = min(n_shell, len(unique_d2))
        unique_d2 = unique_d2[:n_take]

        # Build per-shell, per-orbital groupings
        nn_shell = []   # list over shells -> list over orbitals -> (deg_i, dim_r)
        idx_shell = []  # list over shells -> list over orbitals -> (deg_i, 2+dim_r)

        for s, d2_target in enumerate(unique_d2):
            mask_s = (d2_sorted == d2_target)
            R_s = R_sorted[mask_s]
            idx_s = idx_sorted[mask_s]

            # Split by central-orbital index i (idx column 0)
            shell_R_by_i = []
            shell_idx_by_i = []
            for i in range(norb):
                m_i = (idx_s[:, 0] == i)
                shell_R_by_i.append(R_s[m_i])
                shell_idx_by_i.append(idx_s[m_i])
            nn_shell.append(shell_R_by_i)
            idx_shell.append(shell_idx_by_i)

        # Optionally print a compact text report
        if report:
            lines = []           
            lines.append("nn-shell report (per-orbital)")
            lines.append("═" * 60)
            lines.append(f"dim_r: {dim_r}   norb: {norb}   shells: {len(unique_d2)}")
            for s, d2_target in enumerate(unique_d2, start=1):
                radius = np.sqrt(d2_target)
                total_deg = sum(Rs.shape[0] for Rs in nn_shell[s-1])
                lines.append(f"shell {s:>2}: |Δr|={radius:.8g} (degeneracy total={total_deg})")
                # Show first few for each i
                for i in range(norb):
                    Rs = nn_shell[s-1][i]
                    Id = idx_shell[s-1][i]
                    if Rs.size == 0:
                        continue
                    head = min(Rs.shape[0], 6)
                    lines.append(f"  i={i}: {Rs.shape[0]} neighbors")
                    for k in range(head):
                        j = int(Id[k, 1])
                        Rvec = Id[k, 2:2+dim_r]
                        dr_str = np.array2string(Rs[k], precision=6, floatmode='maxprec_equal', suppress_small=True)
                        R_str = np.array2string(Rvec, formatter={'int': lambda x: f"{int(x):>2}"}, separator=', ')
                        lines.append(f"     → j={j:>2}, R={R_str}   Δr={dr_str}")
                    if Rs.shape[0] > head:
                        lines.append(f"     … (+{Rs.shape[0]-head} more)")
            print("\n".join(lines))

        return nn_shell, idx_shell
    

    def nn_orb_bonds(self, n_shell: int):
        """
        Convenience wrapper around `nn_orb_shell` that returns shell radii,
        per-orbital neighbor lists (j, R), and a flattened set of unique
        (i, j, R) bonds per shell suitable for tb.set_hop(...).

        Returns
        -------
        result : dict with keys
        - 'radii' : np.ndarray shape (S,)
        - 'pairs_by_orb' : list[S] of list[norb] of int arrays (deg_i, 1+dim_r) rows [j, R...]
        - 'dr_by_orb'    : list[S] of list[norb] of float arrays (deg_i, dim_r)
        - 'deg_per_orb'  : np.ndarray shape (S, norb)
        - 'bonds_by_shell': list[S] of list[tuple] each (i, j, R_tuple) unique
        """
        nn_shell, idx_shell = self.nn_orb_shell(n_shell, report=False)

        S = len(nn_shell)
        norb = self.norb
        dim_r = self.dim_r

        # Radii: infer from any non-empty per-shell block
        radii = np.zeros(S, float)
        for s in range(S):
            found = False
            for i in range(norb):
                if len(nn_shell[s][i]) > 0:
                    radii[s] = float(np.linalg.norm(nn_shell[s][i][0]))
                    found = True
                    break
            if not found:
                radii[s] = 0.0

        # Strip the redundant 'i' column from idx_shell into pairs_by_orb
        pairs_by_orb = []
        dr_by_orb = []
        deg_per_orb = np.zeros((S, norb), int)
        bonds_by_shell = []

        for s in range(S):
            pb_i = []
            dr_i = []
            # canonical unique set of bonds (i, j, R) per shell
            seen = set()
            bonds = []
            for i in range(norb):
                Id = idx_shell[s][i]  # shape (deg_i, 2+dim_r): [i, j, R...]
                Rs = nn_shell[s][i]   # shape (deg_i, dim_r)
                if Id.size == 0:
                    pb_i.append(np.empty((0, 1 + dim_r), dtype=int))
                    dr_i.append(np.empty((0, dim_r), dtype=float))
                    continue
                # [j, R...] without the leading i
                pairs = Id[:, 1:2+dim_r].astype(int, copy=False)
                pb_i.append(pairs)
                dr_i.append(Rs)
                deg_per_orb[s, i] = pairs.shape[0]

                # Build unique bonds with a canonical orientation:
                # keep (i, j, R) as-is, and do not add its conjugate (j, i, -R)
                for k in range(pairs.shape[0]):
                    j = int(pairs[k, 0])
                    R = tuple(int(x) for x in pairs[k, 1:])
                    # define a key that equals for a bond and its conjugate
                    conj_key = (j, i, tuple(-x for x in R))
                    if conj_key in seen:
                        continue
                    key = (i, j, R)
                    if key not in seen:
                        seen.add(key)
                        bonds.append(key)
            pairs_by_orb.append(pb_i)
            dr_by_orb.append(dr_i)
            bonds_by_shell.append(bonds)

        return {
            "radii": radii,
            "pairs_by_orb": pairs_by_orb,
            "dr_by_orb": dr_by_orb,
            "deg_per_orb": deg_per_orb,
            "bonds_by_shell": bonds_by_shell,
        }


    def nn_k_shell(self, nks: tuple, n_shell: int, report: bool = False):
        """Generates shells of k-points around the Gamma point.

        Returns array of vectors connecting the origin to nearest 
        neighboring k-points in the mesh. The function

        Parameters
        ----------
        nks : tuple of int
            Number of k-points along each periodic direction. Length must be `dim_k`.
        n_shell : int
            Number of nearest neighbor shells to include.
        report : bool
            If True, prints a summary of the k-shell.

        Returns
        -------
        k_shell : list[np.ndarray[float]]
            List of :math:`\mathbf{b}` vectors in inverse units of lattice vectors
            connecting nearest neighbor k-mesh points. Length is `n_shell`.
        idx_shell : list[np.ndarray[int]]
            Each entry is an array of integer shifts that takes a k-point 
            index in the mesh to its nearest neighbors.
            Length is `n_shell`.
        """
        if not isinstance(n_shell, int) or n_shell < 1:  
            raise ValueError("Invalid n_shell: must be a positive integer.")

        recip_lat_vecs = self.recip_lat_vecs
        dim_k = self.dim_k

        if dim_k != len(nks):
            raise ValueError("Mesh is not full, cannot generate k-shells.")

        # basis vectors connecting neighboring mesh points (in inverse Cartesian units)
        dk = np.array([recip_lat_vecs[i] / nk for i, nk in enumerate(nks)])

        # array of integers e.g. in 2D for n_shell = 1 would be 
        # [-1,-1], [-1,0], [-1,1], [0,-1], [0,1], [1,-1], [1,0], [1,1]
        nnbr_idx = list(product(range(-n_shell, n_shell + 1), repeat=dim_k))
        nnbr_idx.remove((0,) * dim_k)
        nnbr_idx = np.array(nnbr_idx)
        
        # Vectors connecting k-points near Gamma point (in inverse Cartesian units)
        # (M, dim_k) @ (dim_k, dim_k) -> (M, dim_k)
        b_vecs = nnbr_idx @ dk 

        # Squared norms
        d2 = np.einsum('ij,ij->i', b_vecs, b_vecs)
        # remove numerical noise
        d2r = np.round(d2, 12)

        # Sort by increasing radius^2
        sorted_idxs = np.argsort(d2r)
        d2r_sorted = d2r[sorted_idxs]
        b_sorted = b_vecs[sorted_idxs]
        idx_sorted = nnbr_idx[sorted_idxs]

        # Unique radii^2 in order; take first n_shell shells
        unique_d2 = sorted(list(set(d2r_sorted)))  # removes repeated distances
        unique_d2 = unique_d2[:n_shell]  # keep only distances up to n_shell away

        # keep only b_vecs in n_shell shells
        k_shell = [
            b_sorted[np.isin(d2r_sorted, unique_d2[i])]
            for i in range(len(unique_d2))
        ]
        idx_shell = [
            idx_sorted[np.isin(d2r_sorted, unique_d2[i])]
            for i in range(len(unique_d2))
        ]

        if report:
            # Pretty report
            lines = []
            lines.append("k-shell report")
            lines.append("═" * 46)
            lines.append(f"dim_k: {dim_k}   nks: {nks}")
            # Compact step info: show |dk_i| and vectors
            step_norms = [np.linalg.norm(dk[i]) for i in range(dim_k)]
            steps_str  = ", ".join(f"|dk_{i}|={step_norms[i]:.6g}" for i in range(dim_k))
            lines.append(f"step sizes: {steps_str}")
            # Optional: show dk rows
            lines.append("dk vectors:")
            for i in range(dim_k):
                lines.append(f"  dk[{i}] = {np.array2string(dk[i], precision=6, floatmode='maxprec_equal', suppress_small=True)}")

            lines.append("")
            lines.append("Shells (by increasing |b|):")
            for si, (B, I) in enumerate(zip(k_shell, idx_shell), start=1):
                deg = B.shape[0]
                radius = np.sqrt(unique_d2[si-1])
                lines.append(f"  • shell {si:>2}: |b|={radius:.6g}   degeneracy={deg}")
                # Show a few representatives from this shell
                head = min(deg, 6)
                for j in range(head):
                    b_str = np.array2string(B[j], precision=6, floatmode='maxprec_equal', suppress_small=True)
                    i_str = np.array2string(I[j], formatter={'int':lambda x: f"{x:>2}"})
                    lines.append(f"      idx={i_str}   b={b_str}")
                if deg > head:
                    lines.append(f"      … (+{deg-head} more)")
            print("\n".join(lines))

        return k_shell, idx_shell
    
    
    def k_shell_weights(self, nks: tuple, n_shell : int = 1, report: bool = False):
        r"""Generates the finite difference weights on a k-shell.

        This function uses the k-shells generated by :func:`nn_k_shell` 
        to compute the  weights for finite difference approximations of 
        :math:`\nabla_{\mathbf{k}}` on a Monkhorst-Pack k-mesh. To linear
        order, the following expression must be satisfied

        .. math::

            \sum_{s}^{N_{\rm sh}} w_s \sum_{i}^{M_s} b_{\alpha}^{i,s}
            b_{\beta}^{i,s} = \delta_{\alpha,\beta}

        where :math:`N_{\rm sh} \equiv` ``n_shell` is the number of shells
        defining the order of nearest neighbors, :math:`M_s` is the number of
        k-points in the :math:`s`-th shell, and :math:`b_{\alpha}^{i,s}` is the
        :math:`\alpha`-th Cartesian component of :math:`i`-th vector
        connecting k-points to their nearest neighbors in the 
        :math:`s`-th shell.

        Parameters
        ----------
        n_shell : int
            The number of shells to consider.
        report : bool
            Whether to print a report of the k-shells.

        Returns
        -------
        w : np.ndarray
            The finite difference weights.
        k_shell : list[np.ndarray[float]]
            List of :math:`\mathbf{b}` vectors in inverse units of lattice vectors
            connecting nearest neighbor k-mesh points. Length is `n_shell`.
        idx_shell : list[np.ndarray[int]]
            Each entry is an array of integer shifts that takes a k-point 
            index in the mesh to its nearest neighbors.
            Length is `n_shell`.
        """
        from itertools import combinations_with_replacement as comb

        k_shell, idx_shell = self.nn_k_shell(nks, n_shell=n_shell, report=report)
        dim_k = self.dim_k
        cart_idx = list(comb(range(dim_k), 2))
        n_comb = len(cart_idx)

        A = np.zeros((n_comb, n_shell))
        q = np.zeros((n_comb))

        for j, (alpha, beta) in enumerate(cart_idx):
            if alpha == beta:
                q[j] = 1
            for s in range(n_shell):
                b_star = k_shell[s]
                for i in range(b_star.shape[0]):
                    b = b_star[i]
                    A[j, s] += b[alpha] * b[beta]

        U, D, Vt = np.linalg.svd(A, full_matrices=False)
        w = (Vt.T @ np.linalg.inv(np.diag(D)) @ U.T) @ q
        if report:
            print(f"Finite difference weights: {w}")
        return w, k_shell, idx_shell
    

    def visualize(
        self,
        proj_plane=None,
        n_cells=1,
    ):
        r"""Visualizes the lattice geometry.

        Returns
        -------
            fig : matplotlib.figure.Figure
                Figure object from matplotlib.pyplot module
            ax : matplotlib.axes.Axes
                Axes object from matplotlib.pyplot module

        Notes
        -----
        - This function is intended for visualizing two dimensional lattices.
          For three-dimensional visualizations, consider using
          the :func:`visualize_3d` method.

        See Also
        --------
        - :ref:`haldane-edge-nb`,
        - :ref:`visualize-nb`.

        """
        from pythtb.plotting import plot_lattice
        return plot_lattice(self, n_cells=n_cells, proj_plane=proj_plane)

    def visualize_3d(
        self,
        n_cells=1,
        site_colors=None,
        site_names=None,
        show_lattice_info=True,
    ):
        r"""Visualize a 3D tight-binding model using ``Plotly``.

        This function creates an interactive 3D plot of your tight-binding model,
        showing the unit-cell origin, lattice vectors (with arrowheads), orbitals,
        hopping lines, and (optionally) an eigenstate overlay with marker sizes
        proportional to amplitude and colors reflecting the phase.

        Parameters
        ----------
        show_lattice_info: bool, optional
            Whether to display lattice information (lattice vectors, orbital positions).
        site_colors: list of str, optional
            List of colors for each orbital site (e.g. ["red", "blue", "green"]).
        site_names: list of str, optional
            List of names for each orbital site (e.g. ["A", "B", "C"]). 
            If provided, these names will be displayed next to the corresponding orbitals.

        Returns
        -------
        plotly.graph_objs.Figure
        """
        from pythtb.plotting import plot_lattice_3d
        return plot_lattice_3d(
            self,
            n_cells=n_cells,
            show_lattice_info=show_lattice_info,
            site_colors=site_colors,
            site_names=site_names,
        )
    
