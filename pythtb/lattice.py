import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.info("Lattice module loaded.")

__all__ = ["Lattice"]

class Lattice():
    r"""Class for lattice structure.
    
    Parameters
    ----------
    lat_vecs : array_like
        Array of shape (dim_r, dim_r) containing the real-space lattice vectors as rows
        in Cartesian coordinates.
    orb_vecs : array_like, int
        Array of shape (norb, dim_r) containing the orbital positions as rows
        in reduced coordinates (fractions of the lattice vectors). If ``orb_vecs`` 
        is an integer, it specifies the number of orbitals at the origin.
    periodic_dirs : list of bool, optional
        List of length dim_r indicating which lattice directions are periodic (True)
        and which are open (False). If None (default), all directions are considered open.
    
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
        periodic_dirs = None
    ):
        if isinstance(lat_vecs, (list, np.ndarray)):
            lat_vecs = np.array(lat_vecs, dtype=float)
        else:
            raise TypeError("Lattice vectors must be a list or numpy array.")
        
        dim_r = lat_vecs.shape[0]

        if dim_r == 0:
            lat_vecs = np.identity(0, dtype=float)

        if lat_vecs.shape[1] != dim_r:
            raise ValueError(
                "Wrong lat array dimensions. Must have shape (dim_r, dim_r)."
            )
        
        if dim_r > 3:
            raise ValueError("Argument dim_r must be from 0 to 3.")
        if dim_r > 0:
            det_lat = np.linalg.det(lat_vecs)
            if det_lat < 0:
                raise ValueError("Lattice vectors need to form right handed system.")
            elif det_lat < 1e-10:
                raise ValueError("Volume of unit cell is zero.")
        
        self._lat_vectors = lat_vecs
        self._dim_r = dim_r

        if isinstance(orb_vecs, int):
            if orb_vecs < 1:
                raise ValueError("Number of orbitals must be positive.")
            orb_vecs = np.zeros((orb_vecs, dim_r), dtype=float)
        elif isinstance(orb_vecs, (list, np.ndarray)):
            orb_vecs = np.array(orb_vecs, dtype=float)
            if orb_vecs.ndim != 2 or orb_vecs.shape[1] != dim_r:
                raise ValueError(
                    "Wrong orb array dimensions. Must have shape (norb, dim_r)."
                )
        else:
            raise TypeError("Orbital vectors must be an integer, list, or numpy array.")

        self._orb_vectors = orb_vecs
        self._orb_vecs_cart = orb_vecs @ lat_vecs
        self._norb = orb_vecs.shape[0]

        if periodic_dirs is None:
            logger.info("All lattice directions are considered open (non-periodic).")
            periodic_dirs = []
        elif isinstance(periodic_dirs, (list, tuple, np.ndarray)):
            periodic_dirs = list(periodic_dirs)
            if len(periodic_dirs) > dim_r:
                raise ValueError(
                    "Wrong periodic_dirs length. Must be of length <= dim_r."
                )
        else:
            raise TypeError("periodic_dirs must be a list of integers.")

        self._dim_k = len(periodic_dirs) if periodic_dirs is not None else 0
        self._periodic_dirs = periodic_dirs

        # Cell volume
        if self._dim_r == 0:
            self._cell_vol = 0.0
        else:
            lat_vecs = self._lat_vectors
            vol = np.sqrt(np.linalg.det(lat_vecs @ lat_vecs.T))
            self._cell_vol = vol

        # Reciprocal lattice
        self._recip_lat = self._get_recip_lat() if self._dim_k > 0 else None
        if self._dim_k == 0:
            self._recip_vol = 0.0
        else:
            self._recip_vol = np.sqrt(np.linalg.det(self._recip_lat @ self._recip_lat.T))

    @property
    def dim_r(self) -> int:
        """The dimensionality of real space."""
        return self._dim_r

    @property
    def dim_k(self) -> int:
        """The dimensionality of reciprocal space (periodic directions)."""
        return self._dim_k

    @property
    def norb(self) -> int:
        """The number of tight-binding orbitals in the model."""
        return self._norb
    
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
        """Volume of the reciprocal unit cell.

        .. versionadded:: 2.0.0
        """
        return self._recip_vol
    
    @property
    def cell_volume(self) -> float:
        """Volume of the real-space unit cell."""
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
                "float_kind": lambda x: f"{0:^7.0f}" if abs(x) < 1e-10 else f"{x:^7.3f}"
            }
        output.append("Lattice vectors (Cartesian):")
        for i, vec in enumerate(self.lat_vecs):
            # print(f"  # {i} ===> {np.array2string(vec, formatter=formatter, separator=', ')}")
            output.append(
                f"  # {i} ===> {np.array2string(vec, formatter=formatter, separator=', ')}"
            )

        output.append("Orbital vectors (dimensionless):")
        for i, orb in enumerate(self.orb_vecs):
            # print(f"  # {i} ===> {np.array2string(orb, formatter=formatter, separator=', ')}")
            output.append(
                f"  # {i} ===> {np.array2string(orb, formatter=formatter, separator=', ')}"
            )
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