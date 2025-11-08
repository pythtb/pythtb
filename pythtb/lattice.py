import numpy as np
import logging
import copy
from typing import Iterable, Literal
from types import EllipsisType

logger = logging.getLogger(__name__)

__all__ = ["Lattice"]


def _parse_kpts(kpts, dim):
    """
    Parse special string cases for 1D and ensure array shape (n_nodes, dim).
    """
    if isinstance(kpts, str) and dim == 1:
        presets = {
            "full": [[0.0], [0.5], [1.0]],
            "fullc": [[-0.5], [0.0], [0.5]],
            "half": [[0.0], [0.5]],
        }
        return np.array(presets[kpts], float)

    arr = np.array(kpts, float)
    if arr.ndim == 1 and dim == 1:
        arr = arr[:, None]
    return arr


class Lattice:
    r"""Store lattice and orbital information.

    The :class:`Lattice` class encapsulates the real-space lattice vectors, orbital
    positions, and periodicity information for tight-binding models. It provides methods
    to access and manipulate lattice properties, such as retrieving lattice vectors,
    orbital positions, and cutting finite pieces from periodic lattices. Internally,
    it also computes the reciprocal lattice vectors based on the specified periodic directions.

    ..  versionadded:: 2.0.0

    Parameters
    ----------
    lat_vecs : array_like
        Array of shape ``(dim_r, dim_r)`` containing the real-space lattice vectors as rows
        in Cartesian coordinates.
    orb_vecs : array_like, int
        Array of shape ``(norb, dim_r)`` containing the orbital positions as rows
        in reduced coordinates (fractions of the lattice vectors). If ``orb_vecs``
        is an integer, it specifies the number of orbitals at the origin.
    periodic_dirs : array_like of int or {'all'} or Ellipsis, optional
        Real-space lattice directions that treated as periodic. The indices
        refer to the ``lat_vecs`` array, e.g. ``[0]`` would indicate that the first
        lattice vector is periodic. Use ``...`` or ``"all"`` to indicate that all directions
        are periodic. If an empty list (default) or None, all directions are considered
        finite (open boundary conditions).

    Notes
    -----
    - The dimension of the real-space lattice, ``dim_r``, is inferred from the shape of ``lat_vecs``.
    - The dimension of the k-space, ``dim_k``, is inferred from the number of entries in ``periodic_dirs``.
    - The lattice vectors must form a right-handed system with non-zero volume.
    - Orbital positions are given in reduced coordinates, i.e., fractions of the lattice vectors.
    - Works for 0D, 1D, 2D, and 3D lattices. For 0D, use empty arrays for ``lat_vecs`` and an
      integer for ``orb_vecs``.

    """

    def __init__(
        self,
        lat_vecs: np.ndarray,
        orb_vecs: np.ndarray,
        periodic_dirs: Iterable[int] | Literal["all"] | EllipsisType = [],
    ):
        self._periodic_dirs = []  # temporary placeholder for lat_vecs setter
        self.lat_vecs = lat_vecs

        if periodic_dirs in ("all", Ellipsis):
            periodic_dirs = list(range(self.dim_r))
        elif periodic_dirs is None:
            logger.info("All lattice directions are considered open (non-periodic).")
            periodic_dirs = []
        elif isinstance(periodic_dirs, (list, tuple, np.ndarray)):
            periodic_dirs = list(periodic_dirs)
        else:
            raise TypeError(
                "periodic_dirs must be a list of integers, 'all', or Ellipsis."
            )

        self.periodic_dirs = periodic_dirs  # set only after lat_vecs are set
        self.orb_vecs = orb_vecs
        self._nsuper = [1 for _ in range(self.dim_r)]  # default supercell sizes

    def __eq__(self, value):
        if not isinstance(value, Lattice):
            return False
        return (
            np.allclose(self.lat_vecs, value.lat_vecs)
            and np.allclose(self.orb_vecs, value.orb_vecs)
            and self.periodic_dirs == value.periodic_dirs
        )

    def _set_orb_vecs(self, orb_vecs):
        if isinstance(orb_vecs, int):
            if orb_vecs < 0:
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

        if hasattr(self, "_lat_vectors"):
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
            self._recip_vol = np.sqrt(
                np.linalg.det(self._recip_lat @ self._recip_lat.T)
            )

        if hasattr(self, "_orb_vectors"):
            # reframe fractional orbital positions into new lattice
            self.orb_vecs = self.orb_vecs @ np.linalg.inv(lat_vecs)

    @property
    def periodic_dirs(self) -> list[int]:
        """List of periodic directions."""
        return self._periodic_dirs

    @property
    def orb_vecs(self) -> np.ndarray:
        """Orbital vectors in reduced coordinates with shape ``(norb, dim_r)``."""
        return self._orb_vectors.copy()

    @property
    def lat_vecs(self) -> np.ndarray:
        """Lattice vectors in Cartesian coordinates with shape ``(dim_r, dim_r)``."""
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
        if not hasattr(self, "_lat_vectors"):
            raise AttributeError(
                "Lattice vectors must be defined before setting periodic_dirs."
            )

        dim_r = self._lat_vectors.shape[0]
        validated: list[int] = []
        seen = set()
        for idx in new_per:
            if not isinstance(idx, (int, np.integer)):
                raise TypeError("periodic_dirs entries must be integers.")
            val = int(idx)
            if val < 0 or val >= dim_r:
                raise ValueError(
                    f"Periodic direction {val} is out of bounds for lattice dimension {dim_r}."
                )
            if val in seen:
                raise ValueError("periodic_dirs entries must be unique.")
            seen.add(val)
            validated.append(val)

        self._periodic_dirs = validated

        # Update reciprocal lattice since periodic directions may have changed
        if self.dim_k == 0:
            self._recip_lat = None
            self._recip_vol = 0.0
        else:
            self._recip_lat = self._get_recip_lat()
            self._recip_vol = np.sqrt(
                np.linalg.det(self._recip_lat @ self._recip_lat.T)
            )

    # Read-only properties inferred from mutable attributes
    @property
    def nsuper(self) -> list[int]:
        """List of supercell sizes along each real-space lattice vector."""
        return self._nsuper.copy()

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
        return self.orb_vecs.shape[0]

    @property
    def recip_lat_vecs(self) -> np.ndarray:
        """Reciprocal lattice vectors in Cartesian coordinates with shape ``(dim_k, dim_r)``."""
        if self._recip_lat is None:
            raise ValueError(
                "Reciprocal lattice vectors are not defined for zero-dimensional k-space."
            )
        return self._recip_lat.copy()

    @property
    def recip_volume(self) -> float:
        """Volume of the reciprocal unit cell in Cartesian coordinates."""
        return self._recip_vol

    @property
    def cell_volume(self) -> float:
        """Volume of the real-space unit cell in Cartesian coordinates."""
        return self._cell_vol

    def __str__(self) -> str:
        return self.info(show=False)

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

        output.append(
            f"Volume of unit cell (Cartesian) = {self.cell_volume:5.3f} [A^d]\n"
        )

        if self.dim_k > 0:
            output.append("Reciprocal lattice vectors (Cartesian):")
            for i, vec in enumerate(self.recip_lat_vecs):
                output.append(
                    f"  # {i} ===> {np.array2string(vec, formatter=formatter, separator=', ')}"
                )
            output.append(
                f"Volume of reciprocal unit cell = {self.recip_volume:5.3f} [A^-d]\n"
            )

        output.append("Orbital vectors (Cartesian):")
        for i, orb in enumerate(self._orb_vecs_cart):
            output.append(
                f"  # {i} ===> {np.array2string(orb, formatter=formatter, separator=', ')}"
            )

        output.append("")

        output.append("Orbital vectors (fractional):")
        for i, orb in enumerate(self.orb_vecs):
            output.append(
                f"  # {i} ===> {np.array2string(orb, formatter=formatter, separator=', ')}"
            )

        output.append("----------------------------------------")

        return output

    def info(self, show: bool = True) -> str:
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
            logger.warning(
                "Reciprocal lattice vectors are not defined for zero-dimensional k-space."
            )
            return None

        # Select the real-space lattice vectors that generate k-space.
        # Prefer an explicit list (e.g. self.per holds indices of periodic directions).
        # Fallback: take the first dim_k lattice vectors.
        lat = np.asarray(
            self.lat_vecs, dtype=float
        )  # shape (dim_r, dim_r) in Cartesian coords
        per = np.asarray(self.periodic_dirs, dtype=int)
        if per.size != self.dim_k:
            raise ValueError(
                f"'per' must list exactly dim_k={self.dim_k} periodic directions."
            )

        A_sub = lat[per, :]  # (dim_k, dim_r)

        # Fast path: square case -> inverse transpose
        if self.dim_k == self.dim_r:
            # rows of B satisfy A_sub @ B.T = 2π I → B = 2π (A_sub^{-1})^T
            try:
                B = (2.0 * np.pi) * np.linalg.inv(A_sub).T
            except np.linalg.LinAlgError:
                raise ValueError(
                    "Real-space lattice vectors are linearly dependent (singular)."
                )
        else:
            # Rectangular case (dim_k < dim_r): minimum-norm solution
            # Check independence via Cholesky of Gram matrix (SPD iff independent)
            G = A_sub @ A_sub.T  # (dim_k, dim_k)

            try:
                # Cholesky proves PD and is faster than matrix_rank/SVD
                np.linalg.cholesky(G)
            except np.linalg.LinAlgError:
                raise ValueError(
                    "Periodic real-space vectors are not linearly independent; "
                    "cannot construct reciprocal lattice for k-subspace."
                )

            # Solve G X = A_sub -> A_sub @ B.T = 2pi I, with X = G^{-1} A_sub
            X = np.linalg.solve(G, A_sub)  # (dim_k, dim_r)
            # Rows b_i satisfy A_sub @ B^T = 2pi I_{dim_k}
            B = (2.0 * np.pi) * X

        return B

    def cut_piece(self, num_cells, periodic_dir) -> "Lattice":
        r"""Cut a (d-1)-dimensional piece out of a d-dimensional Lattice.

        Constructs a (d-1)-dimensional Lattice out of a
        d-dimensional one by repeating the unit cell a given number of
        times along one of the periodic lattice vectors. The lattice
        vector along which the cut is made is no longer considered periodic,
        but otherwise remains unchanged. Orbitals are added in the new unit cells
        by translating the original orbitals by integer multiples of the lattice vectors.

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
        --------
        :ref:`cubic-slab-hwf-nb` : For an example
        :ref:`three-site-thouless-nb` : For an example

        Notes
        -----
        - Orbitals in `fin_lat` are numbered so that the `i`-th orbital of the `n`-th unit
          cell has index ``i + norb * n`` (here `norb` is the number of orbitals in the original model).
        - The real-space lattice vectors of the returned model are the same as those of
          the original model; only the dimensionality of reciprocal space
          is reduced.
        """
        if self.dim_k == 0:
            raise Exception("Lattice is already finite")
        if not isinstance(num_cells, int):
            raise TypeError("Parameter `num_cells` is not an integer")
        if num_cells < 1:
            raise ValueError("Argument num_cells must be positive!")

        new_per = copy.deepcopy(self.periodic_dirs)
        if periodic_dir not in new_per:
            raise Exception("Can not make model finite along this direction!")
        # remove index which is no longer periodic
        new_per.remove(periodic_dir)

        # generate orbitals of a finite model
        fin_orb = []
        for i in range(num_cells):  # go over all cells in finite direction
            for j in range(self.norb):  # go over all orbitals in one cell
                orb_tmp = np.copy(self.orb_vecs[j, :])
                # change coordinate along finite direction
                orb_tmp[periodic_dir] += float(i)
                fin_orb.append(orb_tmp)
        fin_orb = np.array(fin_orb)

        fin_lat = Lattice(self.lat_vecs, fin_orb, periodic_dirs=new_per)
        fin_lat._nsuper = copy.deepcopy(self.nsuper)
        fin_lat._nsuper[periodic_dir] = num_cells
        return fin_lat

    def add_orb(self, orb_pos):
        r"""Add an orbital to the lattice.

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

        self.orb_vecs = np.vstack([self.orb_vecs, orb_pos])

    def remove_orb(self, to_remove):
        r"""Remove an orbital from the lattice.

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
            self.orb_vecs = np.delete(self.orb_vecs, orb_ind, 0)

    def change_nonperiodic_vector(self, fin_dir: int, new_lat_vec=None):
        r"""Change non-periodic lattice vector.

        Returns  :class:`Lattice` in which one of the non-periodic "lattice vectors"
        is changed. Non-periodic lattice vectors are those that are not listed as periodic
        with the `periodic_dirs` parameter. The returned object has modified reduced coordinates
        of orbitals, consistent with the new choice of lattice vector. Therefore, the actual
        (Cartesian) coordinates of orbitals in original and new :class:`Lattice`
        are the same.

        Parameters
        ----------
        fin_dir : int
            Index of non-periodic lattice vector to change.

        new_lat_vec : array_like, optional
            The new non-periodic lattice vector. If None (default), the new
            non-periodic lattice vector is constructed to be orthogonal to all periodic
            vectors and to have the same length as the original non-periodic vector.

        See Also
        --------
        periodic_dirs : Property listing periodic directions.
        :ref:`boron-nitride-nb` : For an example.
        """
        if not isinstance(fin_dir, int):
            raise TypeError("Argument fin_dir must be an integer")
        if fin_dir in self.periodic_dirs:
            raise ValueError(f"Selected direction {fin_dir} is not nonperiodic")

        if new_lat_vec is None:
            # construct new nonperiodic lattice vector
            per_temp = np.zeros_like(self.lat_vecs)
            for direc in self.periodic_dirs:
                per_temp[direc] = self.lat_vecs[direc]
            # find projection coefficients onto space of periodic vectors
            coeffs = np.linalg.lstsq(per_temp.T, self.lat_vecs[fin_dir], rcond=None)[0]
            projec = np.dot(self.lat_vecs.T, coeffs)
            # subtract off to get new nonperiodic vector
            np_lattice_vec = self.lat_vecs[fin_dir] - projec

            if np.linalg.norm(np_lattice_vec) < 1e-10:
                raise ValueError("""New nonperiodic vector has zero length!?""")

            # normalize new nonperiodic vector to have same length as original
            np_lattice_vec /= np.linalg.norm(np_lattice_vec)
            np_lattice_vec *= np.linalg.norm(self.lat_vecs[fin_dir])

            # check that new nonperiodic vector is perpendicular to all periodic vectors
            for i in self.periodic_dirs:
                if np.abs(np.dot(self.lat_vecs[i], np_lattice_vec)) > 1e-6:
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
        new_lat[fin_dir] = np_lattice_vec

        # Update reduced orb vecs
        new_orb = []
        for orb_cart in og_orb_cart:
            # convert to reduced coordinates
            new_orb.append(np.linalg.solve(new_lat.T, orb_cart))

        # update lattice vectors and orbitals
        self.lat_vecs = np.array(new_lat, dtype=float)
        self.orb_vecs = np.array(new_orb, dtype=float)

        logger.info(
            f"Updated lattice vectors to {new_lat} and orbitals to {new_orb} after changing nonperiodic vector."
        )

        # Are cartesian coordinates of orbitals the same in two cases?
        for idx, orb_cart in enumerate(og_orb_cart):
            cart_old = orb_cart
            cart_new = self._orb_vecs_cart[idx]
            if np.max(np.abs(cart_old - cart_new)) > 1e-6:
                raise ValueError(
                    """This shouldn't happen. New choice of nonperiodic vector
                        somehow changed Cartesian coordinates of orbitals."""
                )

    def _prepare_supercell_geometry(self, sc_red_lat):
        """Validate and build geometric data for a super-cell transformation."""
        if self.dim_r == 0:
            raise ValueError(
                "Must have at least one periodic direction to make a super-cell"
            )

        use_sc_red_lat = np.asarray(sc_red_lat)
        if use_sc_red_lat.shape != (self.dim_r, self.dim_r):
            raise ValueError("Dimension of sc_red_lat array must be dim_r*dim_r")
        if not np.issubdtype(use_sc_red_lat.dtype, np.integer):
            raise TypeError("sc_red_lat array elements must be integers")
        use_sc_red_lat = use_sc_red_lat.astype(int, copy=False)

        per_mask = np.isin(np.arange(self.dim_r), self.periodic_dirs)
        diag = np.diag(use_sc_red_lat)
        if np.any(~per_mask & (diag != 1)):
            raise ValueError(
                "Diagonal elements of sc_red_lat for non-periodic directions must equal 1."
            )

        off_mask = ~(per_mask[:, None] & per_mask[None, :])
        off_diag = use_sc_red_lat - np.diag(diag)
        if np.any(off_mask & (off_diag != 0)):
            raise ValueError(
                "Off-diagonal elements of sc_red_lat for non-periodic directions must equal 0."
            )

        vol = float(np.linalg.det(use_sc_red_lat))
        if abs(vol) < 1e-6:
            raise ValueError("Super-cell lattice vectors volume too close to zero.")
        if vol < 0:
            raise ValueError(
                "Super-cell lattice vectors volume is negative. Must form right-handed system."
            )

        red_transform = np.linalg.inv(use_sc_red_lat.astype(float))
        max_R = int(np.max(np.abs(use_sc_red_lat))) * self.dim_r
        candidate_axis = np.arange(-max_R, max_R + 1, dtype=int)
        if self.dim_r == 1:
            candidates = candidate_axis[:, None]
        else:
            grids = np.meshgrid(*([candidate_axis] * self.dim_r), indexing="ij")
            candidates = np.stack(grids, axis=-1).reshape(-1, self.dim_r)

        reduced = candidates @ red_transform
        eps_shift = np.sqrt(2) * 1e-8
        inside = np.all((-eps_shift < reduced) & (reduced <= 1 - eps_shift), axis=1)
        sc_vec = candidates[inside]

        expected = int(round(abs(vol)))
        if sc_vec.shape[0] != expected:
            raise Exception(
                "Super-cell generation failed! Wrong number of super-cell vectors found."
            )

        sc_cart_lat = use_sc_red_lat @ self.lat_vecs
        orb_grid = sc_vec[:, None, :] + self.orb_vecs[None, :, :]
        sc_orb = (
            (orb_grid.reshape(-1, self.dim_r) @ red_transform)
            if orb_grid.size
            else np.zeros((0, self.dim_r), dtype=float)
        )

        return {
            "lat_vecs": sc_cart_lat,
            "orb_vecs": sc_orb,
            "translations": sc_vec,
            "red_transform": red_transform,
            "sc_red_lat": use_sc_red_lat,
        }

    def make_supercell(
        self,
        sc_red_lat,
        return_sc_vectors: bool = False,
        to_home: bool = True,
        to_home_warning: bool = True,
    ):
        r"""Make lattice a super-cell.

        .. versionchanged:: 2.0.0
            Parameter `to_home_supress_warning` has been renamed to `to_home_warning`.
            Note: this change inverts the meaning of the boolean parameter.

        Constructs a :class:`pythtb.TBModel` representing a super-cell
        of the current object. This function can be used together with :func:`cut_piece`
        in order to create slabs with arbitrary surfaces.

        By default all orbitals will be shifted to the home cell after
        unit cell has been created. That way all orbitals will have
        reduced coordinates between 0 and 1. If you wish to avoid this
        behavior, you need to set, *to_home* argument to *False*.
        """

        geom = self._prepare_supercell_geometry(sc_red_lat)
        self.lat_vecs = geom["lat_vecs"]
        self.orb_vecs = geom["orb_vecs"]
        self._nsuper = list(np.diag(geom["sc_red_lat"]))

        if to_home:
            self._shift_orb_to_home(to_home_warning=to_home_warning)

        if return_sc_vectors:
            return geom["translations"].copy()

    def _shift_orb_to_home(self, to_home_warning: bool = True):
        r"""Shifts orbital coordinates (along periodic directions) to the home
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
                elif (
                    k not in self.periodic_dirs and shift != 0 and to_home_warning
                ):  # check for shift in non-periodic directions
                    logger.warning(
                        f"Orbital {i} has reduced coordinate {self.orb_vecs[i, k]:.3f} "
                        f"along non-periodic direction {k}, which is outside the home cell."
                    )

            orb_vecs_new[i] -= disp_vec

        self.orb_vecs = orb_vecs_new

    def nn_bonds(self, n_shell: int, report: bool = False):
        r"""Enumerate nearest-neighbor shells of the lattice.

        The lattice's orbitals are treated as points in real space (Cartesian
        coordinates). We form all displacement vectors connecting each orbital
        to every other orbital translated by integer lattice vectors and group
        them into ``n_shell`` distinct radial shells.

        Parameters
        ----------
        n_shell : int
            Number of shells to return (starting from the shortest non-zero
            displacement).
        report : bool, optional
            If True, print a human-readable table of the shells and their
            degeneracies. Default is ``False``.

        Returns
        -------
        dict
            Dictionary with the following keys:

            ``shells``
                List of length ``n_shell``. Each entry is a dictionary with
                the keys ``shell`` (1-based shell index), ``radius`` (float),
                ``distance_sq`` (float), ``degeneracy_total`` (int), and
                ``orbitals`` (list of per-orbital neighbour data). Each
                orbital entry contains the indices of the bonded orbital,
                the lattice translation and the multiplicity.

            ``displacements``
                List of ``n_shell`` entries. Each entry contains, for every
                orbital in the home cell, an array of Cartesian displacement
                vectors :math:`\Delta\mathbf{r}` pointing to the neighbours in
                that shell.

            ``indices``
                Same structure as ``displacements`` but holding integer
                metadata of the form ``[i, j, R_1, ..., R_{dim_r}]`` where ``i``
                and ``j`` are orb indices and ``R`` is the lattice translation.

            ``radii``
                One-dimensional :class:`numpy.ndarray` of shell radii in
                Cartesian units.

        Raises
        ------
        ValueError
            If ``n_shell`` is not a positive integer, or if the lattice has no
            orbitals or zero real-space dimension.
        """
        if not isinstance(n_shell, int) or n_shell < 1:
            raise ValueError("Invalid n_shell: must be an integer > 1.")
        n_shell = int(n_shell)

        if self.norb == 0:
            raise ValueError("No orbitals in the lattice.")
        if self.dim_r == 0:
            raise ValueError("Lattice dimension is zero.")

        lat_vecs = self.lat_vecs
        dim_r = self.dim_r
        orb_cart = self._orb_vecs_cart
        norb = self.norb

        # Enumerate candidate neighbors up to a reasonable window of lattice shifts
        # R in Z^{dim_r} with components in [-n_shell, n_shell]
        from itertools import product as _product

        d2_list = []  # squared distances
        dr_list = []  # Cartesian displacement vectors delta_r
        idx_list = []  # integer meta: [i, j, [R]]

        periodic_dirs = set(self.periodic_dirs)
        shift_ranges = [
            range(-n_shell - 1, n_shell + 2) if k in periodic_dirs else (0,)
            for k in range(dim_r)
        ]
        shifts = np.array(list(_product(*shift_ranges)), dtype=int)

        for i in range(norb):
            ri = orb_cart[i]  # Cartesian position of orbital i
            for R in shifts:
                T_R = R @ lat_vecs  # (dim_r,) Cartesian translation vector
                for j in range(norb):
                    if i == j and not np.any(R):
                        # Skip self-pair with zero shift
                        continue

                    rj = orb_cart[j] + T_R  # Cartesian position of orbital j
                    dr = rj - ri  # delta_r (Cartesian)
                    d2 = float(dr @ dr)  # squared distance

                    d2_list.append(d2)
                    dr_list.append(dr)
                    idx_list.append(np.concatenate(([i, j], R)))

        if not d2_list:
            # No neighbors (e.g., single-orbital 0D with no shifts requested)
            return {
                "shells": [],
                "displacements": [],
                "indices": [],
                "radii": np.empty(0, dtype=float),
            }

        # Convert to arrays
        dr_arr = np.vstack(dr_list)  # (N, dim_r)
        idx_arr = np.vstack(idx_list).astype(int)  # (N, 2+dim_r)
        d2_arr = np.asarray(d2_list)

        # Numerical stability: round squared norms to cluster nearly-equal distances
        d2_rounded = np.round(d2_arr, 12)

        # Sort by distance
        order = np.argsort(d2_rounded)
        d2_sorted = d2_rounded[order]
        dr_sorted = dr_arr[order]
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
        nn_shell = []  # list over shells -> list over orbitals -> (deg_i, dim_r)
        idx_shell = []  # list over shells -> list over orbitals -> (deg_i, 2+dim_r)

        shell_summaries = []
        shell_radii = []
        for s, d2_target in enumerate(unique_d2):
            mask_s = d2_sorted == d2_target
            dr_s = dr_sorted[mask_s]
            idx_s = idx_sorted[mask_s]

            # Split by central-orbital index i (idx column 0)
            shell_R_by_i = []
            shell_idx_by_i = []
            per_orbital = {}
            for i in range(norb):
                m_i = idx_s[:, 0] == i
                dr_si = dr_s[m_i]
                idx_si = idx_s[m_i]

                shell_R_by_i.append(dr_si)
                shell_idx_by_i.append(idx_si)

                if dr_si.size == 0:
                    continue

                per_orbital[int(i)] = {
                    "degeneracy": int(dr_si.shape[0]),
                    "neighbors": idx_si[:, 1].astype(int).tolist(),
                    "shifts": idx_si[:, 2:].astype(int).tolist(),
                    "displacements": dr_si.tolist(),
                }

            nn_shell.append(shell_R_by_i)
            idx_shell.append(shell_idx_by_i)

            radius = float(np.sqrt(d2_target))
            total_deg = int(sum(block.shape[0] for block in shell_R_by_i))
            shell_summaries.append(
                {
                    "shell": s + 1,
                    "radius": radius,
                    "degeneracy_total": total_deg,
                    "orbitals": per_orbital,
                }
            )
            shell_radii.append(radius)

        # Optionally print a compact text report
        if report:
            lines = []
            lines.append("nn-shell report (per-orbital)")
            lines.append("═" * 60)
            lines.append(f"dim_r: {dim_r}   norb: {norb}   shells: {len(unique_d2)}")
            for s, d2_target in enumerate(unique_d2, start=1):
                radius = np.sqrt(d2_target)
                total_deg = sum(Rs.shape[0] for Rs in nn_shell[s - 1])
                lines.append(
                    f"shell {s:>2}: |Δr|={radius:.8g} (degeneracy total={total_deg})"
                )
                # Show first few for each i
                for i in range(norb):
                    Rs = nn_shell[s - 1][i]
                    Id = idx_shell[s - 1][i]
                    if Rs.size == 0:
                        continue
                    head = min(Rs.shape[0], 6)
                    lines.append(f"  i={i}: {Rs.shape[0]} neighbors")
                    for k in range(head):
                        j = int(Id[k, 1])
                        Rvec = Id[k, 2 : 2 + dim_r]
                        dr_str = np.array2string(
                            Rs[k],
                            precision=6,
                            floatmode="maxprec_equal",
                            suppress_small=True,
                        )
                        R_str = np.array2string(
                            Rvec,
                            formatter={"int": lambda x: f"{int(x):>2}"},
                            separator=", ",
                        )
                        lines.append(f"     -> j={j:>2}, R={R_str}   Δr={dr_str}")
                    if Rs.shape[0] > head:
                        lines.append(f"     ... (+{Rs.shape[0] - head} more)")
            print("\n".join(lines))

        S = len(nn_shell)
        norb = self.norb
        dim_r = self.dim_r

        radii = np.asarray(shell_radii, dtype=float)

        if radii.shape[0] != S:
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
        bonds_by_shell = []

        for s in range(S):
            # canonical unique set of bonds (i, j, R) per shell
            seen = set()
            bonds = []
            for i in range(norb):
                Id = idx_shell[s][i]  # shape (deg_i, 2+dim_r): [i, j, R...]
                # [j, R...] without the leading i
                pairs = Id[:, 1 : 2 + dim_r].astype(int, copy=False)

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
            bonds_by_shell.append(bonds)

        return shell_summaries, bonds_by_shell

    def nn_k_shell(self, nks: tuple, n_shell: int, report: bool = False):
        r"""Generates shells of k-points around the :math:`\Gamma` point.

        .. versionadded:: 2.0.0

        Returns array of vectors connecting the origin to nearest
        neighboring k-points in the mesh. The functions works only
        for full k-meshes, i.e., when the number of k-points is specified
        along all periodic directions.

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
        if not isinstance(nks, (list, tuple, np.ndarray)):
            raise TypeError("nks must be a sequence of ints with length dim_k.")
        nks = np.asarray(nks, dtype=int)
        if nks.ndim != 1 or nks.size != self.dim_k:
            raise ValueError(f"nks must have length dim_k={self.dim_k}.")
        if np.any(nks <= 0):
            raise ValueError("All mesh sizes in nks must be positive.")
        if not isinstance(n_shell, int) or n_shell < 1:
            raise ValueError("n_shell must be a positive integer.")
        if self.dim_k == 0:
            raise ValueError("k-shells are not defined when dim_k == 0.")

        # Reciprocal primitive vectors for periodic dirs: rows (dim_k) embed in R^{dim_r}
        B = self.recip_lat_vecs

        def shift_to_dk_cart(s):
            # s: (dim_k, ), delta_k = sum (s_mu /nks_mu) b_mu
            coeffs = s / nks.astype(float)  # (dim_k,)
            return coeffs @ B  # (dim_r,)

        shells = []
        idx_shell = []
        seen_radii = []
        tol = 1e-12
        rmax = 1

        while len(shells) < n_shell:
            # enumerate integer shifts in the hypercube [-rmax, rmax]^dim_k excluding zero
            from itertools import product

            ranges = [range(-rmax, rmax + 1) for _ in range(self.dim_k)]
            cand_shifts = np.array(
                [s for s in product(*ranges) if any(si != 0 for si in s)], dtype=int
            )
            if cand_shifts.size == 0:
                rmax += 1
                continue

            dk_cart = np.array(
                [shift_to_dk_cart(s) for s in cand_shifts]
            )  # (Ncand, dim_r)
            dists = np.linalg.norm(dk_cart, axis=1)

            order = np.argsort(dists)
            cand_shifts = cand_shifts[order]
            dk_cart = dk_cart[order]
            dists = dists[order]

            i = 0
            while i < len(dists) and len(shells) < n_shell:
                r0 = dists[i]
                # skip if this radius already captured
                if any(abs(r0 - rr) <= tol * max(1.0, rr) for rr in seen_radii):
                    j = i + 1
                    while j < len(dists) and abs(dists[j] - r0) <= tol * max(1.0, r0):
                        j += 1
                    i = j
                    continue
                # collect current shell
                j = i + 1
                while j < len(dists) and abs(dists[j] - r0) <= tol * max(1.0, r0):
                    j += 1
                shells.append(dk_cart[i:j].copy())
                idx_shell.append(cand_shifts[i:j].copy())
                seen_radii.append(r0)
                i = j

            rmax += 1

        if report:
            print("k-shell summary:")
            for s, (vecs, shifts, rrep) in enumerate(
                zip(shells, idx_shell, seen_radii)
            ):
                print(f"  shell {s}: M_s={len(vecs)}, |Δk|≈{rrep:.6e} (first)")

        return shells, idx_shell

    def k_shell_weights(
        self,
        nks: tuple,
        n_shell: int = 1,
        return_shell: bool = True,
        report: bool = False,
    ):
        r"""Generates the finite difference weights on a k-shell.

        This function uses the k-shells generated by :func:`nn_k_shell`
        to compute the  weights for finite difference approximations of
        :math:`\nabla_{\mathbf{k}}` on a Monkhorst-Pack k-mesh. To linear
        order, the following expression must be satisfied

        .. math::

            \sum_{s}^{N_{\rm sh}} w_s \sum_{i}^{M_s} b_{\alpha}^{i,s}
            b_{\beta}^{i,s} = \delta_{\alpha,\beta}

        where :math:`N_{\rm sh} \equiv` ``n_shell`` is the number of shells
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
        k_shell : list[np.ndarray[float]], optional
            List of :math:`\mathbf{b}` vectors in inverse units of lattice vectors
            connecting nearest neighbor k-mesh points. Length is `n_shell`.
        idx_shell : list[np.ndarray[int]], optional
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
        if return_shell:
            return w, k_shell, idx_shell
        return w

    def k_path(self, k_nodes, nk: int, report: bool = False):
        r"""Interpolates a path in reciprocal space.

        Interpolates a path in reciprocal space between specified
        k-points. In 2D or 3D the k-path can consist of several
        straight segments connecting high-symmetry points ("nodes").
        The interpolated path is constructed so that the k-points
        are (nearly) equidistant in Cartesian k-space.

        Parameters
        ----------
        k_nodes : array-like, str
          Array of k-vectors in reduced units, between
          which the interpolated path will be constructed.
          In 1D k-space, value may be a string:

          - `"full"`: Implies  ``[0.0, 0.5, 1.0]`` (full BZ)
          - `"fullc"`: Implies  ``[-0.5, 0.0, 0.5]`` (full BZ, centered)
          - `"half"`: Implies  ``[ 0.0, 0.5]``  (half BZ)

        nk : int
            Total number of k-points along the path.
        report : bool, optional
            Optional parameter specifying whether printout
            is desired (default is False).

        Returns
        -------
        k_vec : np.ndarray
            Array of (nearly) equidistant interpolated k-points. Shape is ``(nk, dim_k)``.
        k_dist : np.ndarray
            Array giving accumulated k-distance to each
            k-point in the path. This array is useful when plotting
            bands along the path. The distances between the points
            can be used to ensure that the plot accurately reflects
            the k-space geometry. Shape is ``(nk,)``.

            .. versionchanged:: 2.0.0
                The returned ``k_dist`` now includes the factors of :math:`2\pi`.
                See notes for further details.

        k_node : np.ndarray
            Array giving accumulated k-distance to each
            node on the path in Cartesian coordinates. This array can
            be used to reference the nodes on the 1D ``k_dist`` array,
            e.g., when plotting high-symmetry points. Shape is ``(n_nodes,)``.

        Notes
        -----
        - The distance between the points is calculated in the Cartesian frame,
          however coordinates themselves are given in dimensionless reduced coordinates!
          This is done so that this array can be directly passed to function
          :func:`pythtb.TBModel.solve_ham`.
        - Unlike array ``k_vec``, ``k_dist`` has dimensions! Units are defined
          so that for a one-dimensional crystal with lattice constant equal to
          for example :math:`10` the length of the Brillouin zone would equal
          :math:`2\pi/10`.

        Examples
        --------
        Construct a path connecting four nodal points in k-space
        Path will contain 401 k-points, roughly equally spaced

        >>> path = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.5], [0.0, 0.0]]
        >>> (k_vec, k_dist, k_node) = my_model.k_path(path,401)

        Solve for eigenvalues on that path and plot the band structure

        >>> import matplotlib.pyplot as plt
        >>> evals = tb.solve_ham(k_vec)
        >>> for n in range(evals.shape[1]):
        ...     plt.plot(k_dist, evals[:, n], 'b-')
        >>> for node_dist in k_node:
        ...     plt.axvline(x=node_dist, color='k', linestyle='--')
        >>> plt.xlabel('k')
        >>> plt.ylabel('Energy (eV)')
        >>> plt.show()
        """
        # Parse kpts and validate
        k_list = _parse_kpts(k_nodes, self.dim_k)
        if k_list.shape[1] != self.dim_k:
            raise ValueError(
                f"Dimension mismatch: kpts shape {k_list.shape}, model dim {self.dim_k}"
            )
        if nk < len(k_list):
            raise ValueError("nk must be >= number of nodes in kpts")

        # Extract periodic lattice and compute k-space metric
        lat_per = self.lat_vecs[self.periodic_dirs]
        B = self.recip_lat_vecs
        k_metric = B @ B.T

        # Compute segment vectors and lengths in Cartesian metric
        diffs = k_list[1:] - k_list[:-1]
        seg_lengths = np.sqrt(np.einsum("ij,ij->i", diffs @ k_metric, diffs))

        # Accumulated node distances
        k_node = np.concatenate(([0.0], np.cumsum(seg_lengths)))

        # Determine indices in the final array corresponding to each node
        node_index = np.rint(k_node / k_node[-1] * (nk - 1)).astype(int)

        # Initialize output arrays
        k_vec = np.empty((nk, self.dim_k))
        k_dist = np.empty(nk)

        # Interpolate each segment
        for i, (start, end) in enumerate(zip(node_index[:-1], node_index[1:])):
            length = end - start
            t = np.linspace(0, 1, length + 1)
            k_vec[start : end + 1] = k_list[i] + np.outer(t, diffs[i])
            k_dist[start : end + 1] = k_node[i] + t * seg_lengths[i]

        # Trim any round-off overshoot
        k_vec = k_vec[:nk]
        k_dist = k_dist[:nk]

        if report:
            print("----- k_path report -----")
            np.set_printoptions(precision=5)
            print("Real-space lattice vectors:\n", lat_per)
            print("K-space metric tensor:\n", k_metric)
            print("Nodes (reduced coords):\n", k_list)
            if lat_per.shape[0] == lat_per.shape[1]:
                gvecs = np.linalg.inv(lat_per).T
                print("Reciprocal-space vectors:\n", gvecs)
                print("Nodes (Cartesian coords):\n", k_list @ gvecs)

            print("Segments:")
            for n in range(1, len(k_node)):
                length = k_node[n] - k_node[n - 1]
                print(
                    f"  Node {n - 1} {k_list[n - 1]} to Node {n} {k_list[n]}: "
                    f"distance = {length:.5f}"
                )

            print("Node distances (cumulative):", k_node)
            print("Node indices in path:", node_index)
            print("-------------------------")

        return k_vec, k_dist, k_node

    @staticmethod
    def k_uniform_mesh(
        mesh_size, gamma_centered: bool = False, include_endpoints: bool = True
    ):
        r"""
        Generate a uniform grid of k-points in reduced (fractional) coordinates.

        The grid spans the interval :math:`[0,1)` along each periodic crystal direction
        and always contains the origin. The total number of k-points is the product of
        the entries in ``mesh_size``.

        Parameters
        ----------
        mesh_size : array_like
            The number of k-points along each periodic direction.
            Its length must equal the number of periodic dimensions of the model.
            For example, ``mesh_size = [n1, n2, n3]`` produces a 3D mesh with
            ``n1 x n2 x n3`` points.

        gamma_centered : bool, optional
            If ``True``, the mesh is centered around the Gamma point,
            spanning the interval :math:`[-0.5, 0.5)` along each periodic direction.
            Default is ``False``.

            .. versionadded:: 2.0.0

        include_endpoints : bool, optional
            If ``True``, the mesh includes the endpoint at 1.0 (or 0.5 if
            ``gamma_centered=True``) along each periodic direction.
            Default is ``True``.

            .. versionadded:: 2.0.0

        Returns
        -------
        k_points : numpy.ndarray
            An array of shape ``(n1, n2, ..., dim_k)`` where the final index runs
            over the reduced-coordinate components of each k-vector.

            - If ``mesh_size = [n1, n2, ..., nD]`` and the model has ``dim_k = D``,
              then the returned array has shape ``(n1, n2, ..., nD, D)``.

        Notes
        -----
        This uniform mesh is suitable for evaluating model quantities on a regular
        Brillouin-zone sampling grid, e.g. via :func:`~pythtb.TBModel.solve_ham`.

        Examples
        --------
        Construct a 10x20x30 mesh for a model with three periodic directions:

        >>> k_points = my_model.k_uniform_mesh([10, 20, 30])
        >>> k_points.shape
        (10, 20, 30, 3)

        Solve the model on the mesh:

        >>> evals = my_model.solve_ham(k_points)

        """
        use_mesh = np.array(list(map(round, mesh_size)), dtype=int)
        if np.min(use_mesh) <= 0:
            raise ValueError("Mesh must have positive non-zero number of elements.")

        if gamma_centered:
            start, stop = -0.5, 0.5
        else:
            start, stop = 0.0, 1.0

        axes = [
            np.linspace(start, stop, n, endpoint=include_endpoints) for n in use_mesh
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        k_points = np.stack(mesh, axis=-1).reshape(-1, len(use_mesh))

        return k_points

    def get_kpath_distance(
        self, kpts, k_nodes=None, labels=None, cartesian=False, tol=1e-8
    ):
        """Transform k-points to k-distance along a path.

        Parameters
        ----------
        kpts : array_like
            Array of k-points in reduced coordinates.
        k_nodes : array_like, optional
            Array of special k-point nodes along the path.
        labels : list of str, optional
            Labels corresponding to the k_nodes.
        tol : float, optional
            Tolerance for matching k-points, by default 1e-8

        Returns
        -------
        k_dist : array
            Cumulative k-point distances along the path.
        node_dist : array, optional
            Distances of the special k-point nodes.
        node_indices : dict, optional
            Indices of the special k-point nodes.
        label_indices : dict, optional
            Mapping from labels to k-point nodes.
        """
        B = self.recip_lat_vecs

        if cartesian:
            kpts = kpts @ np.linalg.inv(B).T

        # cumulative distances between points
        _, k_dist, _ = self.k_path(kpts, nk=kpts.shape[0], report=False)
        # k_dist = kpath_distance(kpts, b1=B[0], b2=B[1], b3=B[2])

        if k_nodes is not None:
            node_indices = {}
            for i, node in enumerate(k_nodes):
                mask = np.all(np.isclose(kpts, node, atol=tol), axis=1)
                idx = np.where(mask)[0]

                if len(idx) > 0:
                    node_indices[f"{str(node)}"] = list(idx)
                else:
                    node_indices[f"{str(node)}"] = None

            # flatten all values into a single list
            all_indices = [idx for indices in node_indices.values() for idx in indices]

            # sort them
            all_indices_sorted = np.sort(all_indices)

            # now you can index into k_dist
            node_dist = k_dist[all_indices_sorted]

            if labels is not None:
                label_indices = {}
                for i, node in enumerate(k_nodes):
                    label_indices[labels[i]] = node
                return k_dist, node_dist, node_indices, label_indices
            else:
                return k_dist, node_dist, node_indices

        else:
            return k_dist

    def visualize(
        self,
        proj_plane=None,
        n_cells=1,
    ):
        r"""Visualizes the lattice geometry.

        This function creates a 2D plot of your tight-binding model,
        showing the unit-cell origin, lattice vectors (with arrowheads),
        and orbitals.

        Parameters
        ----------
        proj_plane : list of int, optional
            List of two integers specifying the directions to project onto.
            For example, in a 3D model, ``proj_plane=[0, 2]`` would project
            onto the x-z plane. If not provided, the function will attempt
            to automatically select a suitable projection plane based on
            the model's dimensionality.
        n_cells : int, optional
            Number of unit cells to display along each lattice vector.
            Default is 1.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes

        Notes
        -----
        - This function is intended for visualizing two dimensional lattices.
          For three-dimensional visualizations, consider using
          the :func:`visualize_3d` method.

        """
        from pythtb.visualization import plot_lattice

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
        n_cells: int, optional
            Number of unit cells to display along each lattice vector.
            Default is 1.
        site_colors: list of str, optional
            List of colors for each orbital site (e.g. ``["red", "blue", "green"]``).
            Must abide by Plotly color specifications. If not provided, default colors will be used.
        site_names: list of str, optional
            List of names for each orbital site (e.g. ``["A", "B", "C"]``).
            If provided, these names will be displayed next to the corresponding orbitals.
        show_lattice_info: bool, optional
            Whether to display lattice information (lattice vectors, orbital positions).

        Returns
        -------
        plotly.graph_objs.Figure
        """
        from pythtb.visualization import plot_lattice_3d

        return plot_lattice_3d(
            self,
            n_cells=n_cells,
            show_lattice_info=show_lattice_info,
            site_colors=site_colors,
            site_names=site_names,
        )
