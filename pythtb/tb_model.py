import numpy as np
import copy
import logging
from itertools import product
import warnings
import functools
from itertools import combinations_with_replacement as comb
from .plotting import plot_bands, plot_tb_model, plot_tb_model_3d
from .utils import _is_int, _offdiag_approximation_warning_and_stop, is_Hermitian

# set up logging
logger = logging.getLogger(__name__)

__all__ = ["TBModel"]

SIGMA0 = np.array([[1, 0], [0, 1]], dtype=complex)
SIGMAX = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMAY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMAZ = np.array([[1, 0], [0, -1]], dtype=complex)


def deprecated(message: str, category=FutureWarning):
    """
    Decorator to mark a function as deprecated.
    Raises a FutureWarning with the given message when the function is called.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__qualname__} is deprecated and will be removed in a future release: {message}",
                category=category,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _report(k_list, lat_per, k_metric, k_node, node_index):
    """
    Print a concise report of the k-path construction, including
    segment distances and the start/end node coordinates.
    """
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
            f"  Node {n-1} {k_list[n-1]} to Node {n} {k_list[n]}: "
            f"distance = {length:.5f}"
        )

    print("Node distances (cumulative):", k_node)
    print("Node indices in path:", node_index)
    print("-------------------------")

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


class TBModel:
    r"""Tight-binding model constructor.

    This class contains the tight-binding model information. 
    It is designed to handle various aspects of tight-binding models, including the lattice structure, 
    orbital positions, and periodic boundary conditions. It will also provide methods for constructing
    the Hamiltonian matrix and diagonalizing the tight-binding model.

    Parameters
    ----------
    dim_k : int
        Dimensionality of reciprocal space, i.e., specifies how
        many directions are considered to be periodic.

    dim_r : int
        Dimensionality of real space, i.e., specifies how many
        real space lattice vectors there are and how many coordinates are
        needed to specify the orbital coordinates.

    lat : array_like, optional
        Array containing lattice vectors in Cartesian coordinates
        (in arbitrary units). By default, lattice vectors are an identity matrix. 

    orb : int, array_like, optional
        Array containing reduced coordinates of all
        tight-binding orbitals. If ``orb`` is an integer code will assume
        that there are these many orbitals all at the origin of the unit cell.  
        By default ``orb=1`` and the code will assume a single orbital at the origin.

    per : array_like, optional
        Specifies the indices of lattice vectors which are considered to be periodic.
        By default, all lattice vectors are assumed to be periodic. If ``dim_k`` is smaller than ``dim_r``, 
        then by default the first ``dim_k`` vectors are considered to be periodic.

        In the example below, only the vector ``[0.0,2.0]`` is considered to be periodic 
        (since ``per=[1]``). 

    nspin : {1, 2}, optional
        Number of explicit spin components assumed for each
        orbital in ``orb``. Allowed values of ``nspin`` are ``1`` and ``2``. If
        ``nspin=1`` then the model is spinless, if ``nspin=2`` then it
        is explicitly a spinfull model and each orbital is assumed to
        have two spin components. Default value of this parameter is
        ``nspin=1``.

    Notes
    -----
    Parameter ``dim_r`` can be larger than ``dim_k``! For example,
    a polymer is a three-dimensional molecule (one needs three
    coordinates to specify orbital positions), but it is periodic
    along only one direction. For a polymer, therefore, we should
    have ``dim_k=1`` and ``dim_r=3``. See the :ref:`trestle-nb`
    or :ref:`buckled-layer-nb` examples.

    Examples
    --------
    Creates model that is two-dimensional in real space but only
    one-dimensional in reciprocal space. The first lattice vector has coordinates
    ``[1, 1/2]`` while the second  one has coordinates ``[0, 2]``.
    The second lattice vector is chosen to be periodic (since ``per=[1]``).
    Three orbital coordinates are specified in reduced units. The first orbital
    is defined with reduced coordinates ``[0.2, 0.3]``. Its Cartesian coordinates
    are therefore 0.2 times the first lattice vector plus 0.3 times the second lattice 
    vector.

    >>> from pythtb import TBModel
    >>> tb = TBModel(
    ...        dim_k=1, dim_r=2,
    ...        lat=[[1, 1/2], [0, 2]],
    ...        orb=[[0.2, 0.3], [0.1, 0.1], [0.2, 0.2]],
    ...        per=[1]
    ...    )
    >>> print(tb)

    """

    def __init__(
        self, dim_k: int, dim_r: int, lat=None, orb=1, per=None, nspin: int = 1
    ):

        # Dimensionality of real space
        if not isinstance(dim_r, int):
            raise TypeError("Argument dim_r must be an integer")
        if dim_r > 4:
            raise ValueError("Argument dim_r must be less than 4.")

        # Dimensionality of k-space
        if not isinstance(dim_k, int):
            raise TypeError("Argument dim_k must be an integer.")
        if dim_k > dim_r:
            raise ValueError("Argument dim_k must be less than dim_r.")

        self._dim_r = dim_r
        self._dim_k = dim_k

        # initialize lattice vectors
        # shape: (dim_r, dim_r)
        # idx: (lattice direction, cartesian components)
        # default: None implies unit matrix
        if lat is None:
            self._lat = np.identity(dim_r, float)
            logger.info("Lattice vectors not specified. Using identity matrix.")
        elif isinstance(lat, (list, np.ndarray)):
            lat = np.array(lat, dtype=float)
            if lat.shape != (dim_r, dim_r):
                raise ValueError(
                    "Wrong lat array dimensions. Must have shape (dim_r, dim_r)."
                )
            self._lat = lat
        else:
            raise TypeError("Lattice vectors must be a list or numpy array.")

        # check that volume is not zero and that have right handed system
        if dim_r > 0:
            det_lat = np.linalg.det(self._lat)
            if det_lat < 0:
                raise ValueError("Lattice vectors need to form right handed system.")
            elif det_lat < 1e-10:
                raise ValueError("Volume of unit cell is zero.")

        # Initialize orbitals defined in reduced coordinates
        # shape: (norb, dim_r)
        # idx: (orbital, reduced components)
        # default: 1
        if isinstance(orb, (int, np.integer)):
            self._norb = orb
            self._orb = np.zeros((orb, dim_r))
            logger.info(
                f"Orbital positions is an integer. Assuming {orb} orbitals at the origin"
            )
        elif isinstance(orb, (list, np.ndarray)):
            orb = np.array(orb, dtype=float)
            if orb.ndim != 2:
                raise ValueError(
                    "Orbtial array must have two axes; the first for orbital, the second for reduced unit values."
                )
            if orb.shape[1] != dim_r:
                raise ValueError(
                    "Number of components along second axes of orbital array must match real space dimension."
                )
            self._orb = orb  # orbital vectors
            self._norb = orb.shape[0]  # number of orbitals
        else:
            raise TypeError(
                "Orbital vectors must be array-type or an integer."
            )

        # Specifying which dimensions are periodic.
        if per is None:
            logger.info(
                "Periodic directions not specified. Using the first dim_k directions."
            )
            self._per = list(range(self._dim_k))
        else:
            per = list(per)
            if len(per) != self._dim_k:
                raise ValueError(
                    "Number of periodic directions must equal the k-space dimension, dim_k."
                )
            self._per = per

        # Validate number of spin components
        if nspin not in [1, 2]:
            raise ValueError("nspin must be 1 or 2")
        self._nspin = nspin

        # Number of electronic states at each k-point
        self._nstate = self._norb * self._nspin

        # By default, assume model did not come from w90 object and that
        # position operator is diagonal
        self._assume_position_operator_diagonal = True

        # Initialize onsite energies to zero
        if self._nspin == 1:
            self._site_energies = np.zeros((self._norb), dtype=float)
        elif self._nspin == 2:
            self._site_energies = np.zeros((self._norb, 2, 2), dtype=complex)

        # The onsite energies and hoppings are not specified
        # when creating a 'TBModel' object.  They are speficied
        # subsequently by separate function calls defined below.

        # remember which onsite energies user has specified
        self._site_energies_specified = np.zeros(self._norb, dtype=bool)
        self._site_energies_specified[:] = False

        # Initialize hoppings to empty list
        self._hoppings = []

    def __repr__(self):
        r"""Return a string representation of the ``TBModel`` object.

        Returns
        -------
        str
            String representation of the TBModel.
        """
        return (
            f"pythtb.TBModel(dim_r={self._dim_r}, dim_k={self._dim_k}, "
            f"norb={self._norb}, nspin={self._nspin})"
        )

    def __str__(self):
        r"""Return a string representation of the ``TBModel`` object.

        Returns
        -------
        str
            String representation of the TBModel.
        """
        return self.report(show=False)

    def __eq__(self, other):
        r"""Compare two TBModel objects for equality.

        Compares structural parameters, arrays, and hoppings.

        Parameters
        ----------
        other : TBModel
            Another TBModel instance to compare.

        Returns
        -------
        bool
            True if the models are equal, False otherwise.
        """
        if not isinstance(other, TBModel):
            return NotImplemented
        # Compare simple attributes
        if (
            self._dim_r != other._dim_r
            or self._dim_k != other._dim_k
            or self._nspin != other._nspin
            or self._norb != other._norb
            or self._per != other._per
        ):
            return False
        # Compare numpy arrays
        if not np.allclose(self._lat, other._lat):
            return False
        if not np.allclose(self._orb, other._orb):
            return False
        if not np.allclose(self._site_energies, other._site_energies):
            return False
        # Compare hoppings list
        if len(self._hoppings) != len(other._hoppings):
            return False
        for h1, h2 in zip(self._hoppings, other._hoppings):
            amp1, i1, j1, *R1 = h1
            amp2, i2, j2, *R2 = h2
            if i1 != i2 or j1 != j2:
                return False
            if not np.allclose(amp1, amp2):
                return False
            if R1 and R2:
                if not np.array_equal(R1[0], R2[0]):
                    return False
            elif R1 or R2:
                return False
        return True

    @deprecated(
        "The 'display' method is deprecated and will be removed in a future release. Use 'print(model)' or 'model.report(show=True)' instead."
    )
    def display(self):
        r"""
        .. deprecated:: 2.0.0
            `display` has been deprecated, it is recommended to use `print(model)` or `model.report(show=True)` instead.
        """
        return self.report(show=True)

    def report(self, show: bool = True, short: bool = False):
        r"""Print or return a report about the tight-binding model.

        .. versionadded:: 2.0.0
            The `short` parameter was added to control the verbosity of the report.
            The `show` parameter was added to control whether to print the report or return it as a string.

        Parameters
        ----------
        show : bool, optional
            If True, prints the report to stdout. If False, returns the report as a string.
        short : bool, optional
            If True, print only a short summary. If False, print full details.

        Returns
        -------
        str or None
            Returns the report string if `show` is False, otherwise prints and returns None.

        Notes
        -----
        The report includes lattice vectors, orbital positions, site energies, hoppings, and hopping distances.
        """
        output = []
        header = (
            "----------------------------------------\n"
            "       Tight-binding model report       \n"
            "----------------------------------------\n"
            f"r-space dimension           = {self._dim_r}\n"
            f"k-space dimension           = {self._dim_k}\n"
            f"number of spin components   = {self._nspin}\n"
            f"periodic directions         = {self._per}\n"
            f"number of orbitals          = {self._norb}\n"
            f"number of electronic states = {self._nstate}\n"
        )
        output.append(header)

        # Print Lattice and Orbital Vectors
        if not short:
            formatter = {
                "float_kind": lambda x: f"{0:^7.0f}" if abs(x) < 1e-10 else f"{x:^7.3f}"
            }
            output.append("Lattice vectors (Cartesian):")
            for i, vec in enumerate(self._lat):
                # print(f"  # {i} ===> {np.array2string(vec, formatter=formatter, separator=', ')}")
                output.append(
                    f"  # {i} ===> {np.array2string(vec, formatter=formatter, separator=', ')}"
                )

            output.append("Orbital vectors (dimensionless):")
            for i, orb in enumerate(self._orb):
                # print(f"  # {i} ===> {np.array2string(orb, formatter=formatter, separator=', ')}")
                output.append(
                    f"  # {i} ===> {np.array2string(orb, formatter=formatter, separator=', ')}"
                )

            # Print Site Energies
            output.append("Site energies:")
            for i, site in enumerate(self._site_energies):
                if self._nspin == 1:
                    energy_str = f"{site:^7.3f}"
                elif self._nspin == 2:
                    energy_str = str(site).replace("\n", " ")

                output.append(f"  # {i} ===> {energy_str}")

            output.append("Hoppings:")
            for i, hopping in enumerate(self._hoppings):
                out_str = f"  < {hopping[1]:^1} | H | {hopping[2]:^1}"
                if len(hopping) == 4:
                    out_str += " + ["
                    for j, v in enumerate(hopping[3]):
                        out_str += f"{v:^5.1f}"
                        if j != len(hopping[3]) - 1:
                            out_str += ", "
                        else:
                            out_str += "] >  ===> "
                if self._nspin == 1:
                    out_str += f"{hopping[0]:^7.4f}"
                elif self._nspin == 2:
                    out_str += str(hopping[0].round(4)).replace("\n", " ")
                output.append(out_str)

            output.append("Hopping distances:")

            for i, hopping in enumerate(self._hoppings):
                hop_from = hopping[1]
                hop_to = hopping[2]

                pos_i = np.dot(self._orb[hopping[1]], self._lat)
                pos_j = np.dot(self._orb[hopping[2]], self._lat)

                out_str = f"  | pos({hop_from:^1}) - pos({hop_to:^1}"

                if len(hopping) == 4:
                    pos_j += np.dot(hopping[3], self._lat)

                    out_str += " + ["
                    for j, Rv in enumerate(hopping[3]):
                        out_str += f"{Rv:^5.1f}"
                        if j != len(hopping[3]) - 1:
                            out_str += ", "
                        else:
                            out_str += "]"

                distance = np.linalg.norm(pos_j - pos_i)

                out_str += f") | = {distance:^7.3f}"
                output.append(out_str)

        if show:
            print("\n".join(output))
        else:
            return "\n".join(output)

    def _get_periodic_H(self, H_flat, k_vals):
        r"""
        Transform Hamiltonian to periodic gauge so that :math:`H(\mathbf{k}+\mathbf{G}) = H(\mathbf{k})`.

        If `nspin`= 2, `H_flat` should only be flat along k and NOT spin.

        Parameters
        ----------
        H_flat : np.ndarray
            Hamiltonian flattened along the k-direction, shape (Nk, nstate, nstate[, nspin]).
        k_vals : np.ndarray
            Array of k-point values, shape (Nk, dim_k).

        Returns
        -------
        np.ndarray
            Hamiltonian in periodic gauge, shape (Nk, nstate, nstate[, nspin]).

        Notes
        -----
        The transformation applies phase factors to ensure periodicity in reciprocal space.
        """
        if k_vals.ndim != 2:
            raise ValueError(f"Invalid k_vals shape: {k_vals.shape}. Expected (Nk, dim_k).")
        if k_vals.shape[1] != self._dim_k:
            raise ValueError(f"Invalid k_vals shape: {k_vals.shape}. Expected (Nk, {self._dim_k}).")
        
        orb_vecs = self.orb_vecs  # reduced units
        orb_vec_diff = orb_vecs[:, None, :] - orb_vecs[None, :, :]
        if self._dim_k == 0:
            logger.warning(
                "No periodic directions in k-space. Returning H_flat unchanged."
            )
            return H_flat
        print(orb_vec_diff.shape, k_vals.shape)
        orb_vec_diff = orb_vec_diff[..., self._per]
        orb_phase = np.exp(
            1j * 2 * np.pi * np.matmul(orb_vec_diff, k_vals.T)
        ).transpose(2, 0, 1)
        H_per_flat = H_flat * orb_phase
        return H_per_flat

    # Property decorators for read-only access to model attributes
    @property
    def dim_r(self) -> int:
        """The dimensionality of real space.

        .. versionadded:: 2.0.0
        """
        return self._dim_r

    @property
    def dim_k(self) -> int:
        """The dimensionality of reciprocal space (periodic directions).

        .. versionadded:: 2.0.0
        """
        return self._dim_k

    @property
    def nspin(self) -> int:
        """The number of spin components.

        .. versionadded:: 2.0.0
        """
        return self._nspin

    @property
    def per(self) -> list[int]:
        """Periodic directions as a list of indices.

        .. versionadded:: 2.0.0

        Each index corresponds to a lattice vector in the model.
        """
        return self._per

    @property
    def norb(self) -> int:
        """The number of tight-binding orbitals in the model.

        .. versionadded:: 2.0.0
        """
        return self._norb

    @property
    def nstate(self) -> int:
        """The number of electronic states in the model is ``norb * nspin``.

        .. versionadded:: 2.0.0
        """
        return self._nstate

    @property
    def lat_vecs(self) -> np.ndarray:
        """Lattice vectors in Cartesian coordinates with shape ``(dim_r, dim_r)``.

        .. versionadded:: 2.0.0
        """
        return self._lat.copy()

    @property
    def orb_vecs(self) -> np.ndarray:
        """Orbital vectors in reduced coordinates with shape ``(norb, dim_r)``.

        .. versionadded:: 2.0.0
        """
        return self._orb.copy()

    @property
    def site_energies(self) -> np.ndarray:
        """On-site energies for each orbital. 

        .. versionadded:: 2.0.0

        Shape is ``(norb,)`` for spinless models, ``(norb, 2, 2)`` for spinful models.
        """
        return self._site_energies.copy()

    @property
    def hoppings(self) -> list[dict]:
        """List of hopping dictionaries for the model.

        .. versionadded:: 2.0.0

        Each hopping is represented as a dictionary with keys:
            - 'amplitude': hopping amplitude (complex or matrix)
            - 'from_orbital': index of starting orbital
            - 'to_orbital': index of ending orbital
            - 'lattice_vector': (optional) lattice vector displacement
        """
        raw = copy.deepcopy(self._hoppings)
        formatted = []
        for hop in raw:
            amp, i, j, *R = hop
            entry = {
                "amplitude": amp,
                "from_orbital": i,
                "to_orbital": j,
            }
            if R:
                entry["lattice_vector"] = R[0].tolist()
            formatted.append(entry)
        return formatted

    @property
    def assume_position_operator_diagonal(self) -> bool:
        """
        Is the position operator is diagonal.
        """
        return self._assume_position_operator_diagonal

    @assume_position_operator_diagonal.setter
    def assume_position_operator_diagonal(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("assume_position_operator_diagonal must be a boolean.")
        self._assume_position_operator_diagonal = value

    def copy(self) -> "TBModel":
        """Return a deep copy of the TBModel object.

        .. versionadded:: 2.0.0

        Returns
        -------
        TBModel
            A deep copy of the model.

        Examples
        --------
        >>> tb2 = tb.copy()
        """
        return copy.deepcopy(self)

    def clear_hoppings(self):
        """Clear all hoppings in the model.

        .. versionadded:: 2.0.0

        Notes
        -----
        This is useful for resetting the model to a state without any hoppings.
        """
        self._hoppings.clear()
        logger.info("Cleared all hoppings.")

    def clear_onsite(self):
        """Clear all on-site energies in the model.

        .. versionadded:: 2.0.0

        Notes
        -----
        This is useful for resetting the model to a state without any on-site energies.
        """
        self._site_energies.fill(0)
        self._site_energies_specified.fill(False)
        logger.info("Cleared all on-site energies.")

    @deprecated("Use 'norb' property instead.")
    def get_num_orbitals(self):
        """
        .. deprecated:: 2.0.0
           Use 'norb' property instead.
        """
        return self.norb

    def get_orb(self, cartesian=False):
        """Return orbital positions.

        .. versionadded:: 2.0.0
            Added support for Cartesian coordinates with the `cartesian` parameter.

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
        orbs = self.orb_vecs
        if cartesian:
            return orbs @ self.lat_vecs
        else:
            return orbs

    def get_lat(self):
        """Return lattice vectors in Cartesian coordinates.

        Returns
        -------
        np.ndarray
            Lattice vectors, shape ``(dim_r, dim_r)``.
        """
        return self.lat_vecs
    
    def get_recip_lat(self):
        r"""Reciprocal lattice vectors in inverse Cartesian coordinates.

        .. versionadded:: 2.0.0

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
            return np.zeros((0, self.dim_r))

        if self.dim_k > self.dim_r:
            # You cannot have more k-dimensions than real-space dimensions.
            raise ValueError(
                f"dim_k ({self.dim_k}) cannot exceed dim_r ({self.dim_r}); "
                "k-space must be defined by a set of independent real-space periods."
            )

        # Select the real-space lattice vectors that generate k-space.
        # Prefer an explicit list (e.g. self.per holds indices of periodic directions).
        # Fallback: take the first dim_k lattice vectors.
        lat = np.asarray(self.lat_vecs)            # shape (dim_r, dim_r) in Cartesian coords
        if hasattr(self, "per") and self.per is not None:
            per = np.asarray(self.per, dtype=int)
            if per.size != self.dim_k:
                raise ValueError(f"'per' must list exactly dim_k={self.dim_k} periodic directions.")
            A_sub = lat[per, :]                    # (dim_k, dim_r)
        else:
            A_sub = lat[: self.dim_k, :]           # (dim_k, dim_r)

        # Check linear independence of the chosen periodic vectors
        if np.linalg.matrix_rank(A_sub) != self.dim_k:
            raise ValueError(
                "Periodic real-space vectors are not linearly independent; "
                "cannot construct reciprocal lattice for k-subspace."
            )

        # Minimum-norm reciprocal set in the embedding R^{dim_r}:
        # rows b_i satisfy A_sub @ B^T = 2pi I_{dim_k}
        G = A_sub @ A_sub.T                        # (dim_k, dim_k) Gram matrix
        X = np.linalg.solve(G, A_sub)     # (dim_k, dim_r)
        B = (2 * np.pi) * X             # (dim_k, dim_r)
        return B

    def get_recip_vol(self):
        """Return the volume of the reciprocal lattice.

        .. versionadded:: 2.0.0

        The volume is defined as the absolute value of the determinant
        of the reciprocal lattice vectors.

        Returns
        -------
        float
            Volume of the reciprocal lattice.

        Notes
        -----
        Only defined when ``dim_k == dim_r``.
        """
        recip_lat_vecs = self.get_recip_lat()
        if self._dim_k == 0:
            logger.warning(
                "Reciprocal volume is not defined for zero-dimensional k-space."
            )
            return 0.0
        if self._dim_k != self._dim_r:
            logger.warning(
                "Reciprocal volume is not defined for systems where k-space and real-space dimensions differ."
            )
            return 0.0
        if (
            recip_lat_vecs.shape[0] != self._dim_k
            or recip_lat_vecs.shape[1] != self._dim_r
        ):
            raise ValueError(
                "Reciprocal lattice vectors must have shape (dim_k, dim_r)."
            )
        if np.linalg.det(recip_lat_vecs) == 0:
            raise ValueError("Reciprocal lattice vectors are not linearly independent.")
        # Calculate the volume of the reciprocal lattice
        # The volume is the absolute value of the determinant of the reciprocal lattice vectors
        return abs(np.linalg.det(recip_lat_vecs))

    def get_cell_vol(self):
        """Returns the volume of a 2D unit cell. 

        Returns
        -------
        float
            Volume of the 2D unit cell.
        """
        if self.dim_r == 0:
            logger.warning(
                "Cell volume is not defined for zero-dimensional real-space."
            )
            return 0.0
        
        lat_vecs = self.lat_vecs
        if self.dim_r == 1:
            return np.linalg.norm(lat_vecs[0])
        elif self.dim_r == 2:
            lat_vecs = self.lat_vecs
            return np.linalg.norm(np.cross(lat_vecs[0], lat_vecs[1]))
        elif self.dim_r == 3:
            return abs(np.linalg.det(lat_vecs))
        else:
            raise NotImplementedError("Cell volume is only defined for 1D, 2D, and 3D systems.")

    def set_onsite(self, onsite_en, ind_i=None, mode="set"):
        r"""Define on-site energies for tight-binding orbitals.

        You can set the energy for a single orbital (by specifying `ind_i`), or for all
        orbitals at once (by passing a list/array to `onsite_en`).

        .. deprecated:: 2.0.0
            Using 'reset' for `mode` is deprecated, use 'set' instead.

        Parameters
        ----------
        onsite_en : float, array-like, np.ndarray of shape ``(2, 2)``
            If `ind_i` is unspecified or None, `onsite_en` must be a list/array of length `norb`.
            Otherwise, it may be a single value or a 2x2 matrix in the spinful case.

            For spinless models (``nspin=1``):
                - Real scalar or list/array of real scalars (one per orbital).
            For spinful models (``nspin=2``):
                - Scalar: interpreted as :math:`a I` for both spin components.
                - 4-vector ``[a, b, c, d]``: interpreted as :math:`a I + b \sigma_x + c \sigma_y + d \sigma_z`: 
                    
                    .. math::
                        \begin{bmatrix}
                            a + d & b - i c \\
                            b + i c & a - d
                        \end{bmatrix}

                - Full 2x2 Hermitian matrix.

        ind_i : int, optional
            Index of tight-binding orbital to update. If None, all orbitals are updated and
            an array of the same shape as `onsite_en` is expected.
        mode : {'set', 'add'}, optional
            Specifies how `onsite_en` is used
            - "set": On-site energy is set to the value of `onsite_en`. (Default)
            - "add": Adds to the previous value of on-site energy.

        Notes
        -----
        If called multiple times with "add", values are accumulated.

        Examples
        --------
        >>> tb.set_onsite([0.0, 1.0, 2.0])
        >>> tb.set_onsite(100.0, 1, mode="add")
        >>> tb.set_onsite(0.0, 1, mode="set")
        >>> tb.set_onsite([2.0, 3.0, 4.0], mode="set")
        """
        # Handle deprecated 'reset' mode
        mode = mode.lower()
        if mode == "reset":
            logger.warning(
                "The 'reset' mode is deprecated as of v2.0. Use 'set' instead to set the onsite energy."
                "This will be removed in a future version."
            )
            mode = "set"

        def process(val):
            block = self._val_to_block(val)
            if not is_Hermitian(block):
                raise ValueError(
                    "Onsite terms should be real, or in case where it is a matrix, Hermitian."
                )
            return block

        # prechecks
        if ind_i is None:
            # when ind_i is not specified, onsite_en should be a list or array
            if not isinstance(onsite_en, (list, np.ndarray)):
                raise TypeError(
                    "When ind_i is not specified, onsite_en must be a list or array."
                )
            # the number of onsite energies must match the number of orbitals,
            if len(onsite_en) != self._norb:
                raise ValueError(
                    "List of onsite energies must include a value for every orbital."
                )

            processed = [process(val) for val in onsite_en]
            indices = np.arange(self._norb)
        else:
            if ind_i < 0 or ind_i >= self._norb:
                raise ValueError(
                    "Index ind_i is not within the range of number of orbitals."
                )
            processed = [process(onsite_en)]
            indices = [ind_i]

        if mode == "set":
            for idx, block in zip(indices, processed):
                if self._site_energies_specified[idx]:
                    logger.warning(
                        f"Onsite energy for site {idx} was already set; resetting to the specified values."
                    )
                self._site_energies[idx] = block
                self._site_energies_specified[idx] = True

        elif mode == "add":
            for idx, block in zip(indices, processed):
                self._site_energies[idx] += block
                self._site_energies_specified[idx] = True
        else:
            raise ValueError("Mode should be either 'set' or 'add'.")
        
    def _ensure_hop_index(self):
        """Create a hash index (i,j,R_per)->list index for O(1) lookups."""
        if getattr(self, "_hop_index", None) is None:
            self._hop_index = {}
            for idx, h in enumerate(self._hoppings):
                if self._dim_k == 0:
                    key = (h[1], h[2], None)
                else:
                    key = (h[1], h[2], tuple(np.array(h[3])[self._per]))
                self._hop_index[key] = idx

    def _hop_keys(self, ind_i, ind_j, ind_R):
        """Return primary and conjugate keys using only periodic components of R."""
        if self._dim_k == 0:
            return (ind_i, ind_j, None), (ind_j, ind_i, None)
        Rp = tuple(np.array(ind_R)[self._per])
        conj_Rp = tuple((-np.array(ind_R))[self._per])
        return (ind_i, ind_j, Rp), (ind_j, ind_i, conj_Rp)


    def set_hop(
        self,
        hop_amp,
        ind_i: int,
        ind_j: int,
        ind_R=None,
        mode="set",
        allow_conjugate_pair=False,
    ):
        r"""Define hopping parameters between tight-binding orbitals.

        In the notation of tight-binding formalism, this function specifies:

        .. math::
            H_{ij}(\mathbf{R}) = \langle \phi_{\\mathbf{0},i} | H | \phi_{\mathbf{R},j} \rangle

        where :math:`\langle \phi_{\mathbf{0},i} |` is the i-th orbital in the home unit cell,
        and :math:`| \phi_{\mathbf{R},j} \rangle` is the j-th orbital in a cell shifted by lattice vector :math:`\mathbf{R}`.

        .. deprecated:: 2.0.0
            Using 'reset' for `mode` is deprecated, use 'set' instead.

        Parameters
        ----------
        hop_amp : scalar, array-like, np.ndarray of shape ``(2, 2)``
            For spinless models (`nspin=1`):
                - Real scalar or list/array of real scalars (one per orbital).
            For spinful models (`nspin=2`):
                - Scalar: interpreted as :math:`a I` for both spin components.
                - 4-vector ``[a, b, c, d]``: interpreted as :math:`a I + b \sigma_x + c \sigma_y + d \sigma_z`:

                    .. math::
                        \begin{bmatrix}
                            a + d & b - i c \\
                            b + i c & a - d
                        \end{bmatrix}

                - Full 2x2 Hermitian matrix.
        ind_i : int
            Index of bra orbital (in home unit cell).
        ind_j : int
            Index of ket orbital (in cell shifted by `ind_R`).
        ind_R : array-like of int, optional
            Lattice vector in reduced coordinates pointing to the unit cell
            where the ket orbital is located. Must have length `dim_r`. If model is non-periodic,
            can be omitted.
        mode : {'set', 'add'}, optional
            Specifies how `hop_amp` is used
                - "set": Set the hopping term to the value of `hop_amp`. (Default)
                - "add": Add `hop_amp` to the previous value.
        allow_conjugate_pair : bool, optional
            If True, allows specification of both a hopping and its conjugate pair.
            If False, prevents double-counting.

        Notes
        -----
        Strictly speaking, this term specifies hopping amplitude for hopping from site j+R to site i, not vice-versa.
        There is no need to specify hoppings in both :math:`i \rightarrow j+\mathbf{R}` and
        :math:`j \rightarrow i-\mathbf{R}` directions, since the latter is included automatically as

        .. math::
            H_{ji}(-\mathbf{R}) = \left[ H_{ij}(\mathbf{R}) \right]^*

        Examples
        --------
        >>> tb.set_hop(0.3+0.4j, 0, 2, [0, 1])
        >>> tb.set_hop(0.1+0.2j, 0, 2, [0, 1], mode="set")
        >>> tb.set_hop(100.0, 0, 2, [0, 1], mode="add")
        """
        #### Prechecks and formatting ####
        # deprecation warning
        if mode == "reset":
            logger.warning(
                "The 'reset' mode is deprecated as of v2.0. Use 'set' instead to set the hopping term."
                "This will be removed in a future version."
            )
            mode = "set"

        # make sure ind_i and ind_j are not out of scope
        if ind_i < 0 or ind_i >= self.norb:
            raise ValueError("Index ind_i is not within range of number of orbitals.")
        if ind_j < 0 or ind_j >= self.norb:
            raise ValueError("Index ind_j is not within range of number of orbitals.")

        # Check ind_R
        if isinstance(ind_R, np.ndarray):
            if ind_R.ndim != 1:
                raise ValueError("If ind_R is a numpy array, it must be 1-dimensional.")
            if ind_R.shape[0] != self._dim_r:
                raise ValueError(
                    "If ind_R is a numpy array, its length must equal dim_r."
                )
        elif isinstance(ind_R, (list, tuple)):
            if len(ind_R) != self._dim_r:
                raise ValueError(
                    "If ind_R is a list or tuple, its length must equal dim_r."
                )
            ind_R = np.array(ind_R)
            if ind_R.ndim != 1:
                raise ValueError("If ind_R is a list or tuple, it must be 1-dimensional.")
        elif isinstance(ind_R, (int, np.integer)):
            if self.dim_k != 1:
                raise ValueError(
                    "If dim_k is not 1, should not use integer for ind_R. Instead use list."
                )
            tmpR = np.zeros(self.dim_r, dtype=int)
            tmpR[self.per] = ind_R
            ind_R = tmpR
        elif ind_R is None:
            if self.dim_k != 0:
                raise ValueError("Must specify ind_R when we have a periodic direction.")
        else:
            raise TypeError(
                "ind_R is not of correct type. Should be array-type or integer."
            )

        # Do not allow onsite hoppings to be specified here
        if ind_i == ind_j:
            # not extended
            if self._dim_k == 0:
                raise ValueError(
                    "Do not use set_hop for onsite terms. Use set_onsite instead."
                )
            # hopping within unit cell
            elif ind_R is not None and bool(np.all(ind_R == 0)):
                raise ValueError(
                    "Do not use set_hop for onsite terms. Use set_onsite instead."
                )

        self._ensure_hop_index()
        key, conj_key = self._hop_keys(int(ind_i), int(ind_j), ind_R)
        if not allow_conjugate_pair and (conj_key in self._hop_index):
            raise ValueError(
                f"Conjugate element already specified for i={ind_i}, j={ind_j}, R={ind_R}. "
                "Either avoid double entry or set allow_conjugate_pair=True."
            )

        # convert to 2x2 matrix if needed
        hop_use = self._val_to_block(hop_amp)
        # hopping term parameters to be stored
        if self.dim_k == 0:
            new_hop = [hop_use, int(ind_i), int(ind_j)]
        else:
            new_hop = [hop_use, int(ind_i), int(ind_j), np.array(ind_R)]

        use_index = self._hop_index.get(key, None)
        mode = mode.lower()
        if mode == "set":
            if use_index is not None:
                # Reset
                self._hoppings[use_index] = new_hop
            else:
                # Append
                self._hoppings.append(new_hop)
                self._hop_index[key] = len(self._hoppings) - 1
        elif mode == "add":
            if use_index is not None:
                self._hoppings[use_index][0] += new_hop[0]
            else:
                self._hoppings.append(new_hop)
                self._hop_index[key] = len(self._hoppings) - 1
        else:
            raise ValueError("Wrong value of mode parameter. Should be either `set` or `add`.")
    

    def _val_to_block(self, val):
        r"""
        Convert input value to appropriate matrix block for onsite or hopping.

        For nspin=1, returns the value (should be real or complex scalar).
        For nspin=2:
            - Scalar: returns a 2x2 matrix proportional to the identity.
            - Array with up to four elements: returns a 2x2 matrix as
              :math:`a I + b \sigma_x + c \sigma_y + d \sigma_z`.
            - 2x2 matrix: returns the matrix as is.

        Parameters
        ----------
        val : float, complex, list, np.ndarray
            Value to convert.

        Returns
        -------
        float, complex, or np.ndarray
            Matrix block for onsite or hopping.

        Raises
        ------
        ValueError
            If input is not a valid format.
        """
        # spinless case
        if self._nspin == 1:
            if not isinstance(
                val, (int, np.integer, np.floating, float, complex, np.complexfloating)
            ):
                raise TypeError("For spinless case, value must be a scalar.")
            return val

        # spinful case: construct 2x2 matrix
        coeffs = np.array(val, dtype=complex)
        paulis = [SIGMA0, SIGMAX, SIGMAY, SIGMAZ]
        if coeffs.shape == ():
            # scalar -> identity
            return coeffs * SIGMA0
        elif coeffs.shape == (4,):
            block = sum([val * paulis[i] for i, val in enumerate(coeffs)])
        elif coeffs.shape == (2, 2):
            block = coeffs
        else:
            raise TypeError(
                "For spinful models, value should be a scalar, length-4 iterable, or 2x2 array."
            )
        return block
    

    def grad_ham(self, k_pts, cartesian=False):
        r"""Generate the gradient of the Hamiltonian

        The gradient is the derivative of the block Hamiltonian along each reciprocal
        lattice direction :math:`v_k = \partial_k H_k` for an array of k-points.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        k_pts : array of shape (Nk, dim_k)
            Array of k-points in reduced coordinates.
        cartesian : bool, optional
            If True, use Cartesian coordinates for the velocity operator.

        Returns
        -------
        vel : array of shape (dim_k, Nk, norb, norb)
            Velocity operators at each k-point. First axis indexes the cartesian direction if `cartesian` is True.
            Otherwise, it indexes the reduced direction.

        Notes
        -----
        The velocity operator is defined via the derivative of the Hamiltonian
        with respect to k, i.e.,

        .. math::
            v_k^{\mu} = i \frac{\partial H(k)}{\partial k_{\mu}}
        """
        dim_k = self._dim_k

        if k_pts is not None:
            # if kpnt is just a number then convert it to an array
            if isinstance(k_pts, (int, np.integer, float)):
                if dim_k != 1:
                    raise ValueError(
                        "k_pts should be a 2D array of shape (n_kpts, dim_k)."
                    )
                k_arr = np.array([[k_pts]])
            elif isinstance(k_pts, (list, np.ndarray)):
                k_arr = np.array(k_pts)
                if k_arr.ndim == 1:
                    if k_arr.shape[0] != dim_k:
                        return ValueError(
                            "If 'k_pts' is a single k-point, it must be of shape dim_k."
                        )
                    else:
                        # Reshape to (1, dim_k)
                        k_arr = k_arr[None, :]
            else:
                raise TypeError(
                    "k_pts should be a list or numpy array, or possibly a number for 1d k-space."
                )

            # check that k-vector is of corect size
            if k_arr.ndim != 2 or k_arr.shape[-1] != dim_k:
                raise ValueError("k_arr should be a 2D array of shape (n_kpts, dim_k).")
        else:
            raise TypeError("k_pts should not be None for velocity operator.")

        norb = self._norb
        nspin = self._nspin
        per = np.asarray(self._per)
        orb_red = np.asarray(self._orb)  # shape (norb, dim_r)
        hoppings = self._hoppings

        i_indices = np.array([h[1] for h in hoppings])
        j_indices = np.array([h[2] for h in hoppings])
        amps = np.array([h[0] for h in hoppings], dtype=complex)

        # Precompute delta_r for all hoppings
        orb_i = orb_red[i_indices]  # Shape: (n_hoppings, dim_r)
        orb_j = orb_red[j_indices]  # Shape: (n_hoppings, dim_r)

        ind_Rs = np.array([h[3] for h in hoppings], dtype=float)

        delta_r = ind_Rs - orb_i + orb_j  # Shape: (n_hoppings, dim_r)
        delta_r_per = delta_r[:, per]  # Shape: (n_hoppings, dim_k)

        # # Compute phase factors for all k-points and hoppings
        k_dot_r = k_arr @ delta_r_per.T  # Shape: (n_kpts, n_hoppings)
        phases = np.exp(1j * 2 * np.pi * k_dot_r)  # Shape: (n_kpts, n_hoppings)
        if cartesian:
            deriv_phase = (1j * delta_r_per @ self.get_lat()[self._per, :]).T[
                :, None, :
            ] * phases[None, ...]
        else:
            deriv_phase = (1j * 2 * np.pi * delta_r_per).T[:, None, :] * phases[
                None, ...
            ]

        n_hops = len(hoppings)
        if nspin == 1:
            T_f = np.zeros((n_hops, norb, norb), complex)
            T_r = np.zeros((n_hops, norb, norb), complex)
            idx = np.arange(n_hops)
            T_f[idx, i_indices, j_indices] = amps
            T_r[idx, j_indices, i_indices] = amps.conj()

        else:
            # spinful: each amp is a 2×2 block
            T_f = np.zeros((n_hops, norb, 2, norb, 2), complex)
            T_r = np.zeros_like(T_f)
            for h in range(n_hops):
                T_f[h, i_indices[h], :, j_indices[h], :] = amps[h]
                T_r[h, j_indices[h], :, i_indices[h], :] = amps[h].conj().T

        # compute forward contribution into vel array
        vel = np.tensordot(deriv_phase, T_f, axes=([2], [0]))
        # compute reverse contribution in temporary buffer
        temp = np.tensordot(deriv_phase.conj(), T_r, axes=([2], [0]))
        # add in-place to avoid extra allocation
        np.add(vel, temp, out=vel)

        return vel
    

    def hamiltonian(self, k_pts=None, flatten_spin=False):
        r"""Generate the Bloch Hamiltonian for an array of k-points in reduced coordinates.

        The Hamiltonian is computed in tight-binding convention I, which includes phase factors
        associated with orbital positions in the hopping terms:

        .. math::

            H_{ij}(k) = \sum_{\mathbf{R}} t_{ij}(\mathbf{R}) \exp[i \mathbf{k} \cdot (\mathbf{r}_i - \mathbf{r}_j + \mathbf{R})]

        where :math:`t_{ij}(R)` is the hopping amplitude from orbital j to i through lattice vector :math:`\mathbf{R}`.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        k_pts : (Nk, dim_k) array, optional
            Array of k-points in reduced coordinates.
            If `None`, the Hamiltonian is computed at a single point (`dim_k = 0`),
            corresponding to a finite sample.

        Returns
        -------
        ham : np.ndarray 
            Array of Bloch-Hamiltonian matrices defined on the specified k-points. The Hamiltonian is Hermitian by construction.

            - If `dim_k` > 0: shape is (n_kpts, n_orb, n_orb) for spinless models, or (n_kpts, n_orb, 2, n_orb, 2) 
              for spinful models.

            - If `dim_k` = 0: shape is (n_orb, n_orb) for spinless or (n_orb, 2, n_orb, 2) for spinful models.

        Notes
        -----
        In convention I, the Hamiltonian satisfies:

        .. math::

            H(k) \neq H(k + G), \quad \text{but instead} \quad H(k) = U H(k + G) U^{\dagger}

        where :math:`G` is a reciprocal lattice vector and :math:`U` is a unitary transformation
        relating the two.

        Finite difference estimates of :math:`\partial_{k_\mu} H(k)` may not be accurate at
        boundaries due to the gauge discontinuity inherent in convention I.        

        """
        # Cache invariant data to avoid repeated conversions
        dim_k = self._dim_k
        norb = self._norb
        nspin = self._nspin
        per = np.asarray(self._per)
        orb_red = np.asarray(self._orb)  # shape (norb, dim_r)
        orb_idxs = np.arange(norb)
        site_energies = np.asarray(self._site_energies)
        hoppings = self._hoppings

        if k_pts is not None:
            # if kpnt is just a number then convert it to an array
            if isinstance(k_pts, (int, np.integer, float)):
                if dim_k != 1:
                    raise ValueError(
                        "k_pts should be a 2D array of shape (n_kpts, dim_k)."
                    )
                k_arr = np.array([[k_pts]])
            elif isinstance(k_pts, (list, np.ndarray)):
                k_arr = np.asarray(k_pts)
                if k_arr.ndim == 1:
                    if k_arr.shape[0] != dim_k:
                        return ValueError(
                            "If 'k_pts' is a single k-point, it must be of shape dim_k."
                        )
                    else:
                        # Reshape to (1, dim_k)
                        k_arr = k_arr[None, :]
            else:
                raise TypeError(
                    "k_pts should be a list or numpy array, or possibly a number for 1d k-space."
                )

            # check that k-vector is of corect size
            if k_arr.ndim != 2 or k_arr.shape[-1] != dim_k:
                raise ValueError("k_arr should be a 2D array of shape (n_kpts, dim_k).")

            n_kpts = k_arr.shape[0]
            if nspin == 1:
                ham = np.zeros((n_kpts, norb, norb), dtype=complex)
            elif nspin == 2:
                ham = np.zeros((n_kpts, norb, 2, norb, 2), dtype=complex)
            else:
                raise ValueError("Invalid spin value.")
        else:
            if dim_k != 0:
                raise ValueError(
                    "Must provide a list of k-vectors for the Bloch Hamiltonian of extended systems."
                )
            else:  # finite sample
                if nspin == 1:
                    ham = np.zeros((norb, norb), dtype=complex)
                elif nspin == 2:
                    ham = np.zeros((norb, 2, norb, 2), dtype=complex)
                else:
                    raise ValueError("Invalid spin value.")

        hop_amps = np.array([h[0] for h in hoppings], dtype=complex)
        i_indices = np.array([h[1] for h in hoppings])
        j_indices = np.array([h[2] for h in hoppings])
        n_hops = len(hoppings)

        if dim_k == 0:
            if nspin == 1:
                ham = np.zeros((norb, norb), complex)
                np.add.at(ham, (i_indices, j_indices), hop_amps)
                np.add.at(ham, (j_indices, i_indices), hop_amps.conj())
                np.fill_diagonal(ham, site_energies)
            elif nspin == 2:
                ham = np.zeros((norb, 2, norb, 2), dtype=complex)
                for h in range(n_hops):
                    ham[i_indices[h], :, j_indices[h], :] += hop_amps[h]
                    ham[j_indices[h], :, i_indices[h], :] += hop_amps[h].conj().T

                for orb in orb_idxs:
                    ham[orb, :, orb, :] += site_energies[orb]

        else:
            # Compute phase factors for all k-points and hoppings
            orb_i = orb_red[i_indices]  # Shape: (n_hoppings, dim_r)
            orb_j = orb_red[j_indices]  # Shape: (n_hoppings, dim_r)
            ind_Rs = np.array([h[3] for h in hoppings], dtype=float)

            delta_r = ind_Rs - orb_i + orb_j  # Shape: (n_hoppings, dim_r)
            delta_r_per = delta_r[:, per]  # Shape: (n_hoppings, dim_k)

            k_dot_r = k_arr @ delta_r_per.T  # Shape: (n_kpts, n_hoppings)
            phases = np.exp(1j * 2 * np.pi * k_dot_r)  # Shape: (n_kpts, n_hoppings)

            if nspin == 1:
                T_f = np.zeros((n_hops, norb, norb), complex)
                T_r = np.zeros((n_hops, norb, norb), complex)
                idx = np.arange(n_hops)
                T_f[idx, i_indices, j_indices] = hop_amps
                T_r[idx, j_indices, i_indices] = hop_amps.conj()
            elif nspin == 2:
                T_f = np.zeros((n_hops, norb, 2, norb, 2), complex)
                T_r = np.zeros((n_hops, norb, 2, norb, 2), complex)
                for h in range(n_hops):
                    T_f[h, i_indices[h], :, j_indices[h], :] = hop_amps[h]
                    T_r[h, j_indices[h], :, i_indices[h], :] = hop_amps[h].conj().T

            ham = np.tensordot(phases, T_f, axes=([1], [0]))
            ham_hc = np.tensordot(phases.conj(), T_r, axes=([1], [0]))
            np.add(ham, ham_hc, out=ham)

            # fill diagonal elements with onsite energies
            for orb in orb_idxs:
                if nspin == 1:
                    ham[..., orb, orb] += site_energies[orb]
                elif nspin == 2:
                    ham[..., orb, :, orb, :] += site_energies[orb]
        
        if flatten_spin and nspin == 2:
            ham = ham.reshape(*ham.shape[:-4], self.norb * self.nspin, self.norb * self.nspin)

        return ham

    def _sol_ham(self, ham, return_eigvecs=False, keep_spin_ax=True, tf_speedup=False):
        """Solves Hamiltonian and returns eigenvectors, eigenvalues"""

        # shape(ham): (Nk, n_orb, n_orb), (Nk, n_orb, n_spin, n_orb, n_spin)
        # or in finite cases (n_orb, n_orb), (n_orb, n_spin, n_orb, n_spin)
        # flatten spin axes
        if ham.ndim == 2 * self.nspin + 1:
            # have k points
            new_shape = (ham.shape[0],) + (self.nstate, self.nstate)
            if self.nspin == 1:
                shape_evecs = (ham.shape[0],) + (self.norb, self.norb)
            elif self.nspin == 2:
                shape_evecs = (ham.shape[0],) + (
                    self.nstate,
                    self.norb,
                    self.nspin,
                )
        elif ham.ndim == 2 * self.nspin:
            # must be a finite sample, no k-points
            new_shape = (self.nstate, self.nstate)
            if self.nspin == 1:
                shape_evecs = (self.norb, self.norb)
            elif self.nspin == 2:
                shape_evecs = (self.nstate, self.norb, self.nspin)
        else:
            raise ValueError("Hamiltonian has wrong shape.")

        ham_use = ham.reshape(*new_shape)

        if not np.allclose(ham_use, ham_use.swapaxes(-1, -2).conj()):
            raise ValueError("Hamiltonian matrix is not Hermitian.")

        # solve matrix
        if not return_eigvecs:
            if tf_speedup:
                from tensorflow import convert_to_tensor
                from tensorflow import complex64 as tfcomplex64
                from tensorflow.linalg import eigvalsh as tfeigvalsh

                H_tf = convert_to_tensor(ham_use, dtype=tfcomplex64)
                eval = tfeigvalsh(H_tf)
                eval = eval.numpy()
                return eval
            else:
                return np.linalg.eigvalsh(ham_use)
        else:
            if tf_speedup:
                from tensorflow import convert_to_tensor
                from tensorflow import complex64 as tfcomplex64
                from tensorflow.linalg import eigh as tfeigh

                H_tf = convert_to_tensor(ham_use, dtype=tfcomplex64)
                eval, evec = tfeigh(H_tf)
                eval = eval.numpy()
                evec = evec.numpy()
            else:
                eval, evec = np.linalg.eigh(ham_use)

            # transpose matrix eig since otherwise it is confusing
            # now eig[i,:] is eigenvector for eval[i]-th eigenvalue
            evec = evec.swapaxes(-1, -2)
            if keep_spin_ax:
                evec = evec.reshape(*shape_evecs)

            return eval, evec

    def solve_ham(
            self, 
            k_list = None, 
            return_eigvecs: bool = False, 
            keep_spin_ax: bool = True,
            tf_speedup: bool = False):
        r"""Diagonalize the Hamiltonian 
        
        Solve for eigenvalues and optionally eigenvectors of the tight-binding model
        at a list of one-dimensional k-vectors.

        .. versionadded:: 2.0.0
            Merged :func:`solve_all` and :func:`solve_one` into this function.

        Parameters
        ----------
        k_list : array_like, optional
            One-dimensional list or array of k-vectors, each given in reduced coordinates.
            Shape should be (Nk, dim_k), where dim_k is the number of periodic directions.
            Should not be specified for systems with zero-dimensional reciprocal space.

        return_eigvecs : bool, optional
            If True, both eigenvalues and eigenvectors are returned.
            If False (default), only eigenvalues are returned.

        keep_spin_ax : bool, optional
            If True (default), the spin axes are kept in the output eigenvectors.
            If False, the spin axes are flattened.

        Returns
        -------
        eval : {(Nk, nstate), (nstate)} np.ndarray 
            Array of eigenvalues. Shape is:
            - (Nk, nbnd) for periodic systems
            - (nbnd,) for zero-dimensional (molecular) systems

        evec : {(Nk, nstate, nstate), (nstate, nstate), (Nk, nstate, norb, 2), (nstate, norb, 2)} np.ndarray, optional
            Array of eigenvectors (if `return_eigvecs=True`). The ordering of bands matches that in `eval`.

            Eigenvectors :code:`evec[k, n, j]` correspond to the coefficient
            :math:`C^{n \mathbf{k}}_j` in the expansion in orbital basis.

            For spinless models:
            - Shape is (Nk, nbnd, norb)
            For spinful models:
            - Shape is (Nk, nbnd, norb, 2)
            In zero-dimensional systems:
            - (nbnd, n_orb) or (nbnd, n_orb, 2)

        Notes
        -----
        This function uses the convention described in section 3.1 of the
        :download:`pythtb notes on tight-binding formalism </misc/pythtb-formalism.pdf>`.
        The returned wavefunctions correspond to the cell-periodic part
        :math:`u_{n \mathbf{k}}(\mathbf{r})` and not the full Bloch function
        :math:`\Psi_{n \mathbf{k}}(\mathbf{r})`.

        In many cases, using the :class:`pythtb.wf_array.WFArray` class offers a more
        elegant interface for handling eigenstates on a regular k-mesh.

        Examples
        --------
        Solve for eigenvalues at several k-points:

        >>> eval = tb.solve_ham([[0.0, 0.0], [0.0, 0.2], [0.0, 0.5]])

        Solve for eigenvalues and eigenvectors:

        >>> eval, evec = tb.solve_ham([[0.0, 0.0], [0.0, 0.2]], return_eigvecs=True)
        """

        Ham = self.hamiltonian(k_list)

        if return_eigvecs:
            eigvals, eigvecs = self._sol_ham(
                Ham, return_eigvecs=return_eigvecs, keep_spin_ax=keep_spin_ax, tf_speedup=tf_speedup
            )
            if self._dim_k != 0:
                if eigvals.ndim != 2:
                    raise ValueError("Wrong shape of eigvals")
                # if only one k_point, remove that redundant axis (reproduces solve_one)
                if eigvals.shape[0] == 1:
                    eigvals = eigvals[0]
                    eigvecs = eigvecs[0]

            return eigvals, eigvecs
        else:
            eigvals = self._sol_ham(Ham, return_eigvecs=return_eigvecs)

            if self._dim_k != 0:
                if eigvals.ndim != 2:
                    raise ValueError("Wrong shape of eigvals")
                # if only one k_point, remove that redundant axis (reproduces solve_one)
                if eigvals.shape[0] == 1:
                    eigvals = eigvals[0]
            return eigvals

    @deprecated("use .solve_ham() instead (since v2.0).", category=FutureWarning)
    def solve_one(self, k_list=None, eig_vectors=False):
        """
        .. deprecated:: 2.0.0
            Use .solve_ham() instead.
        """
        return self.solve_ham(
            k_list=k_list, return_eigvecs=eig_vectors, keep_spin_ax=True
        )

    @deprecated("use .solve_ham() instead (since v2.0).", category=FutureWarning)
    def solve_all(self, k_list=None, eig_vectors=False):
        """
        .. deprecated:: 2.0.0
            Use .solve_ham() instead.
        """
        return self.solve_ham(
            k_list=k_list, return_eigvecs=eig_vectors, keep_spin_ax=True
        )

    def cut_piece(self, num, fin_dir, glue_edgs=False) -> "TBModel":
        """Cut a (d-1)-dimensional piece out of a d-dimensional tight-binding model.
        
        Constructs a (d-1)-dimensional tight-binding model out of a
        d-dimensional one by repeating the unit cell a given number of
        times along one of the periodic lattice vectors. 
        
        Parameters
        ----------
        num : int
            How many times to repeat the unit cell.

        fin_dir : int
            Index of the real space lattice vector along
            which you no longer wish to maintain periodicity.

        glue_edgs : bool, optional
            If True, allow hoppings from one edge to the other of a cut model.

        Returns
        -------
        fin_model : TBModel
            Object of type :class:`pythtb.TBModel` representing a cutout
            tight-binding model. 

        See Also
        ---------
        :ref:`haldane-fin-nb` : For an example
        :ref:`haldane-edge-nb` : For an example

        Notes
        -----
        - Orbitals in `fin_model` are numbered so that the `i`-th orbital of the `n`-th unit 
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
        if self._dim_k == 0:
            raise Exception("\n\nModel is already finite")
        if not _is_int(num):
            raise TypeError("\n\nArgument num not an integer")

        # check value of num
        if num < 1:
            raise ValueError("\n\nArgument num must be positive!")
        if num == 1 and glue_edgs:
            raise ValueError("\n\nCan't have num=1 and glueing of the edges!")

        # generate orbitals of a finite model
        fin_orb = []
        onsite = []  # store also onsite energies
        for i in range(num):  # go over all cells in finite direction
            for j in range(self._norb):  # go over all orbitals in one cell
                # make a copy of j-th orbital
                orb_tmp = np.copy(self._orb[j, :])
                # change coordinate along finite direction
                orb_tmp[fin_dir] += float(i)
                # add to the list
                fin_orb.append(orb_tmp)
                # do the onsite energies at the same time
                onsite.append(self._site_energies[j])
        onsite = np.array(onsite)
        fin_orb = np.array(fin_orb)

        # generate periodic directions of a finite model
        fin_per = copy.deepcopy(self._per)
        # find if list of periodic directions contains the one you
        # want to make finite
        if fin_per.count(fin_dir) != 1:
            raise Exception("\n\nCan not make model finite along this direction!")
        # remove index which is no longer periodic
        fin_per.remove(fin_dir)

        # generate object of TBModel type that will correspond to a cutout
        fin_model = TBModel(
            self._dim_k - 1,
            self._dim_r,
            copy.deepcopy(self._lat),
            fin_orb,
            fin_per,
            self._nspin,
        )

        # remember if came from w90
        fin_model._assume_position_operator_diagonal = (
            self._assume_position_operator_diagonal
        )

        # now put all onsite terms for the finite model
        fin_model.set_onsite(onsite, mode="set")

        # put all hopping terms
        for c in range(num):  # go over all cells in finite direction
            for h in range(len(self._hoppings)):  # go over all hoppings in one cell
                # amplitude of the hop is the same
                amp = self._hoppings[h][0]

                # lattice vector of the hopping
                ind_R = copy.deepcopy(self._hoppings[h][3])
                # store by how many cells is the hopping in finite direction
                jump_fin = ind_R[fin_dir]
                if fin_model._dim_k != 0:
                    ind_R[fin_dir] = 0  # one of the directions now becomes finite

                # index of "from" and "to" hopping indices
                hi = self._hoppings[h][1] + c * self._norb
                #   have to compensate  for the fact that ind_R in finite direction
                #   will not be used in the finite model
                hj = self._hoppings[h][2] + (c + jump_fin) * self._norb

                # decide whether this hopping should be added or not
                to_add = True
                # if edges are not glued then neglect all jumps that spill out
                if not glue_edgs:
                    if hj < 0 or hj >= self._norb * num:
                        to_add = False
                # if edges are glued then do mod division to wrap up the hopping
                else:
                    hj = int(hj) % int(self._norb * num)

                # add hopping to a finite model
                if to_add:
                    if fin_model._dim_k == 0:
                        fin_model.set_hop(
                            amp, hi, hj, mode="add", allow_conjugate_pair=True
                        )
                    else:
                        fin_model.set_hop(
                            amp, hi, hj, ind_R, mode="add", allow_conjugate_pair=True
                        )

        return fin_model

    def make_finite(self, lx, ly):
        r"""Returns a finite model with open boundary conditions.

        Parameters
        ----------
        lx : int
            Number of unit cells of the sample along the :math:`\mathbf{a}_1` direction.
        ly : int
            Number of unit cells of the sample along the :math:`\mathbf{a}_2` direction.

        Returns
        -------
        finite : :class:`pythtb.TBModel`
                A model whose periodic hoppings have been removed (OBC model).
        """
        if not (lx > 0 and ly > 0):
            raise ValueError("Number of sites along finite direction must be greater than 0")

        ribbon = self.cut_piece(num = ly, fin_dir = 1, glue_edgs = False)
        finite = ribbon.cut_piece(num = lx, fin_dir = 0, glue_edgs = False)

        return finite

    def reduce_dim(self, remove_k, value_k) -> "TBModel":
        r"""Reduces dimensionality of the model by taking a reciprocal-space slice

        This function returns a d-1 dimensional tight-binding model obtained
        by constraining one of k-vector components in :math:`{\cal H}_{\bf k}`
        to be a constant.

        Parameters
        ----------
        remove_k : int
            Which reciprocal space unit vector component
            you wish to keep constant.

        value_k : float
            Value of the k-vector component to which you are
            constraining this model. Must be given in reduced coordinates.

        Returns
        -------
        red_tb : :class:`pythtb.TBModel`
            Reduced tight-binding model.

        Notes
        -----
        Reduces dimensionality of the model by taking a reciprocal-space
        slice of the Bloch Hamiltonian :math:`{\cal H}_{\bf k}`. The Bloch
        Hamiltonian (defined in :download:`notes on tight-binding
        formalism </misc/pythtb-formalism.pdf>` in section 3.1 equation 3.7) of a
        d-dimensional model is a function of d-dimensional k-vector.

        Examples
        --------- 
        Constrains second k-vector component to equal 0.3

        >>> red_tb = tb.reduce_dim(1, 0.3)

        """
        if self._dim_k == 0:
            raise Exception("\n\nCan not reduce dimensionality even further!")
        # make a copy
        red_tb = copy.deepcopy(self)
        # make one of the directions not periodic
        red_tb._per.remove(remove_k)
        red_tb._dim_k = len(red_tb._per)
        # check that really removed one and only one direction
        if red_tb._dim_k != self._dim_k - 1:
            raise Exception("\n\nSpecified wrong dimension to reduce!")

        # specify hopping terms from scratch
        red_tb._hoppings = []
        # set all hopping parameters for this value of value_k
        for h in range(len(self._hoppings)):
            hop = self._hoppings[h]
            if self._nspin == 1:
                amp = complex(hop[0])
            elif self._nspin == 2:
                amp = np.array(hop[0], dtype=complex)
            i, j = hop[1], hop[2]
            ind_R = np.array(hop[3], dtype=int)
            # vector from one site to another
            rv = -red_tb._orb[i, :] + red_tb._orb[j, :] + np.array(ind_R, dtype=float)
            # take only r-vector component along direction you are not making periodic
            rv = rv[remove_k]
            # Calculate the part of hopping phase, only for this direction
            phase = np.exp((2.0j) * np.pi * (value_k * rv))
            # store modified version of the hop
            # Since we are getting rid of one dimension, it could be that now
            # one of the hopping terms became onsite term because one direction
            # is no longer periodic
            if i == j and (np.all(np.array(ind_R[red_tb._per], dtype=int) == 0)):
                if ind_R[remove_k] == 0:
                    # in this case this is really an onsite term
                    red_tb.set_onsite(amp * phase, i, mode="add")
                else:
                    # in this case must treat both R and -R because that term would
                    # have been counted twice without dimensional reduction
                    if self._nspin == 1:
                        red_tb.set_onsite(
                            amp * phase + (amp * phase).conj(), i, mode="add"
                        )
                    elif self._nspin == 2:
                        red_tb.set_onsite(
                            amp * phase + (amp.T * phase).conj(), i, mode="add"
                        )
            else:
                # just in case make the R vector zero along the reduction dimension
                ind_R[remove_k] = 0
                # add hopping term
                red_tb.set_hop(
                    amp * phase, i, j, ind_R, mode="add", allow_conjugate_pair=True
                )

        return red_tb

    def change_nonperiodic_vector(
        self, 
        np_dir: int, 
        new_latt_vec=None, 
        to_home=True, 
        to_home_warning:bool=True
    ) -> "TBModel":
        """Change non-periodic lattice vector 

        .. versionchanged:: 2.0.0
            Parameter `to_home_supress_warning` has been renamed to `to_home_warning`.
            Note: this change inverts the meaning of the boolean parameter.
        
        Returns tight-binding model :class:`pythtb.TBModel` in which one of
        the non-periodic "lattice vectors" is changed.  Non-periodic vectors are those 
        elements of `lat` that are not listed as periodic with the `per` parameter.

        The returned object has modified reduced coordinates of orbitals, 
        consistent with the new choice of `lat`. Therefore, the actual 
        (Cartesian) coordinates of orbitals in original and returned :class:`pythtb.TBModel`
        are the same.

        Parameters
        ----------

        np_dir : int
            Index of nonperiodic lattice vector to change.

        new_latt_vec : array_like, optional
            The new nonperiodic lattice vector. If None (default), the new
            nonperiodic lattice vector is the same as the original one except
            that all components in the periodic space have been projected out
            (so that the new nonperiodic vector is perpendicular to all
            periodic vectors).

        to_home : bool, optional
            If ``True`` (default), shift all orbitals to the home cell along
            non-periodic directions.

        to_home_warning : bool, optional
            If ``True`` (default), code will print a warning message whenever
            returned object has an orbital with at least one reduced coordinate
            smaller than 0 or larger than 1 along a non-periodic direction. If
            ``False`` the warning message will not be printed.

            Note that this parameter has no effect on the model; it only determines whether a
            warning message is printed or not. 

        Returns
        --------
        nnp_tb : :class:`pythtb.TBModel`
            An equivalent tight-binding model with
            one redefined nonperiodic lattice vector.

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

        # Check that selected direction is nonperiodic
        if self.per.count(np_dir) == 1:
            raise ValueError(f"Selected direction {np_dir} is not nonperiodic")

        if new_latt_vec is None:
            # construct new nonperiodic lattice vector
            per_temp = np.zeros_like(self._lat)
            for direc in self._per:
                per_temp[direc] = self._lat[direc]
            # find projection coefficients onto space of periodic vectors
            coeffs = np.linalg.lstsq(per_temp.T, self._lat[np_dir], rcond=None)[0]
            projec = np.dot(self._lat.T, coeffs)
            # subtract off to get new nonperiodic vector
            np_lattice_vec = self._lat[np_dir] - projec
        else:
            # new_latt_vec is passed as argument
            # check shape and convert to numpy array
            np_lattice_vec = np.array(new_latt_vec)
            if np_lattice_vec.shape != (self._dim_r,):
                raise ValueError("Nonperiodic vector has wrong length")

        # define new set of lattice vectors
        np_lat = copy.deepcopy(self._lat)
        np_lat[np_dir] = np_lattice_vec

        # convert reduced vector in original lattice to reduced vector in new cell lattice
        np_orb = []
        for orb in self._orb:  # go over all orbitals
            orb_cart = np.dot(self._lat.T, orb)
            # convert to reduced coordinates
            np_orb.append(np.linalg.solve(np_lat.T, orb_cart))

        # create new TBModel object to be returned
        nnp_tb = copy.deepcopy(self)

        # update lattice vectors and orbitals
        nnp_tb._lat = np.array(np_lat, dtype=float)
        nnp_tb._orb = np.array(np_orb, dtype=float)
        print(np_orb, self.orb_vecs)

        # double check that everything went as planned

        # is the new vector perpendicular to all periodic directions?
        if new_latt_vec is None:
            for i in nnp_tb._per:
                if np.abs(np.dot(nnp_tb._lat[i], nnp_tb._lat[np_dir])) > 1.0e-6:
                    raise ValueError(
                        """\n\nThis shouldn't happen.  New nonperiodic vector 
                        is not perpendicular to periodic vectors!?"""
                    )
        # are cartesian coordinates of orbitals the same in two cases?
        for i in range(self._orb.shape[0]):
            cart_old = np.dot(self._lat.T, self._orb[i])
            cart_new = np.dot(nnp_tb._lat.T, nnp_tb._orb[i])
            if np.max(np.abs(cart_old - cart_new)) > 1.0e-6:
                raise Exception(
                    """\n\nThis shouldn't happen. New choice of nonperiodic vector
                        somehow changed Cartesian coordinates of orbitals."""
                )
        # check that volume of the cell is not zero
        if np.abs(np.linalg.det(nnp_tb._lat)) < 1.0e-6:
            raise Exception(
                "\n\nLattice with new choice of nonperiodic vector has zero volume?!"
            )

        # put orbitals to home cell if asked for
        if to_home:
            nnp_tb._shift_to_home(to_home_warning)

        # return new tb model
        return nnp_tb

    def make_supercell(
        self,
        sc_red_lat,
        return_sc_vectors: bool=False,
        to_home: bool=True,
        to_home_warning: bool=True,
    ) -> "TBModel":
        """Make model on a super-cell.

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

        Parameters
        ----------
        sc_red_lat : array-like
          Super-cell lattice vectors in terms of reduced coordinates
          of the original tight-binding model. Shape must be
          ``(dim_r, dim_r)``. First index in the array specifies super-cell vector,
          while second index specifies coordinate of that super-cell vector. 
          
          If `dim_k` < `dim_r` then still need to specify full array with
          size ``(dim_r, dim_r)`` for consistency, but non-periodic
          directions must have 0 on off-diagonal elements and 1 on
          diagonal.

        return_sc_vectors : bool, optional
            Default value is ``False``. If ``True`` returns also lattice vectors
            inside the super-cell. Internally, super-cell tight-binding model will
            have orbitals repeated in the same order in which these
            super-cell vectors are given, but if argument `to_home`
            is set ``True`` (which it is by default) then additionally,
            orbitals will be shifted to the home cell.

        to_home : bool, optional
            Default value is ``True``. If ``True`` will shift all orbitals
            to the home cell along non-periodic directions.

        to_home_warning : bool, optional
            Default value is ``True``. If ``True`` prints warning messages
            about orbitals being outside the home cell (reduced coordinate larger
            than 1 or smaller than 0 along non-periodic direction). 

            Note that setting this parameter to *True* or *False* has no effect on 
            resulting coordinates of the model. 

        Returns
        -------
        sc_tb : :class:`pythtb.TBModel`
            Tight-binding model in a super-cell.

        sc_vectors : :class:`numpy.ndarray`, optional
          Super-cell vectors, returned only if
          `return_sc_vectors` is set to ``True`` (default value is
          ``False``).

        Notes
        -----
        The super-cell is constructed by repeating the original unit cell
        according to the specified super-cell lattice vectors. The resulting
        model will have a larger Brillouin zone and may exhibit different
        electronic properties compared to the original model.

        Examples
        --------
        Create super-cell out of 2d tight-binding model ``tb``

        >>> sc_tb = tb.make_supercell([[2, 1], [-1, 2]])

        """

        # Can't make super cell for model without periodic directions
        if self._dim_r == 0:
            raise Exception(
                "\n\nMust have at least one periodic direction to make a super-cell"
            )

        # convert array to numpy array
        use_sc_red_lat = np.array(sc_red_lat)

        # checks on super-lattice array
        if use_sc_red_lat.shape != (self._dim_r, self._dim_r):
            raise Exception("\n\nDimension of sc_red_lat array must be dim_r*dim_r")
        if use_sc_red_lat.dtype != int:
            raise Exception("\n\nsc_red_lat array elements must be integers")
        for i in range(self._dim_r):
            for j in range(self._dim_r):
                if (i == j) and (i not in self._per) and use_sc_red_lat[i, j] != 1:
                    raise Exception(
                        "\n\nDiagonal elements of sc_red_lat for non-periodic directions must equal 1."
                    )
                if (
                    (i != j)
                    and ((i not in self._per) or (j not in self._per))
                    and use_sc_red_lat[i, j] != 0
                ):
                    raise Exception(
                        "\n\nOff-diagonal elements of sc_red_lat for non-periodic directions must equal 0."
                    )
        if np.abs(np.linalg.det(use_sc_red_lat)) < 1.0e-6:
            raise Exception(
                "\n\nSuper-cell lattice vectors length/area/volume too close to zero, or zero."
            )
        if np.linalg.det(use_sc_red_lat) < 0.0:
            raise Exception(
                "\n\nSuper-cell lattice vectors need to form right handed system."
            )

        # converts reduced vector in original lattice to reduced vector in super-cell lattice
        def to_red_sc(red_vec_orig):
            return np.linalg.solve(
                np.array(use_sc_red_lat.T, dtype=float),
                np.array(red_vec_orig, dtype=float),
            )

        # conservative estimate on range of search for super-cell vectors
        max_R = np.max(np.abs(use_sc_red_lat)) * self._dim_r

        # candidates for super-cell vectors
        sc_cands = [
            np.array(candidate)
            for candidate in product(range(-max_R, max_R + 1), repeat=self._dim_r)
        ]

        # find all vectors inside super-cell
        # store them here
        sc_vec = []
        eps_shift = (
            np.sqrt(2.0) * 1.0e-8
        )  # shift of the grid, so to avoid double counting
        #
        for vec in sc_cands:
            # compute reduced coordinates of this candidate vector in the super-cell frame
            tmp_red = to_red_sc(vec).tolist()
            # check if in the interior
            inside = True
            for t in tmp_red:
                if t <= -1.0 * eps_shift or t > 1.0 - eps_shift:
                    inside = False
            if inside:
                sc_vec.append(np.array(vec))
        # number of times unit cell is repeated in the super-cell
        num_sc = len(sc_vec)

        # check that found enough super-cell vectors
        if int(round(np.abs(np.linalg.det(use_sc_red_lat)))) != num_sc:
            raise Exception(
                "\n\nSuper-cell generation failed! Wrong number of super-cell vectors found."
            )

        # cartesian vectors of the super lattice
        sc_cart_lat = np.dot(use_sc_red_lat, self._lat)

        # orbitals of the super-cell tight-binding model
        sc_orb = []
        for cur_sc_vec in sc_vec:  # go over all super-cell vectors
            for orb in self._orb:  # go over all orbitals
                # shift orbital and compute coordinates in
                # reduced coordinates of super-cell
                sc_orb.append(to_red_sc(orb + cur_sc_vec))

        # create super-cell TBModel object to be returned
        sc_tb = TBModel(
            self._dim_k,
            self._dim_r,
            sc_cart_lat,
            sc_orb,
            per=self._per,
            nspin=self._nspin,
        )

        # remember if came from w90
        sc_tb._assume_position_operator_diagonal = (
            self._assume_position_operator_diagonal
        )

        # repeat onsite energies
        for i in range(num_sc):
            for j in range(self._norb):
                sc_tb.set_onsite(self._site_energies[j], i * self._norb + j)

        # set hopping terms
        for c, cur_sc_vec in enumerate(sc_vec):  # go over all super-cell vectors
            for h in range(
                len(self._hoppings)
            ):  # go over all hopping terms of the original model
                # amplitude of the hop is the same
                amp = self._hoppings[h][0]

                # lattice vector of the hopping
                ind_R = copy.deepcopy(self._hoppings[h][3])
                # super-cell component of hopping lattice vector
                # shift also by current super cell vector
                sc_part = np.floor(to_red_sc(ind_R + cur_sc_vec))  # round down!
                sc_part = np.array(sc_part, dtype=int)
                # find remaining vector in the original reduced coordinates
                orig_part = ind_R + cur_sc_vec - np.dot(sc_part, use_sc_red_lat)
                # remaining vector must equal one of the super-cell vectors
                pair_ind = None
                for p, pair_sc_vec in enumerate(sc_vec):
                    if False not in (pair_sc_vec == orig_part):
                        if pair_ind is not None:
                            raise Exception("\n\nFound duplicate super cell vector!")
                        pair_ind = p
                if pair_ind is None:
                    raise Exception("\n\nDid not find super cell vector!")

                # index of "from" and "to" hopping indices
                hi = self._hoppings[h][1] + c * self._norb
                hj = self._hoppings[h][2] + pair_ind * self._norb

                # add hopping term
                sc_tb.set_hop(
                    amp, hi, hj, sc_part, mode="add", allow_conjugate_pair=True
                )

        # put orbitals to home cell if asked for
        if to_home:
            sc_tb._shift_to_home(to_home_warning)

        # return new tb model and vectors if needed
        if not return_sc_vectors:
            return sc_tb
        else:
            return (sc_tb, sc_vec)

    def _shift_to_home(self, to_home_warning: bool=True):
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

        # create list of emty lists (one for each real-space direction)
        warning_list = [[]] * self._dim_r
        # go over all orbitals
        for i in range(self._norb):
            # find displacement vector needed to bring back to home cell
            disp_vec = np.zeros(self._dim_r, dtype=int)
            # shift only in periodic directions
            for k in range(self._dim_r):
                shift = np.floor(self._orb[i, k] + 1.0e-6).astype(int)
                if k in self._per:
                    disp_vec[k] = shift
                else:  # check for shift in non-periodic directions
                    if shift != 0:
                        warning_list[k] = warning_list[k] + [i]

        # print warning message if needed
        if to_home_warning:
            warn_str = ""
            for k in range(self._dim_r):
                orbs = warning_list[k]
                if orbs != []:
                    orb_str = ", ".join(str(e) for e in orbs)
                    warn_str += "  * Direction %1d : Orbitals " % k + orb_str + "\n"
            if warn_str != "":
                print(
                    "  "
                    + 69 * "-"
                    + "\n"
                    + """\
  WARNING from '_shift_to_home' (called by 'change_nonperiodic_vector'
  or 'make_supercell'): Orbitals are not "shifted to home" along
  non-periodic directions.  Older versions of PythTb (1.7.2 and older)
  allowed this, but it changes the physical nature of the tight-binding
  model.  PythTB 1.7.3 and newer versions of PythTb no longer shift
  orbitals along non-periodic directions.
  *
  In the present case, the following orbitals would have been assigned
  different coordinates in PythTb 1.7.2 and older:
  *\n"""
                    + warn_str
                    + """  *
  To prevent printing this warning, call 'change_nonperiodic_vector'
  or 'make_supercell' with 'to_home_warning=False'.
  *
  This warning message will be removed in future versions of PythTb.
"""
                    + "  "
                    + 69 * "-"
                    + "\n"
                )

            # shift orbitals
            self._orb[i] -= disp_vec
            # shift hoppings
            if self._dim_k != 0:
                for h in range(len(self._hoppings)):
                    if self._hoppings[h][1] == i:
                        self._hoppings[h][3] -= disp_vec
                    if self._hoppings[h][2] == i:
                        self._hoppings[h][3] += disp_vec

    def add_orb(self, coord):
        """Adds a new orbital to the model with the specified coordinates.
        
        The orbital coordinate must be given in reduced
        coordinates, i.e. in units of the real-space lattice vectors
        of the model. The new orbital is added at the end of the list
        of orbitals, and the orbital index is set to the next available
        index.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        coord : array_like, float
            The reduced coordinates of the new orbital of length `dim_r`. If
            `coord` is a single float or int, it will be converted to a 1D array 
            (`dim_r` must be 1).
        """
        if isinstance(coord, (float, int)):
            coord = np.array([coord], float)
        elif isinstance(coord, list):
            coord = np.array(coord, float)
        elif not isinstance(coord, np.ndarray):
            raise TypeError(f"Expected array_like or float, got {type(coord)}")

        if coord.shape != (self.dim_r, ):
                raise ValueError(
                    f"Orbital coordinate must be a list of length {self._dim_r}, got {coord.shape}"
                )
        
        # Append orbital coordinate
        self._orb = np.vstack([self._orb, coord])
        # Update number of orbitals and states
        self._norb += 1
        self._nstate = self._norb * self._nspin
        # Append default site energy and specified flag
        if self._nspin == 1:
            self._site_energies = np.append(self._site_energies, 0.0)
        else:
            new_block = np.zeros((1, 2, 2), dtype=complex)
            self._site_energies = np.vstack([self._site_energies, new_block])
        self._site_energies_specified = np.append(self._site_energies_specified, False)
        # No hoppings are added by default

    def remove_orb(self, to_remove):
        r"""Removes specified orbitals from the model.

        Parameters
        ----------
        to_remove : array-like or int
            List of orbital indices to be removed, or index of single orbital to be removed

        Returns
        -------
        del_tb : class:`pythtb.TBModel`
            Model with removed orbitals.

        Notes
        -----
        Removing orbitals will reindex the orbitals with indices higher
        than those that are removed. For example, if model has 6 orbitals
        and you remove the 2nd orbital, then the orbitals 3-6 will be
        reindexed to 1-4 (Python counting). Indices of first two orbitals (0 and 1) 
        are unaffected.
         
        Examples
        --------
        If original_model has say 10 orbitals then returned small_model will 
        have only 8 orbitals.

        >>> small_model = original_model.remove_orb([2,5])

        """

        # if a single integer is given, convert to a list with one element
        if _is_int(to_remove):
            orb_index = [to_remove]
        else:
            orb_index = copy.deepcopy(to_remove)

        # check range of indices
        for i, orb_ind in enumerate(orb_index):
            if orb_ind < 0 or orb_ind > self._norb - 1 or (not _is_int(orb_ind)):
                raise Exception("\n\nSpecified wrong orbitals to remove!")
        for i, ind1 in enumerate(orb_index):
            for ind2 in orb_index[i + 1 :]:
                if ind1 == ind2:
                    raise Exception("\n\nSpecified duplicate orbitals to remove!")

        # put the orbitals to be removed in desceding order
        orb_index = sorted(orb_index, reverse=True)

        # make copy of a model
        ret = copy.deepcopy(self)

        # adjust some variables in the new model
        ret._norb -= len(orb_index)
        ret._nstate -= len(orb_index) * self._nspin
        # remove indices one by one
        for i, orb_ind in enumerate(orb_index):
            # adjust variables
            ret._orb = np.delete(ret._orb, orb_ind, 0)
            ret._site_energies = np.delete(ret._site_energies, orb_ind, 0)
            ret._site_energies_specified = np.delete(
                ret._site_energies_specified, orb_ind
            )
            # adjust hopping terms (in reverse)
            for j in range(len(ret._hoppings) - 1, -1, -1):
                h = ret._hoppings[j]
                # remove all terms that involve this orbital
                if h[1] == orb_ind or h[2] == orb_ind:
                    del ret._hoppings[j]
                else:  # otherwise modify term
                    if h[1] > orb_ind:
                        ret._hoppings[j][1] -= 1
                    if h[2] > orb_ind:
                        ret._hoppings[j][2] -= 1
        # return new model
        return ret

    def k_uniform_mesh(self, mesh_size):
        """Uniform grid of k-points in reduced coordinates.

        The mesh along each direction is defined from [0, 1). 
        The mesh always contains the origin.

        Parameters
        ----------
        mesh_size : array_like
            Number of k-points in the mesh in each periodic direction of the model.

        Returns
        -------
        k_vec : np.ndarray
          Array of k-vectors on the mesh that can be directly passed to function 
          :func:`pythtb.TBModel.solve_all`. The shape of the array is 
          (nk1, nk2, ..., dim_k) where nk1, nk2, ... are the number of k-points
          in each direction defined by `mesh_size`.

        Notes
        -----
        The uniform grid of k-points that can be passed to
        function :func:`pythtb.TBModel.solve_all`. 

        Examples
        --------
        Returns a 10x20x30 mesh of a tight binding model 
        with three periodic directions

        >>> k_vec = my_model.k_uniform_mesh([10,20,30])
        >>> print(k_vec.shape)
        (10, 20, 30, 3)

        Solve model on the uniform mesh

        >>> my_model.solve_ham(k_vec)

        """
        use_mesh = np.array(list(map(round, mesh_size)), dtype=int)
        if use_mesh.shape != (self.dim_k,):
            print(use_mesh.shape)
            raise Exception("\n\nIncorrect size of the specified k-mesh!")
        if np.min(use_mesh) <= 0:
            raise Exception("\n\nMesh must have positive non-zero number of elements.")

        axes = [np.linspace(0, 1, n, endpoint=False) for n in use_mesh]
        mesh = np.meshgrid(*axes, indexing="ij")
        k_vec = np.stack(mesh, axis=-1).reshape(-1, len(use_mesh))

        return k_vec

    def k_path(self, kpts, nk:int, report:bool=True):
        r"""Interpolates a path in reciprocal space.

        Interpolates a path in reciprocal space between specified
        k-points. In 2D or 3D the k-path can consist of several
        straight segments connecting high-symmetry points ("nodes"),
        and the results can be used to plot the bands along this path.

        The interpolated path that is returned contains as
        equidistant k-points as possible.

        Parameters
        ----------

        kpts : array-like, str
          Array of k-vectors in reciprocal space between
          which interpolated path should be constructed. These
          k-vectors must be given in reduced coordinates.  As a
          special case, in 1D k-space kpts may be a string:

          - `"full"`: Implies  ``[0.0, 0.5, 1.0]`` (full BZ)
          - `"fullc"`: Implies  ``[-0.5, 0.0, 0.5]`` (full BZ, centered)
          - `"half"`: Implies  ``[ 0.0, 0.5]``  (half BZ)

        nk : int
            Total number of k-points to be used in making the plot.

        report : bool, optional
            Optional parameter specifying whether printout
            is desired (default is True).

        Returns
        -------
        k_vec : np.ndarray
            Array of (nearly) equidistant interpolated k-points. 

        k_dist : np.ndarray
            Array giving accumulated k-distance to each
            k-point in the path. This array can be used to plot path in
            the k-space so that the distances between the k-points in
            the plot are exact.

        k_node : np.ndarray
            Array giving accumulated k-distance to each
            node on the path in Cartesian coordinates. This array is
            typically used to plot nodes (typically special points) on
            the path in k-space.

        Notes
        -----
        - The distance between the points is calculated in the Cartesian frame,
          however coordinates themselves are given in dimensionless reduced coordinates!  
          This is done so that this array can be directly passed to function
          :func:`pythtb.TBModel.solve_ham`.
        - Unlike array `k_vec`, `k_dist` has dimensions! Units are defined here
          so that for a one-dimensional crystal with lattice constant equal to 
          for example `10` the length of the Brillouin zone would equal
          `1/10=0.1`. In other words factors of :math:`2\pi` are
          absorbed into `kpts`.

        Examples
        ---------
        Construct a path connecting four nodal points in k-space
        Path will contain 401 k-points, roughly equally spaced

        >>> path = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.5], [0.0, 0.0]]
        >>> (k_vec, k_dist, k_node) = my_model.k_path(path,401)
        
        Solve for eigenvalues on that path

        >>> evals = tb.solve_all(k_vec)
        """
        dim = self.dim_k

        # Parse kpts and validate
        k_list = _parse_kpts(kpts, dim)
        if k_list.shape[1] != dim:
            raise ValueError(
                f"Dimension mismatch: kpts shape {k_list.shape}, model dim {dim}"
            )
        if nk < len(k_list):
            raise ValueError("nk must be >= number of nodes in kpts")

        # Extract periodic lattice and compute k-space metric
        lat_per = self.lat_vecs[self.per]
        k_metric = np.linalg.inv(lat_per @ lat_per.T)

        # Compute segment vectors and lengths in Cartesian metric
        diffs = k_list[1:] - k_list[:-1]
        seg_lengths = np.sqrt(np.einsum("ij,ij->i", diffs @ k_metric, diffs))

        # Accumulated node distances
        k_node = np.concatenate(([0.0], np.cumsum(seg_lengths)))

        # Determine indices in the final array corresponding to each node
        node_index = np.rint(k_node / k_node[-1] * (nk - 1)).astype(int)

        # Initialize output arrays
        k_vec = np.empty((nk, dim))
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
            _report(k_list, lat_per, k_metric, k_node, node_index)

        return k_vec, k_dist, k_node    
    

    def ignore_position_operator_offdiagonal(self):
        """Set flag to ignore off-diagonal elements of the position operator.

        Call to this function enables one to approximately compute
        Berry-like objects from tight-binding models that were
        obtained from Wannier90.
        """
        self._assume_position_operator_diagonal = True

    def position_matrix(self, evec: np.ndarray, dir: int):
        r"""Position operator matrix elements

        Returns matrix elements of the position operator along
        direction `dir` for eigenvectors `evec` at a single k-point.
        Position operator is defined in reduced coordinates.

        The returned object :math:`X` is

        .. math::

          X_{m n {\bf k}}^{\alpha} = \langle u_{m {\bf k}} \vert
          r^{\alpha} \vert u_{n {\bf k}} \rangle

        Here :math:`r^{\alpha}` is the position operator along direction
        :math:`\alpha` that is selected by `dir`.

        Parameters
        ----------
        evec : np.ndarray
            Eigenvectors for which we are computing matrix
            elements of the position operator.  The shape of this array
            is ``evec[band, orbital]`` if `nspin` = 1 and
            ``evec[band, orbital, spin]`` if `nspin` = 2.

        dir : int
            Direction along which we are computing the center.
            This integer must not be one of the periodic directions
            since position operator matrix element in that case is not
            well defined.

        Returns
        -------
        pos_mat : np.ndarray
            Position operator matrix :math:`X_{m n}` as defined above. 
            This is a square matrix with size determined by number of bands
            given in `evec` input array.  First index of `pos_mat` corresponds to
            bra vector (:math:`m`) and second index to ket (:math:`n`).

        See Also
        --------
        :ref:`haldane-hwf-nb` : For an example.

        Examples
        --------
        Diagonalizes Hamiltonian at some k-points

        >>> (evals, evecs) = my_model.solve_ham(k_vec, return_eigvecs=True)

        Computes position operator matrix elements for 3-rd kpoint
        and bottom five bands along first coordinate

        >>> pos_mat = my_model.position_matrix(evecs[2, :5], 0)

        """

        # make sure specified direction is not periodic!
        if dir in self._per:
            raise Exception(
                "Can not compute position matrix elements along periodic direction!"
            )
        # make sure direction is not out of range
        if dir < 0 or dir >= self._dim_r:
            raise Exception("Direction out of range!")

        # check if model came from w90
        if not self._assume_position_operator_diagonal:
            _offdiag_approximation_warning_and_stop()

        # check shape of evec
        if not isinstance(evec, np.ndarray):
            raise TypeError("evec must be a numpy array.")
        # check number of dimensions of evec
        if self.nspin == 1:
            if evec.ndim != 2:
                raise ValueError(
                    "evec must be a 2D array with shape (band, orbital) for spinless models."
                )
        elif self.nspin == 2:
            if evec.ndim != 3:
                raise ValueError(
                    "evec must be a 3D array with shape (band, orbital, spin) for spinful models."
                )

        # get coordinates of orbitals along the specified direction
        pos_tmp = self._orb[:, dir]
        # reshape arrays in the case of spinfull calculation
        if self._nspin == 2:
            # tile along spin direction if needed
            pos_use = np.tile(pos_tmp, (2, 1)).transpose().flatten()
            evec_use = evec.reshape((evec.shape[0], evec.shape[1] * evec.shape[2]))
        else:
            pos_use = pos_tmp
            evec_use = evec

        # position matrix elements
        pos_mat = np.zeros((evec_use.shape[0], evec_use.shape[0]), dtype=complex)
        # go over all bands
        for i in range(evec_use.shape[0]):
            for j in range(evec_use.shape[0]):
                pos_mat[i, j] = np.dot(evec_use[i].conj(), pos_use * evec_use[j])

        # make sure matrix is Hermitian
        if not np.allclose(pos_mat, pos_mat.T.conj()):
            raise ValueError("Position matrix is not Hermitian.")

        return pos_mat

    def position_expectation(self, evec: np.ndarray, dir: int):
        r"""Returns diagonal matrix elements of the position operator.
        
        These elements :math:`X_{n n}` can be interpreted as an
        average position of n-th Bloch state ``evec[n]`` along
        direction `dir`. 

        Parameters
        ----------
        evec : np.ndarray
            Eigenvectors for which we are computing matrix
            elements of the position operator. The shape of this array
            is ``evec[band, orbital]`` if ``nspin=1`` and
            ``evec[band, orbital, spin]`` if ``nspin=2``.

        dir : int
            Direction along which we are computing matrix
            elements. This integer must not be one of the periodic
            directions since position operator matrix element in that
            case is not well defined.

        Returns
        -------
        pos_exp : np.ndarray
            Diagonal elements of the position operator matrix :math:`X`.
            Length of this vector is determined by number of bands given in *evec* input
            array.
        
        See Also
        --------
        :ref:`haldane-hwf-nb` : For an example.
        position_matrix : For definition of matrix :math:`X`.

        Notes
        -----
        Generally speaking these centers are _not_
        hybrid Wannier function centers (which are instead
        returned by :func:`TBModel.position_hwf`).

        Examples
        --------
        Diagonalizes Hamiltonian at some k-points
          
        >>> (evals, evecs) = my_model.solve_ham(k_vec, return_eigvecs=True)
        
        Computes average position for 3-rd kpoint
        and bottom five bands along first coordinate
        
        >>> pos_exp = my_model.position_expectation(evecs[2, :5], 0)

        """

        # check if model came from w90
        if not self._assume_position_operator_diagonal:
            _offdiag_approximation_warning_and_stop()

        pos_exp = self.position_matrix(evec, dir).diagonal()
        return np.array(np.real(pos_exp), dtype=float)

    def position_hwf(self, evec, dir, hwf_evec=False, basis="orbital"):
        r"""Eigenvalues and eigenvectors of the position operator

        Returns eigenvalues and optionally eigenvectors of the
        position operator matrix :math:`X` in basis of the orbitals
        or, optionally, of the input wave functions (typically Bloch
        functions). The returned eigenvectors can be interpreted as
        linear combinations of the input states ``evec`` that have
        minimal extent (or spread :math:`\Omega` in the sense of
        maximally localized Wannier functions) along direction
        ``dir``. The eigenvalues are average positions of these
        localized states.

        Parameters
        ----------
        evec : np.ndarray
            Eigenvectors for which we are computing matrix
            elements of the position operator. The shape of this array
            is ``evec[band, orbital]`` if ``nspin=1`` and
            ``evec[band, orbital, spin]`` if ``nspin=2``.

        dir : int
            Direction along which we are computing matrix
            elements. This integer must not be one of the periodic
            directions since position operator matrix element in that
            case is not well defined.

        hwf_evec : bool, optional
            Default is ``False``. If set to ``True`` this function will
            return not only eigenvalues but also eigenvectors of :math:`X`. 

        basis : {"orbital", "wavefunction", "bloch"}, optional
            Default is "orbital". If ``basis="wavefunction"`` or ``basis="bloch"``, the hybrid
            Wannier function `hwf` is returned in the basis of the input
            wave functions. That is, the elements of ``hwf[i, j]`` give the amplitudes
            of the i-th hybrid Wannier function on the j-th input state.
            If ``basis="orbital"``, the elements of ``hwf[i, orb]`` (or ``hwf[i, orb, spin]``
            if ``nspin=2``) give the amplitudes of the i-th hybrid Wannier function on
            the specified basis function. 

        Returns
        -------
        hwfc : np.ndarray
            Eigenvalues of the position operator matrix :math:`X`
            (also called hybrid Wannier function centers).
            Length of this vector equals number of bands given in ``evec``
            input array. Hybrid Wannier function centers are ordered in ascending order.
            Note that in general `n`-th hwfc does not correspond to `n`-th
            state in ``evec``.

        hwf : np.ndarray
            Eigenvectors of the position operator matrix :math:`X`.
            (also called hybrid Wannier functions).  These are returned only if
            parameter ``hwf_evec = True``.

            The shape of this array is ``[h,x]`` or ``[h,x,s]`` depending on value of
            ``basis`` and ``nspin``.

            - If ``basis`` is "bloch" then ``x`` refers to indices of
              Bloch states.
            - If ``basis`` is "orbital" then ``x`` (or ``x`` and ``s``)
              correspond to orbital index (or orbital and spin index if ``nspin=2``).

        See Also
        --------
        :ref:`haldane-hwf-nb` : For an example.
        position_matrix : For the definition of the matrix :math:`X`.
        position_expectation : For the position expectation value.

        Notes
        -----
        Note that these eigenvectors are not maximally localized
        Wannier functions in the usual sense because they are
        localized only along one direction. They are also not the
        average positions of the Bloch states `evec`, which are
        instead computed by :func:`position_expectation`.

        See Fig. 3 in [1]_ for a discussion of the hybrid Wannier function centers in the
        context of a Chern insulator.

        References
        ----------
        .. [1]  S. Coh, D. Vanderbilt, Phys. Rev. Lett. 102, 107603 (2009).

        Examples
        --------
        Diagonalizes Hamiltonian at some k-points

        >>> evals, evecs = my_model.solve_ham(k_vec, return_eigvecs=True)

        Computes hybrid Wannier centers (and functions) for 3-rd kpoint
        and bottom five bands along first coordinate

        >>> hwfc, hwf = my_model.position_hwf(evecs[2, :5], 0, hwf_evec=True, basis="orbital")
        """
        # check if model came from w90
        if not self._assume_position_operator_diagonal:
            _offdiag_approximation_warning_and_stop()

        # get position matrix
        pos_mat = self.position_matrix(evec, dir)

        # diagonalize
        if not hwf_evec:
            hwfc = np.linalg.eigvalsh(pos_mat)
            return hwfc
        else:  # find eigenvalues and eigenvectors
            (hwfc, hwf) = np.linalg.eigh(pos_mat)
            # transpose matrix eig since otherwise it is confusing
            # now eig[i,:] is eigenvector for eval[i]-th eigenvalue
            hwf = hwf.T
            # convert to right basis
            if basis.lower().strip() in ["wavefunction", "bloch"]:
                return (hwfc, hwf)
            elif basis.lower().strip() == "orbital":
                if self._nspin == 1:
                    ret_hwf = np.zeros((hwf.shape[0], self._norb), dtype=complex)
                    # sum over bloch states to get hwf in orbital basis
                    for i in range(ret_hwf.shape[0]):
                        ret_hwf[i] = np.dot(hwf[i], evec)
                    hwf = ret_hwf
                else:
                    ret_hwf = np.zeros((hwf.shape[0], self._norb * 2), dtype=complex)
                    # get rid of spin indices
                    evec_use = evec.reshape([hwf.shape[0], self._norb * 2])
                    # sum over states
                    for i in range(ret_hwf.shape[0]):
                        ret_hwf[i] = np.dot(hwf[i], evec_use)
                    # restore spin indices
                    hwf = ret_hwf.reshape([hwf.shape[0], self._norb, 2])
                return (hwfc, hwf)
            else:
                raise Exception(
                    "\n\nBasis must be either 'wavefunction', 'bloch', or 'orbital'"
                )

    def berry_curvature(
        self,
        k_pts,
        evals=None,
        evecs=None,
        occ_idxs=None,
        dirs="all",
        cartesian: bool = False,
        abelian: bool = True,
    ):
        r"""Compute the Berry curvature at a list of k-points.

        The Berry curvature is computed from the derivatives of the Bloch Hamiltonian
        :math:`\partial_\mu H_k`, where :math:`\mu` is the direction in k-space.
        
        Specifically, for :math:`(m,n) \in \text{occ}`,

        .. math::

            \Omega_{\mu \nu;\ mn}(k) =  i\sum_{l \notin \text{occ}}
            \frac{
                \langle u_{mk} | \partial_{\mu} H_k | u_{lk} \rangle
                \langle u_{lk} | \partial_{\nu} H_k | u_{nk} \rangle
                -
                \langle u_{mk} | \partial_{\nu} H_k| u_{lk} \rangle
                \langle u_{lk} | \partial_{\mu} H_k | u_{nk} \rangle
            }{
                (E_{nk} - E_{lk})(E_{mk} - E_{lk})
            }

        Parameters
        ----------
        k_pts : (Nk, dim_k) array-like
            Array of k-points with shape (Nk, dim_k), where Nk is the number of points
            and dim_k is the dimensionality of the k-space.
        evals : (Nk, n_states) array, optional
            Eigenvalues of the Hamiltonian at the k-points. If not provided, they will be computed.
        evecs : (Nk, n_states, n_orb) array, optional
            Eigenvectors of the Hamiltonian. If not provided, they will be computed.
        occ_idxs : 1D array, optional
            Indices of the occupied bands. Defaults to the first half of the states.
        dirs : str or tuple of int, optional
            Directions in k-space for which to compute the curvature.
            If "all", computes all components. If a tuple, restricts to specified indices.
        cartesian : bool, optional
            If True, computes the velocity operator in Cartesian coordinates.
            Default is False (reduced coordinates).
        abelian : bool, optional
            If True, returns the trace of the Berry curvature tensor (abelian case).
            If False, returns the full tensor.

        Returns
        -------
        b_curv : np.ndarray
            Berry curvature tensor. If ``dirs`` is "all", shape is (dim_k, dim_k, Nk, n_orb, n_orb).
            If ``dirs`` is a tuple, shape is (Nk, n_orb, n_orb) and the returned tensor is restricted 
            to the specified directions.
            If ``abelian`` is True, returns the band-trace of the Berry curvature tensor and the last
            two dimensions are not present.

        Notes
        -----
        This quantity is an anti-symmetric under :math:`\mu \leftrightarrow \nu`.
        """

        if self.dim_k < 2:
            raise Exception(
                """
                Berry curvature in this context is only computed for k-space dimensions. 
                Must have dim_k >= 2.
                """
            )

        v_k = self.grad_ham(k_pts, cartesian=cartesian)  # (Nk, dim_k, n_orb, n_orb)
        # flatten spin axis if present
        new_shape = (v_k.shape[:2]) + (self._nstate, self._nstate)
        v_k = v_k.reshape(*new_shape)

        if evals is None or evecs is None:
            evals, evecs = self.solve_ham(
                k_pts, return_eigvecs=True, keep_spin_ax=False
            )

        n_eigs = evecs.shape[-2]

        # Identify occupied bands
        if occ_idxs is None:
            occ_idxs = np.arange(n_eigs // 2)
        else:
            occ_idxs = np.array(occ_idxs)

        # Identify conduction bands as remainder of band indices (assumes gapped)
        cond_idxs = np.setdiff1d(np.arange(n_eigs), occ_idxs)

        # All pairs of energy differences
        delta_E = (
            evals[..., np.newaxis, :] - evals[..., :, np.newaxis]
        )  # shape (Nk, n_states, n_states)
        # Divide by energy differences, diagonals are ignored
        with np.errstate(
            divide="ignore", invalid="ignore"
        ):  # Suppress divide by zero warnings
            inv_delta_E = np.where(delta_E != 0, 1 / delta_E, 0)

        # newaxis for Cartesian direction broadcasting
        evecs_conj = evecs.conj()[np.newaxis, :, :, :]
        # transpose
        evecs_T = evecs.transpose(0, 2, 1)[np.newaxis, :, :, :]
        # project vk into energy eignvector basis
        vk_evecT = np.matmul(v_k, evecs_T)  # intermediate array
        v_k_rot = np.matmul(evecs_conj, vk_evecT)  # (dim_k, n_kpts, n_orb, n_orb)

        # Extract relevant submatrices
        # top right
        v_occ_cond = v_k_rot[..., occ_idxs, :][
            ..., :, cond_idxs
        ]  # shape (dim_k, Nk, n_occ, n_con)
        # bottom left
        v_cond_occ = v_k_rot[..., cond_idxs, :][
            ..., :, occ_idxs
        ]  # shape (dim_k, Nk, n_con, n_occ)
        # top right (bottom left uneeded in Kubo formula)
        delta_E_occ_cond = inv_delta_E[:, occ_idxs, :][
            :, :, cond_idxs
        ]  # shape (Nk, n_con, n_occ)

        # premultiply by energy denominators
        v_occ_cond = v_occ_cond * delta_E_occ_cond
        v_cond_occ = v_cond_occ * delta_E_occ_cond.swapaxes(-1, -2)

        # Berry curvature shape: (dim_k, dim_k, n_kpts, n_orb, n_orb)
        # Where m is conduction indices, and n,l are occupied indices
        # <unk|v_mu|umk> <umk|v_nu|ulk> - <unk|v_nu|umk> <umk|v_mu|ulk> / (Enk - Emk)(Elk - Emk)
        b_curv = 1j * (
            np.matmul(v_occ_cond[:, None], v_cond_occ[None, :])
            - np.matmul(v_occ_cond[None, :], v_cond_occ[:, None])
        )

        if abelian:
            b_curv = np.trace(b_curv, axis1=-1, axis2=-2)
        if dirs == "all":
            return b_curv
        else:
            return b_curv[dirs]

    def chern(self, occ_idxs=None, dirs=(0, 1), nk=200):
        """Computes Chern number for occupied manifold.

        Parameters
        ----------
        occ_idxs : array-like, optional
            Occupied band indices. If none are provided, 
            the lower half bands are considered occupied.

        dirs : tuple
            Indices for reciprocal space directions defining
            2d surface to integrate Berry flux.

        Returns
        -------
        float
            Chern number for the occupied manifold.
        """

        nks = (nk,) * self.dim_k
        k_grid = self.k_uniform_mesh(nks)
        k_flat = k_grid.reshape(-1, self.dim_k)
        
        Omega = self.berry_curvature(k_flat, occ_idxs=occ_idxs)

        Nk = Omega.shape[2]
        dk_sq = 1 / Nk
        Chern = np.sum(Omega[dirs]) * dk_sq / (2 * np.pi)

        return Chern.real

    def local_chern_marker(self, occ_idxs=None):
        r"""Bianco–Resta local Chern marker.

        The local Chern marker is a per-site quantity that captures the
        topological character of the occupied manifold in real space.
        It is defined as

        .. math::
            C_i = -4\pi \, \mathrm{Im} [P[X,P][Y,P]]_{ii},

        where P is the projector onto occupied states, and X,Y are position
        operators.

        Returns
        -------
        C_local : np.ndarray of shape (N,)
            Per-site local Chern marker
        """
        if self.dim_k != 0:
            raise ValueError("Local Chern marker is only defined for real-space models (dim_k=0).")
        
        H = self.hamiltonian()
        coords = self.get_orb(cartesian=True)
        uc_vol = self.get_cell_vol()

        N = H.shape[0]

        # coords: x, y (Cartesian)
        if isinstance(coords, tuple) and len(coords) == 2:
            x = np.asarray(coords[0], float).reshape(N)
            y = np.asarray(coords[1], float).reshape(N)
        else:
            coords = np.asarray(coords, float)
            if coords.ndim != 2 or coords.shape != (N, 2):
                raise ValueError("coords must be (N,2) or a tuple (x,y) of length N.")
            x, y = coords[:, 0], coords[:, 1]

        # Dense eigensolve and projector
        evals, evecs = np.linalg.eigh(H)  # returns sorted ascending
        if occ_idxs is None:
            # Default to half filling (robust for particle-hole symmetric models like Haldane).
            occ_idxs = np.arange(N // 2)

        Uocc = evecs[:, occ_idxs]  # (N, k_occ)
        P = Uocc @ Uocc.conj().T  # (N,N) dense projector

        # Position operators (dense diagonals)
        X = np.diag(x.astype(complex))
        Y = np.diag(y.astype(complex))

        # Commutators (explicit dense)
        XP = X @ P
        PX = P @ X
        YP = Y @ P
        PY = P @ Y
        CX = XP - PX
        CY = YP - PY

        # A = P [X,P] [Y,P]
        A = P @ (CX @ CY)

        # Local marker from diagonal of A
        C_local = 4 * np.pi * np.diag(np.imag(A)) / uc_vol
        return C_local
    

    ##### Plotting functions #####
    # These plotting functions are wrappers to the functions in plotting.py
    def visualize(
        self,
        proj_plane=None,
        eig_dr=None,
        draw_hoppings=True,
        annotate_onsite_en=False,
        ph_color="black",
    ):
        r"""Visualizes the tight-binding model geometry.

        Plots the tight-binding orbitals, hopping between tight-binding orbitals, 
        and optionally the electron eigenstates.

        If eigenvector is not drawn, then orbitals in home cell are drawn
        as red circles, and those in neighboring cells are drawn with
        a lighter shade of red. Hopping term directions are drawn with
        green lines connecting two orbitals. Origin of unit cell is
        indicated with blue dot, while real space unit vectors are drawn
        with blue lines.

        If eigenvector is drawn, then electron eigenstate on each orbital
        is drawn with a circle whose size is proportional to wavefunction
        amplitude while its color depends on the phase. There are various
        coloring schemes for the phase factor; see more details under
        `ph_color` parameter. If eigenvector is drawn and coloring scheme
        is "red-blue" or "wheel", all other elements of the picture are
        drawn in gray or black.

        Parameters
        ----------
        proj_plane : tuple or list of two integers
            Cartesian coordinates to be used for plotting. For example,
            if ``proj_plane=(0,1)`` then x-y projection of the model is
            drawn. This only should be specified if `dim_r` > 2.

        eig_dr : Optional parameter specifying eigenstate to
          plot. If specified, this should be one-dimensional array of
          complex numbers specifying wavefunction at each orbital in
          the tight-binding basis. If not specified, eigenstate is not
          drawn.

        draw_hoppings : Optional parameter specifying whether to
          draw all allowed hopping terms in the tight-binding
          model. Default value is True.

        ph_color : {"black", "red-blue", "wheel"}, optional
            Determines the way the eigenvector phase factors are 
            translated into color. Default value is "black".

            - "black" -- phase of eigenvectors are ignored and wavefunction
              is always colored in black.

            - "red-blue" -- zero phase is drawn red, while phases or :math:`\pi` or
              :math:`-\pi` are drawn blue. Phases in between are interpolated between
              red and blue. Some phase information is lost in this coloring
              because phase of :math:`\pm \pi` have the same color.

            - "wheel" -- each phase is given unique color. In steps of :math:`\pi/3`
              starting from 0, colors are assigned (in increasing hue) as:
              red, yellow, green, cyan, blue, magenta, red.

        Returns
        -------
            fig : matplotlib.figure.Figure
                Figure object from matplotlib.pyplot module
            ax : matplotlib.axes.Axes
                Axes object from matplotlib.pyplot module

        Notes
        -----
        - This function is intended for visualizing tight-binding models
          in two dimensions. For three-dimensional visualizations, consider using
          the :func:`visualize_3d` method.
        - Convention of the wavefunction phase is as
          in convention 1 in section 3.1 of :download:`notes on
          tight-binding formalism  </misc/pythtb-formalism.pdf>`. In
          other words, these wavefunction phases are in correspondence
          with cell-periodic functions :math:`u_{n {\bf k}} ({\bf r})`
          not :math:`\Psi_{n {\bf k}} ({\bf r})`.

        Examples
        --------
        Draws x-y projection of tight-binding model
        tweaks figure and saves it as a PDF.
        
        >>> fig, ax = tb.visualize(0, 1)
        >>> plt.show()

        See Also
        --------
        - :ref:`haldane-edge-nb`,
        - :ref:`visualize-nb`.

        """
        return plot_tb_model(
            self, proj_plane, eig_dr, draw_hoppings, annotate_onsite_en, ph_color
        )

    def visualize_3d(
        self,
        eig_dr=None,
        draw_hoppings=True,
        site_colors=None,
        site_names=None,
        show_model_info=True,
        ph_color="black",
    ):
        r"""Visualize a 3D tight-binding model using ``Plotly``.

        This function creates an interactive 3D plot of your tight-binding model,
        showing the unit-cell origin, lattice vectors (with arrowheads), orbitals,
        hopping lines, and (optionally) an eigenstate overlay with marker sizes
        proportional to amplitude and colors reflecting the phase.

        Parameters
        ----------
        eig_dr : 
            Optional eigenstate (1D array of complex numbers) to display.
        draw_hoppings : bool, optional
            Whether to draw hopping lines between orbitals.
        annotate_onsite_en: bool, optional
            Whether to annotate orbitals with onsite energies.
        ph_color: str, optional
            Coloring scheme for eigenstate phases (e.g. "black", "red-blue", "wheel").

        Returns
        -------
        plotly.graph_objs.Figure
        """
        return plot_tb_model_3d(
            self,
            eig_dr=eig_dr,
            draw_hoppings=draw_hoppings,
            show_model_info=show_model_info,
            ph_color=ph_color,
            site_colors=site_colors,
            site_names=site_names,
        )

    def plot_bands(
        self,
        k_path,
        nk=101,
        k_label=None,
        proj_orb_idx=None,
        proj_spin=False,
        fig=None,
        ax=None,
        title=None,
        scat_size=3,
        lw=2,
        lc="b",
        ls="solid",
        cmap="plasma",
        show=False,
        cbar=True,
    ):
        """Plot the band structure along a specified path in k-space.

        This function allows for customization of the plot, including projection of orbitals,
        spin projection, figure and axis objects, title, scatter size, line width,
        line color, line style, colormap, and whether to show a color bar.

        Parameters
        ----------
        k_path : list
            List of high symmetry points to plot bands through
        nk : int, optional
            Number of k-points to sample along the path. Defaults to 101.
        k_label : list[str], optional
            Labels of high symmetry points. Defaults to None.
        proj_orb_idx : list[int], optional
            List of orbital indices to project onto. Defaults to None.
            This will give the bands a colorscale indicating the weight of 
            the Bloch states onto the list of orbitals.
        proj_spin : bool, optional
            Whether to project the spin components. Defaults to ``False``.
            If ``True``, the bands will be colored according to their spin character.
        fig : matplotlib.figure.Figure, optional
            Figure object to plot on. Defaults to None.
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. Defaults to None.
        title : str, optional
            Title of the plot. Defaults to None.
        scat_size : float, optional
            Size of the scatter points. Defaults to 3.
        lw : float, optional
            Line width of the band lines. Defaults to 2.
        lc : str, optional
            Line color of the band lines. Defaults to "b". Irrelevant
            if `proj_spin` is True or `proj_orb_idx` is not None.
        ls : str, optional
            Line style of the band lines. Defaults to "solid".
            Irrelevant if `proj_spin` is True or `proj_orb_idx` is not None.
        cmap : str, optional
            Colormap for the band plot. Defaults to "plasma". Only relevant if
            `proj_spin` is True or `proj_orb_idx` is not None.
        show : bool, optional
            Whether to show the plot. Defaults to False.
        cbar : bool, optional
            Whether to show a color bar. Defaults to True.
            Only relevant if `proj_spin` is True or `proj_orb_idx` is not None.

        Returns:
            fig : matplotlib.figure.Figure
            ax: matplotlib.axes.Axes
        """
        return plot_bands(
            self,
            k_path,
            nk=nk,
            k_label=k_label,
            proj_orb_idx=proj_orb_idx,
            proj_spin=proj_spin,
            fig=fig,
            ax=ax,
            title=title,
            scat_size=scat_size,
            lw=lw,
            lc=lc,
            ls=ls,
            cmap=cmap,
            show=show,
            cbar=cbar,
        )
