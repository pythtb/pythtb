import copy
import logging
import warnings
import numpy as np
from .plotting import plot_bands, plot_tb_model, plot_tb_model_3d
from .utils import _offdiag_approximation_warning_and_stop, is_Hermitian, deprecated, copydoc
from .lattice import Lattice
from .hoptable import HoppingTable

# set up logging
logger = logging.getLogger(__name__)

# what is exported when "from pythtb import *" is used
__all__ = ["TBModel", "tb_model"]

# Pauli matrices
SIGMA0 = np.array([[1, 0], [0, 1]], dtype=complex)
SIGMAX = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMAY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMAZ = np.array([[1, 0], [0, -1]], dtype=complex)

try:
    from tensorflow import convert_to_tensor
    from tensorflow.linalg import eigvalsh as tf_eigvalsh, eigh as tf_eigh
    from tensorflow import complex64 as tf_complex64, complex128 as tf_complex128
except ImportError:  # TensorFlow not installed – keep optional
    tf_eigvalsh = tf_eigh = tf_complex64 = tf_complex128 = None

def _tensorflow_solve(ham, *, return_eigvecs: bool, use_32_bit: bool):
    if convert_to_tensor is None:
        raise ImportError(
            "TensorFlow support requires `pip install pythtb[speedup]` "
            "or a manual tensorflow install."
        )

    dtype = tf_complex64 if use_32_bit else tf_complex128
    tensor = convert_to_tensor(ham, dtype=dtype)

    if return_eigvecs:
        evals, evecs = tf_eigh(tensor)
        return evals.numpy(), evecs.numpy()
    evals = tf_eigvalsh(tensor)
    return evals.numpy()

class TBModel:
    r"""Tight-binding model constructor.

    This class's primary purpose is to build, store,
    and diagonalize tight-binding Hamiltonians.

    .. versionchanged:: 2.0.0
        The class was refactored to use a separate ``Lattice`` object for lattice
        and orbital information. The ``TBModel`` class now relies on
        the ``Lattice`` object to provide this information. 
        The parameters for ``tb_model`` are now used in the
        constructor of the `Lattice` class, such as ``lat``, ``orb``, ``per``, 
        while ``dim_k`` and ``dim_r`` are inferred from ```lat`` and ``per``.
        The ``nspin`` parameter was renamed to ``spinful`` for clarity.

    Parameters
    ----------
    lattice : Lattice
        The lattice structure of the tight-binding model. This includes
        lattice vectors, orbital positions, and periodic directions. The
        `Lattice` object should be created separately and passed to `TBModel`.

    spinful : bool, optional
        If True, the model is spinful and each orbital is assumed to
        have two spin components. If False, the model is spinless.
        Default value of this parameter is False.

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

    >>> from pythtb import TBModel, Lattice
    >>> lat = Lattice(
    ...    lat_vecs=[[1, 1/2], [0, 2]], 
    ...    orb_vecs=[[0.2, 0.3], [0.1, 0.1], [0.2, 0.2]], 
    ...    periodic_dirs=[1])
    >>> tb = TBModel(lattice=lat, spinful=False)
    """

    def __init__(
        self, lattice: Lattice, spinful: bool = False
    ):
        self._lattice = lattice
        self._nspin = 2 if spinful else 1

        # By default, assume model did not come from w90 object and that
        # position operator is diagonal
        self._assume_position_operator_diagonal = True
        self._from_w90 = False

        # Initialize onsite energies to zero
        if self._nspin == 1:
            self._site_energies = np.zeros((self.norb), dtype=float)
        elif self._nspin == 2:
            self._site_energies = np.zeros((self.norb, 2, 2), dtype=complex)

        # The onsite energies and hoppings are not specified
        # when creating a 'TBModel' object.  They are speficied
        # subsequently by separate function calls defined below.

        # remember which onsite energies user has specified
        self._site_energies_specified = np.zeros(self.norb, dtype=bool)
        self._site_energies_specified[:] = False

        # Initialize hoppings container
        self._hoppings = HoppingTable(self.dim_r, self._nspin == 2)

    def __repr__(self):
        r"""Return a string representation of the ``TBModel`` object.

        Returns
        -------
        str
            String representation of the TBModel.
        """
        return (
            f"pythtb.TBModel(dim_r={self.dim_r}, dim_k={self.dim_k}, "
            f"norb={self.norb}, spinful={self.spinful})"
        )

    def __str__(self):
        r"""Return a string representation of the ``TBModel`` object.

        Returns
        -------
        str
            String representation of the TBModel.
        """
        return self.info(show=False)

    @deprecated(
        "The 'display' method is deprecated and will be removed in a future release. Use 'print(model)' or 'model.info(show=True)' instead."
    )
    def display(self):
        r"""
        .. deprecated:: 2.0.0
            `display` has been deprecated, it is recommended to use `print(model)` or `model.info(show=True)` instead.
        """
        return self.info(show=True)

    def info(self, show: bool = True, short: bool = False):
        r"""Print or return information about the tight-binding model.

        .. versionadded:: 2.0.0
            The `short` parameter was added to control the verbosity of the report.
            The `show` parameter was added to control whether to print the report or return it as a string.

        Parameters
        ----------
        show : bool, optional
            If True, prints the report to stdout. If False, returns the report as a string.
            Default is True.
        short : bool, optional
            If True, prints only a lattice summary. If False, prints hopping and onsite details as well.
            Default is False.

        Returns
        -------
        str or None
            Returns the info string if ``show`` is False, otherwise prints and returns None.

        Notes
        -----
        The report includes lattice vectors, orbital positions, site energies, hoppings, and hopping distances.
        """
        output = []
        header = (
            "----------------------------------------\n"
            "       Tight-binding model report       \n"
            "----------------------------------------"
        )
        output.append(header)
        lat_report = self.lattice._report_list()
        lat_report.pop(0)  # remove header
        lat_report.insert(2, f"spinful                     = {self.spinful}")
        lat_report.insert(4, f"number of spin components   = {self.nspin}")
        lat_report.insert(5, f"number of electronic states = {self.nstate}")
        output.extend(lat_report)

        if not short:
            # Print Site Energies
            output.append("Site energies:")
            for i, site in enumerate(self._site_energies):
                if self._nspin == 1:
                    energy_str = f"{site:^7.3f}"
                elif self._nspin == 2:
                    energy_str = str(site).replace("\n", " ")

                output.append(f"  # {i} ===> {energy_str}")

            amps, i_idx, j_idx, R_vecs = self._hoppings.components()

            output.append("Hoppings:")
            for hop_idx in range(len(self._hoppings)):
                hop_from = int(i_idx[hop_idx])
                hop_to = int(j_idx[hop_idx])
                R_vec = R_vecs[hop_idx]

                coords = ", ".join(f"{value:^5.1f}" for value in R_vec)
                disp = f" + [{coords}]" if self.dim_k else ""
                out_str = f"  < {hop_from:^1} | H | {hop_to:^1}{disp} >  ===> "

                amp = amps[hop_idx]
                if self.spinful:
                    amp_str = str(np.asarray(amp).round(4)).replace("\n", " ")
                else:
                    amp_str = f"{complex(amp):^7.4f}"

                out_str += amp_str
                output.append(out_str)

            output.append("Hopping distances:")
            if len(self._hoppings):
                orb_cart = self.get_orb_vecs(cartesian=True)
                lat_vecs = self.lat_vecs
                for hop_idx in range(len(self._hoppings)):
                    hop_from = int(i_idx[hop_idx])
                    hop_to = int(j_idx[hop_idx])
                    R_vec = R_vecs[hop_idx]

                    pos_i = orb_cart[hop_from]
                    pos_j = orb_cart[hop_to] + R_vec @ lat_vecs

                    coords = ", ".join(f"{value:5.1f}" for value in R_vec)
                    disp = f" + [{coords}]" if self.dim_k else ""

                    distance = np.linalg.norm(pos_j - pos_i)

                    out_str = (
                    f"  | pos({hop_from:>2}) - pos({hop_to:<2}){disp} | = {distance:7.3f}"
                    )
                    output.append(out_str)

        if show:
            print("\n".join(output))
        else:
            return "\n".join(output)

    def _get_periodic_H(self, H_flat, k_vals):
        r"""
        Transform Hamiltonian to periodic gauge so that :math:`H(\mathbf{k}+\mathbf{G}) = H(\mathbf{k})`.

        If ``nspin = 2``, ``H_flat`` should only be flat along `k` and _NOT_ spin.

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
        if k_vals.shape[1] != self.dim_k:
            raise ValueError(f"Invalid k_vals shape: {k_vals.shape}. Expected (Nk, {self.dim_k}).")
        
        if self.dim_k == 0:
            logger.warning(
                "No periodic directions in k-space. Returning H_flat unchanged."
            )
            return H_flat
        
        
        orb_vecs = self._orb_vecs  # reduced units
        orb_vec_diff = orb_vecs[:, None, :] - orb_vecs[None, :, :]
        orb_vec_diff = orb_vec_diff[..., self.per]
        orb_phase = np.exp(
            1j * 2 * np.pi * np.matmul(orb_vec_diff, k_vals.T)
        ).transpose(2, 0, 1)
        H_per_flat = H_flat * orb_phase
        return H_per_flat

    # Property decorators for read-only access to model attributes

    @property
    def lattice(self) -> Lattice:
        """The Lattice object associated with the TBModel.

        .. versionadded:: 2.0.0
        """
        return copy.copy(self._lattice)
    
    @property
    def dim_r(self) -> int:
        """The dimensionality of real space.

        .. versionadded:: 2.0.0
        """
        return self.lattice.dim_r

    @property
    def dim_k(self) -> int:
        """The dimensionality of reciprocal space (periodic directions).

        .. versionadded:: 2.0.0
        """
        return self.lattice.dim_k

    @property
    def nspin(self) -> int:
        """The number of spin components.

        .. versionadded:: 2.0.0
        """
        return self._nspin

    @property
    def spinful(self) -> bool:
        """Whether the model includes spin degrees of freedom.

        .. versionadded:: 2.0.0
        """
        return self._nspin == 2

    @property
    def per(self) -> list[int]:
        """Periodic directions as a list of indices. Alias for `periodic_dirs`.

        .. versionadded:: 2.0.0

        Each index corresponds to a lattice vector in the model.
        """
        return copy.copy(self.periodic_dirs)
    
    @property
    def periodic_dirs(self) -> list[int]:
        """Periodic directions as a list of indices.

        .. versionadded:: 2.0.0

        Each index corresponds to a lattice vector in the model.
        """
        return self.lattice.periodic_dirs

    @property
    def norb(self) -> int:
        """The number of tight-binding orbitals in the model.

        .. versionadded:: 2.0.0
        """
        return copy.copy(self.lattice.norb)

    @property
    def nstate(self) -> int:
        """The number of electronic states in the model is ``norb * nspin``.

        .. versionadded:: 2.0.0
        """
        return self.norb * self.nspin
    
    @property
    def orb_vecs(self) -> np.ndarray:
        """Orbital vectors in reduced coordinates with shape ``(norb, dim_r)``.

        .. versionadded:: 2.0.0
        """
        return copy.copy(self.lattice.orb_vecs)

    @property
    def lat_vecs(self) -> np.ndarray:
        """Lattice vectors in Cartesian coordinates with shape ``(dim_r, dim_r)``.

        .. versionadded:: 2.0.0
        """
        return copy.copy(self.lattice.lat_vecs)
    
    @property
    def recip_lat_vecs(self) -> np.ndarray:
        """Reciprocal lattice vectors in inverse Cartesian units with shape ``(dim_k, dim_k)``.

        .. versionadded:: 2.0.0
        """
        return copy.copy(self.lattice.recip_lat_vecs)

    @property
    def recip_volume(self) -> float:
        """Returns the volume of the reciprocal unit cell in inverse Cartesian units.

        .. versionadded:: 2.0.0
        """
        return copy.copy(self.lattice.recip_volume)

    @property
    def cell_volume(self) -> float:
        """Returns the volume of the unit cell in Cartesian units.

        .. versionadded:: 2.0.0
        """
        return copy.copy(self.lattice.cell_volume)

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

        Returns
        -------
        list[dict]
            A list of hopping dictionaries. Each dictionary contains the following
            keys:

            - ``"amplitude"``: hopping amplitude (complex or matrix)
            - ``"from_orbital"``: index of starting orbital
            - ``"to_orbital"``: index of ending orbital
            - ``"lattice_vector"``: (optional) lattice vector displacement
        """
        amps, i_idx, j_idx, R_vecs = self._hoppings.components()
        formatted: list[dict] = []
        for hop_idx in range(len(self._hoppings)):
            amp = amps[hop_idx]
            if self._nspin == 2:
                amplitude = np.asarray(amp).copy()
            else:
                amplitude = complex(amp)
            entry = {
                "amplitude": amplitude,
                "from_orbital": int(i_idx[hop_idx]),
                "to_orbital": int(j_idx[hop_idx]),
            }
            R_vec = R_vecs[hop_idx]
            if np.any(R_vec):
                entry["lattice_vector"] = R_vec.tolist()
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

    @deprecated("Use 'get_orb_vecs' instead.")
    def get_orb(self):
        """
        .. deprecated:: 2.0.0
           Use 'get_orb_vecs' instead.
        """
        return self.get_orb_vecs(cartesian=False)

    def get_orb_vecs(self, cartesian=False):
        """Return orbital positions.

        .. versionchanged:: 2.0.0
            The name was changed from `get_orb` to `get_orb_vecs`.

        .. versionadded:: 2.0.0
            Support for Cartesian coordinates with the `cartesian` parameter.

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
        return self.lattice.get_orb_vecs(cartesian=cartesian)

    @copydoc(Lattice.nn_bonds)
    def nn_bonds(self, n_shells: int, report: bool = False):
        return self.lattice.nn_bonds(n_shells, report=report)

    @deprecated("Use 'get_lat_vecs' instead.")
    def get_lat(self):
        """
        .. deprecated:: 2.0.0
           Use 'get_lat_vecs' instead.
        """
        return self.get_lat_vecs()
    
    def get_lat_vecs(self):
        """Return lattice vectors in Cartesian coordinates.

        .. versionchanged:: 2.0.0
            The name was changed from `get_lat` to `get_lat_vecs`.

        Returns
        -------
        np.ndarray
            Lattice vectors, shape ``(dim_r, dim_r)``.
        """
        return self.lattice.get_lat_vecs()


    def set_onsite(self, onsite_en, ind_i=None, mode="set"):
        r"""
        Define on-site energies for tight-binding orbitals.

        You can set the energy for a single orbital (by specifying ``ind_i``),
        or for all orbitals at once (by passing a list/array to ``onsite_en``).

        .. deprecated:: 2.0.0
            ``mode="reset"`` is deprecated. Use ``mode="set"`` instead.

        Parameters
        ----------
        onsite_en : float or array-like or (2, 2) ndarray
            If ``ind_i`` is ``None``, ``onsite_en`` must be a list/array of length
            ``norb`` (one value per orbital). Otherwise it may be a single value.
            In spinful models it may also be a 2 x 2 Hermitian matrix.

            **Spinless**  (``spinful=False``)

            - Real scalar, or list/array of real scalars (one per orbital).

            **Spinful**  (``spinful=True``)

            - **Scalar** ``a``: interpreted as :math:`a I` (same value for both spins).
            - **4-vector** ``[a, b, c, d]``: interpreted as :math:`a I + b\,\sigma_x + c\,\sigma_y + d\,\sigma_z`, i.e.
                .. math::

                    \begin{bmatrix}
                    a + d & b - i c \\
                    b + i c & a - d
                    \end{bmatrix}

            - **Full matrix**: a 2 x 2 Hermitian ndarray.

        ind_i : int, optional
            Orbital index to update. If ``None``, all orbitals are updated and
            ``onsite_en`` must be a sequence of length ``norb``.

        mode : {'set', 'add'}, optional
            How to apply ``onsite_en``.

            - ``'set'``: replace the value(s).
            - ``'add'``: add to existing value(s).

        Notes
        -----
        When called multiple times with ``mode='add'``, values accumulate.

        Examples
        --------
        >>> tb.set_onsite([0.0, 1.0, 2.0])              # all orbitals
        >>> tb.set_onsite(100.0, ind_i=1, mode="add")   # single orbital
        >>> tb.set_onsite(0.0, ind_i=1, mode="set")
        >>> tb.set_onsite([2.0, 3.0, 4.0], mode="set")
        >>> tb.set_onsite([1.0, 0.2, 0.0, -0.1], ind_i=0)  # spinful 4-vector
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
            if len(onsite_en) != self.norb:
                raise ValueError(
                    "List of onsite energies must include a value for every orbital."
                )

            processed = [process(val) for val in onsite_en]
            indices = np.arange(self.norb)
        else:
            if ind_i < 0 or ind_i >= self.norb:
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
        

    def _get_flattened_indices(self):
        return self._hoppings.flatten_cache(self.norb)

    def _normalize_kpoints(self, k_pts, *, allow_none_for_finite: bool = False) -> np.ndarray | None:
        """Validate and reshape user-provided k-points."""
        dim_k = self.dim_k
        if dim_k == 0:
            if k_pts is None:
                return None
            if allow_none_for_finite:
                return None
            raise ValueError("k_pts should not be specified for finite (dim_k=0) models.")

        if k_pts is None:
            raise ValueError("Must supply k_pts for periodic systems (dim_k > 0).")

        k_arr = np.asarray(k_pts, dtype=float)
        if k_arr.ndim == 1:
            if k_arr.shape[0] != dim_k:
                raise ValueError(f"k_pts must have shape ({dim_k},) for a single point.")
            k_arr = k_arr.reshape(1, dim_k)
        if k_arr.ndim != 2 or k_arr.shape[1] != dim_k:
            raise ValueError(f"k_pts must have shape (Nk, {dim_k}).")
        return k_arr

    def _hamiltonian_finite(self, amps, i_idx, j_idx, site_energies, *, flatten_spin: bool):
        norb = self.norb

        if not self.spinful:
            amps = amps.astype(complex)
            ham = np.zeros((norb, norb), dtype=complex)
            if amps.size:
                np.add.at(ham, (i_idx, j_idx), amps)
                np.add.at(ham, (j_idx, i_idx), amps.conj())
            np.fill_diagonal(ham, site_energies)
            return ham

        # spinful
        nspin = self.nspin
        amps = np.asarray(amps, dtype=complex)
        ham = np.zeros((norb, nspin, norb, nspin), dtype=complex)
        if amps.size:
            for hop_idx in range(amps.shape[0]):
                block = amps[hop_idx]
                ham[i_idx[hop_idx], :, j_idx[hop_idx], :] += block
                ham[j_idx[hop_idx], :, i_idx[hop_idx], :] += block.conj().T
        for orb in range(norb):
            ham[orb, :, orb, :] += site_energies[orb]
        if flatten_spin:
            ham = ham.reshape(norb * nspin, norb * nspin)
        return ham

    def _hamiltonian_periodic(
        self,
        k_vecs: np.ndarray,
        amps,
        i_idx,
        j_idx,
        R_vecs,
        site_energies,
        *,
        flatten_spin: bool,
    ):
        norb = self.norb
        per = np.asarray(self.per)
        orb_red = np.asarray(self.orb_vecs)

        n_kpts = k_vecs.shape[0]
        n_hops = amps.shape[0]

        i_idx = i_idx.astype(int)
        j_idx = j_idx.astype(int)
        R_vecs = R_vecs.astype(float)

        orb_i = orb_red[i_idx]
        orb_j = orb_red[j_idx]
        delta_r = R_vecs - orb_i + orb_j
        delta_r_per = delta_r[:, per]

        if n_hops:
            k_dot_r = k_vecs @ delta_r_per.T
            phases = np.exp(1j * 2 * np.pi * k_dot_r)
        else:
            phases = None

        if not self.spinful:
            amps = amps.astype(complex)
            ham = np.zeros((n_kpts, norb, norb), dtype=complex)
            if n_hops:
                cache = self._get_flattened_indices()
                order = cache["order"]
                starts = cache["starts"]
                uniq = cache["uniq"]
                cols_transposed = cache["cols_transposed"]

                ham_flat = ham.reshape(n_kpts, -1)
                contrib = phases[:, order] * amps[order]
                sums = np.add.reduceat(contrib, starts, axis=1)
                ham_flat[:, uniq] += sums
                ham_flat[:, cols_transposed] += sums.conj()

            diag = np.arange(norb)
            ham[:, diag, diag] += site_energies
            return ham

        # spinful
        nspin = self.nspin
        amps = np.asarray(amps, dtype=complex)
        ham = np.zeros((n_kpts, norb, nspin, norb, nspin), dtype=complex)
        if n_hops:
            weighted = phases[..., None, None] * amps[None, :, :, :]
            for s_out in range(nspin):
                for s_in in range(nspin):
                    contrib = weighted[..., s_out, s_in]
                    np.add.at(
                        ham[:, :, s_out, :, s_in],
                        (slice(None), i_idx, j_idx),
                        contrib,
                    )
                    np.add.at(
                        ham[:, :, s_in, :, s_out],
                        (slice(None), j_idx, i_idx),
                        contrib.conj(),
                    )
        for orb in range(norb):
            ham[:, orb, :, orb, :] += site_energies[orb]
        if flatten_spin:
            ham = ham.reshape(n_kpts, norb * nspin, norb * nspin)
        return ham

    
    ############################################################################

    def set_nn_hops(self, hop_amps: list, nn_shells: list[int], mode="set"):
        r"""Define nearest-neighbor hopping parameters up to a specified shell.

        This function sets hopping amplitudes for all bonds in the specified
        nearest-neighbor shells. The shells are defined based on the distance
        from each orbital, with shell 1 being the nearest neighbors, shell 2
        being the next-nearest neighbors, and so on.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        hop_amp : list or array-like
            List or array of hopping amplitudes for each shell.
            The length of ``hop_amp`` should match the length of ``nn_shells``, 
            where each element corresponds to the hopping amplitude for that shell.
        nn_shells : list[int]
            List of integers specifying the shells for each hopping amplitude. Counting
            starts from 1, so ``nn_shells=[1, 2]`` indicates nearest-neighbor and 
            next-nearest-neighbor shells. The length of ``nn_shells`` should match the length 
            of ``hop_amp``. Each element in ``nn_shells`` should be a positive integer.
        mode : {'set', 'add'}, optional
            Specifies how `hop_amp` is used
                - "set": Set the hopping term to the value of `hop_amp`. (Default)
                - "add": Add `hop_amp` to the previous value.

        Notes
        -----
        The hopping amplitudes are applied to all bonds in each shell.

        Examples
        --------
        Setting nearest-neighbor and next-nearest-neighbor hoppings:

        >>> tb.set_nn_hops([1.0, 0.5], [1, 2])

        Setting only nearest-neighbor hoppings:

        >>> tb.set_nn_hops([1.0], [1])

        """
        n_shells = max(nn_shells)

        if n_shells == 0:
            raise ValueError("hop_amp must have at least one element.")
        if len(nn_shells) != n_shells:
            raise ValueError("nn_shells must have length equal to n_shells.")
        if not all(isinstance(shell, int) and shell > 0 for shell in nn_shells):
            raise ValueError("Each element in nn_shells must be a positive integer.")
        if not isinstance(hop_amps, (list, np.ndarray)):
            raise TypeError("hop_amp must be a list or array.")
        if any(not isinstance(amp, (int, float, complex, list, np.ndarray)) for amp in hop_amps):
            raise TypeError("Each element in hop_amp must be a scalar, list, or array.")

        shell_bonds = self.nn_bonds(n_shells)[1]

        hops = dict(zip(nn_shells, hop_amps))

        for shell_idx, shell in enumerate(shell_bonds):
            amp = hops.get(shell_idx + 1, None)
            if amp is None:
                continue
            for bond in shell:
                i, j, R = bond
                self.set_hop(amp, i, j, R, mode=mode, allow_conjugate_pair=True)

    def _append_hops(self, hop_amps, i_idx, j_idx, R_vecs):
        hop_amps = np.asarray(hop_amps)
        i_idx = np.asarray(i_idx, dtype=int)
        j_idx = np.asarray(j_idx, dtype=int)
        R_vecs = np.asarray(R_vecs, dtype=int).reshape(len(i_idx), self.dim_r)

        blocks = [self._val_to_block(val) for val in hop_amps]
        self._hoppings.extend(
            blocks,
            i_idx.tolist(),
            j_idx.tolist(),
            R_vecs.tolist(),
        )

    def _set_hops_bulk(self, hop_amps, i_idx, j_idx, R_vecs, mode="set"):
        mode = mode.lower()
        if mode not in {"set", "add"}:
            raise ValueError("mode must be 'set' or 'add'")

        hop_amps = np.asarray(hop_amps)
        i_idx = np.asarray(i_idx, dtype=int)
        j_idx = np.asarray(j_idx, dtype=int)
        R_vecs = np.asarray(R_vecs, dtype=int).reshape(len(i_idx), self.dim_r)

        for amp, i, j, R in zip(hop_amps, i_idx, j_idx, R_vecs, strict=True):
            self.set_hop(
                amp,
                int(i),
                int(j),
                R,
                mode=mode,
                allow_conjugate_pair=True,
            )

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
            H_{ij}(\mathbf{R}) = \langle \phi_{\mathbf{0},i} | H | \phi_{\mathbf{R},j} \rangle

        where :math:`\langle \phi_{\mathbf{0},i} |` is the i-th orbital in the home unit cell,
        and :math:`| \phi_{\mathbf{R},j} \rangle` is the j-th orbital in a cell shifted by lattice vector :math:`\mathbf{R}`.

        .. deprecated:: 2.0.0
            Using 'reset' for `mode` is deprecated, use 'set' instead.

        Parameters
        ----------
        hop_amp : scalar, array-like, np.ndarray of shape ``(2, 2)``
            For spinless models (`spinful=False`):
                - Real or complex scalar.
            For spinful models (`spinful=True`):
                - Scalar: interpreted as :math:`a I` for both spin components.
                - 4-vector ``[a, b, c, d]``: interpreted as :math:`a I + b \sigma_x + c \sigma_y + d \sigma_z`:

                    .. math::
                        \begin{bmatrix}
                            a + d & b - i c \\
                            b + i c & a - d
                        \end{bmatrix}

                - Full 2 x 2 Hermitian matrix.
        ind_i : int
            Index of bra orbital (in home unit cell).
        ind_j : int
            Index of ket orbital (in cell shifted by `ind_R`).
        ind_R : array-like of int, optional
            Lattice vector (integer array, in reduced coordinates)
            pointing to the unit cell where the ket orbital is located.
            The number of coordinates must equal the dimensionality in
            real space (``dim_r``) for consistency, but only the periodic directions of ``ind_R`` are used.
            If reciprocal space is zero-dimensional (as in a molecule), this parameter does not need 
            to be specified.
        mode : {'set', 'add'}, optional
            Specifies how `hop_amp` is used
                - "set": Set the hopping term to the value of `hop_amp`. (Default)
                - "add": Add `hop_amp` to the previous value.
        allow_conjugate_pair : bool, optional
            If True, allows specification of both a hopping and its conjugate pair.
            If False, prevents double-counting.

        Notes
        -----
        Strictly speaking, this term specifies hopping amplitude for hopping from site `j+R` to site i, not vice-versa.
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
        mode = mode.lower()

        # deprecation warning
        if mode == "reset":
            logger.warning(
                "The 'reset' mode is deprecated as of v2.0. Use 'set' instead to set the hopping term."
                "This will be removed in a future version."
            )
            mode = "set"

        ind_i, ind_j, R_vec = self._hoppings.normalize_entry(
            ind_i,
            ind_j,
            ind_R,
            norb=self.norb,
            dim_k=self.dim_k,
            periodic_dirs=self.periodic_dirs,
        )

        # Do not allow onsite hoppings to be specified here
        if ind_i == ind_j:
            if self.dim_k == 0 or bool(np.all(R_vec == 0)):
                raise ValueError(
                    "Do not use set_hop for onsite terms. Use set_onsite instead."
                )

        hop_use = self._val_to_block(hop_amp)
        table = self._hoppings

        existing_idx = table.find(ind_i, ind_j, R_vec)
        if not allow_conjugate_pair:
            conj_idx = table.find(ind_j, ind_i, -R_vec)
            if conj_idx is not None and (existing_idx is None or conj_idx != existing_idx):
                raise ValueError(
                    f"Conjugate element already specified for i={ind_i}, j={ind_j}, R={R_vec.tolist()}. "
                    "Either avoid double entry or set allow_conjugate_pair=True."
                )

        mode = mode.lower()
        if mode == "set":
            if existing_idx is not None:
                table.update(existing_idx, amplitude=hop_use, R=R_vec)
            else:
                table.append(hop_use, ind_i, ind_j, R_vec)
        elif mode == "add":
            if existing_idx is not None:
                table.accumulate(existing_idx, hop_use)
            else:
                table.append(hop_use, ind_i, ind_j, R_vec)
        else:
            raise ValueError("Wrong value of mode parameter. Should be either `set` or `add`.")
    

    def _val_to_block(self, val):
        r"""
        Convert input value to appropriate matrix block for onsite or hopping.

        For spinful=False, returns the value (should be real or complex scalar).
        For nspin=2:
            - Scalar: returns a 2 x 2 matrix proportional to the identity.
            - Array with up to four elements: returns a 2 x 2 matrix as
              :math:`a I + b \sigma_x + c \sigma_y + d \sigma_z`.
            - 2 x 2 matrix: returns the matrix as is.

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
        if not self.spinful:
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
    

    def velocity(
            self, 
            k_pts: np.ndarray, 
            cartesian: bool = False,
            flatten_spin_axis: bool = False
            ) -> np.ndarray:
        r"""Generate the velocity operator in the orbital basis.

        The velocity operator is defined via the derivative of the Hamiltonian
        with respect to k of each reciprocal lattice direction, i.e., 

        .. math::
            v_k^{\mu} = \hbar \frac{\partial H(k)}{\partial k_{\mu}}
        
        Here, we use units where :math:`\hbar = 1`.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        k_pts : array of shape (Nk, dim_k)
            Array of k-points in reduced coordinates.
        cartesian : bool, optional
            If True, use Cartesian coordinates for the velocity operator.

        Returns
        -------
        vel : np.ndarray
            Velocity operators at each k-point. First axis indexes the cartesian direction if `cartesian` is True.
            Otherwise, it indexes the reduced direction. Shape is `(dim_k, Nk, norb, norb)` for spinless models,
            or `(dim_k, Nk, norb, nspin, norb, nspin)` for spinful models.

        """
        dim_k = self.dim_k

        k_arr = self._normalize_kpoints(k_pts)

        norb = self.norb
        per = np.asarray(self.per)
        orb_red = np.asarray(self.orb_vecs)

        table = self._hoppings
        amps, i_indices, j_indices, R_vecs = table.components()
        n_hops = i_indices.size

        i_indices = i_indices.astype(int)
        j_indices = j_indices.astype(int)
        R_vecs = R_vecs.astype(float)

        orb_i = orb_red[i_indices]
        orb_j = orb_red[j_indices]

        delta_r = R_vecs - orb_i + orb_j
        delta_r_per = delta_r[:, per]

        if n_hops:
            k_dot_r = k_arr @ delta_r_per.T
            phases = np.exp(1j * 2 * np.pi * k_dot_r)
        else:
            phases = np.zeros((k_arr.shape[0], 0), dtype=complex)
        if cartesian:
            lattice = self.get_lat_vecs()[self.per, :]
            coeff = (1j * delta_r_per @ lattice).T[:, None, :]
        else:
            coeff = (1j * 2 * np.pi * delta_r_per).T[:, None, :]

        deriv_phase = coeff * phases[None, ...] if n_hops else coeff[:, :, :0]

        if not self.spinful:
            amps_use = np.asarray(amps, dtype=complex)
            vel = np.zeros((dim_k, k_arr.shape[0], norb, norb), dtype=complex)
            if n_hops:
                cache = self._get_flattened_indices()
                order = cache["order"]
                starts = cache["starts"]
                uniq = cache["uniq"]
                cols_transposed = cache["cols_transposed"]

                vel_flat = vel.reshape(dim_k, k_arr.shape[0], -1)
                contrib_sorted = deriv_phase[:, :, order] * amps_use[order]
                sums = np.add.reduceat(contrib_sorted, starts, axis=2)
                vel_flat[..., uniq] += sums
                vel_flat[..., cols_transposed] += sums.conj()
            return vel
        
        nspin = self.nspin
        vel = np.zeros((dim_k, k_arr.shape[0], norb, nspin, norb, nspin), dtype=complex)
        if n_hops:
            weighted = deriv_phase[..., None, None] * amps[None, None, :, :, :]
            for s_out in range(nspin):
                for s_in in range(nspin):
                    contrib = weighted[..., s_out, s_in]
                    np.add.at(
                        vel,
                        (slice(None), slice(None), i_indices, s_out, j_indices, s_in),
                        contrib,
                    )
                    np.add.at(
                        vel,
                        (slice(None), slice(None), j_indices, s_in, i_indices, s_out),
                        contrib.conj(),
                    )
        
        if flatten_spin_axis:
            vel = vel.reshape(dim_k, k_arr.shape[0], norb * nspin, norb * nspin)
        return vel
    
    def hamiltonian(
            self, 
            k_pts: np.ndarray = None, 
            flatten_spin_axis: bool = False
            ) -> np.ndarray:
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
        flatten_spin_axis : bool, optional
            If True, the spin indices are flattened into the orbital indices.
            This results in a Hamiltonian at each k-point of shape ``(norb*nspin, norb*nspin)``.
            If False (default), the Hamiltonian has shape ``(norb, nspin, norb, nspin)``.

        Returns
        -------
        ham : np.ndarray 
            Array of Bloch-Hamiltonian matrices defined on the specified k-points. The Hamiltonian is Hermitian by construction.

            - If ``dim_k`` > 0: shape is ``(n_kpts, norb, norb)`` for spinless models, or ``(n_kpts, norb, nspin, norb, nspin)`` 
              for spinful models, unless `flatten_spin` is True, in which case the shape is ``(n_kpts, norb*nspin, norb*nspin)``.

            - If ``dim_k`` = 0: shape is ``(norb, norb)`` for spinless or ``(norb, nspin, norb, nspin)`` for spinful models,
              unless `flatten_spin` is True, in which case the shape is ``(norb*nspin, norb*nspin)``.

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
        site_energies = np.asarray(self._site_energies)
        amps, i_idx, j_idx, R_vecs = self._hoppings.components()

        if self.dim_k == 0:
            if k_pts is not None:
                raise ValueError("k_pts should not be specified for finite (dim_k=0) models.")
            return self._hamiltonian_finite(
                amps,
                i_idx,
                j_idx,
                site_energies,
                flatten_spin=flatten_spin_axis,
            )

        k_arr = self._normalize_kpoints(k_pts)
        return self._hamiltonian_periodic(
            k_arr,
            amps,
            i_idx,
            j_idx,
            R_vecs,
            site_energies,
            flatten_spin=flatten_spin_axis,
        )

    def _sol_ham(
        self, ham, return_eigvecs=False, keep_spin_ax=True, tf_speedup=False, use_32_bit=False,
        memory_info=False):
        """Solves Hamiltonian and returns eigenvectors, eigenvalues"""
        # NOTE: this function is separate so that it can be jit-compiled if needed

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
        
        if tf_speedup:
            result = _tensorflow_solve(
                ham_use, return_eigvecs=return_eigvecs, use_32_bit=use_32_bit
                )
            if return_eigvecs:
                # return later
                eval, evec = result
            else:
                return result
            
        else:
            if use_32_bit:
                ham_use = ham_use.astype(np.complex64)
            else:
                ham_use = ham_use.astype(np.complex128)
            if return_eigvecs:
                # return later
                eval, evec = np.linalg.eigh(ham_use)
            else:
                return np.linalg.eigvalsh(ham_use)
            
        if return_eigvecs:
            # transpose matrix eig since otherwise it is confusing
            # now eig[i,:] is eigenvector for eval[i]-th eigenvalue
            evec = evec.swapaxes(-1, -2)
            if keep_spin_ax:
                evec = evec.reshape(*shape_evecs)
            return eval, evec

    def solve_ham(
            self, 
            k_pts = None, 
            return_eigvecs: bool = False, 
            flatten_spin_ax: bool = True,
            tf_speedup: bool = False) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        r"""Diagonalize the Hamiltonian 
        
        Solve for eigenvalues and optionally eigenvectors of the tight-binding model
        at a list of one-dimensional k-vectors.

        .. versionchanged:: 2.0.0
            Merged :func:`solve_all` and :func:`solve_one` into :func:`solve_ham`.
            This function will equivalently handle both a single k-point and
            multiple k-points. 

            Parameter `eig_vectors` renamed to `return_eigvecs`.
            Parameter `k_list` renamed to `k_pts`.

        .. versionadded:: 2.0.0
            Parameter `keep_spin_ax` and `tf_speedup` added.

        Parameters
        ----------
        k_pts : array_like, optional
            One-dimensional list or array of k-vectors, each given in reduced coordinates.
            Shape should be ``(Nk, dim_k)``, where ``dim_k`` is the number of periodic directions.
            Should not be specified for systems with zero-dimensional reciprocal space.
        return_eigvecs : bool, optional
            If True, both eigenvalues and eigenvectors are returned.
            If False (default), only eigenvalues are returned.
        flatten_spin_axis : bool, optional
            If True (default), the spin axes are kept in the output eigenvectors.
            If False, the spin axes are flattened.
        tf_speedup : bool, optional
            If True, use TensorFlow to accelerate the diagonalization.
            This requires TensorFlow to be installed. Default is False.

        Returns
        -------
        eval : {(Nk, nstate), (nstate)} np.ndarray 
            Array of eigenvalues. Shape is:

            - (Nk, nstates) for periodic systems
            - (nstates,) for zero-dimensional (molecular) systems

        evec : {(Nk, nstate, nstate), (nstate, nstate), (Nk, nstate, norb, 2), (nstate, norb, 2)} np.ndarray, optional
            Array of eigenvectors (if ``return_eigvecs=True``). The ordering of bands matches that in `eval`.

            Each entry :code:`evec[k, n, j]` is the coefficient of orbital `j` in the Bloch eigenstate
            :math:`C^{n \mathbf{k}}_j`.

            For spinless models:

            - Shape is ``(Nk, nstates, norb)`` in periodic systems
            - Shape is ``(nstates, norb)`` in zero-dimensional systems
            - If only one k-point is provided, the redundant k-axis is removed, resulting in shape ``(nstates, norb)``.

            For spinful models:

            - Shape is ``(Nk, nstates, norb, 2)`` for periodic systems
            - Shape is ``(nstates, norb, 2)`` for zero-dimensional systems
            - If only one k-point is provided, the redundant k-axis is removed, resulting in shape ``(nstates, norb, 2)``.
            - If `keep_spin_ax=False` and the model is spinful, the spin axes are flattened into the orbital axes,
              resulting in shapes ``(Nk, nstates, norb*2)`` or ``(nstates, norb*2)``.
            
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
        logger.debug("Initializing Hamiltonian...")
        Ham = self.hamiltonian(k_pts)

        logger.debug("Diagonalizing Hamiltonian...")
        if return_eigvecs:
            eigvals, eigvecs = self._sol_ham(
                Ham, return_eigvecs=return_eigvecs, keep_spin_ax=flatten_spin_ax, tf_speedup=tf_speedup
            )
            if self.dim_k != 0:
                if eigvals.ndim != 2:
                    raise ValueError("Wrong shape of eigvals")
                # if only one k_point, remove that redundant axis (reproduces solve_one)
                if eigvals.shape[0] == 1:
                    eigvals = eigvals[0]
                    eigvecs = eigvecs[0]

            return eigvals, eigvecs
        else:
            eigvals = self._sol_ham(Ham, return_eigvecs=return_eigvecs)

            if self.dim_k != 0:
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
    
    def compute_bands(self, k_nodes, nk=10):
        r"""Compute band structure along a specified k-point path.

        The band structure is computed by diagonalizing the Hamiltonian at
        a series of k-points along the specified path in reciprocal space.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        k_nodes : list of array_like
            List of k-points defining the path in reduced coordinates.
            Each k-point should be an array-like of length `dim_k`.
            The path is constructed by linearly interpolating between
            consecutive k-points in the list.

        n_kperseg : int, optional
            Number of k-points to interpolate between each pair of consecutive
            k-points in `k_path`. Default is 10.

        flatten_spin : bool, optional
            If True, the spin indices are flattened into the orbital indices.
            This results in a Hamiltonian at each k-point of shape ``(norb*nspin, norb*nspin)``.
            If False (default), the Hamiltonian has shape ``(norb, nspin, norb, nspin)``.

        Returns
        -------
        k_vecs : np.ndarray of shape (N, dim_k)
            Array of interpolated k-points along the path.

        evals : np.ndarray of shape (N, nbnd)
            Array of eigenvalues at each k-point along the path.

        Notes
        -----
        This function uses linear interpolation to generate intermediate k-points
        between those specified in `k_nodes`. The total number of k-points returned
        is ``nk``.

        Examples
        --------
        Compute band structure along a path from Gamma to X to M in a 2D square lattice:

        >>> k_nodes = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]]
        >>> k_vecs, evals = tb.compute_bands(k_nodes, n_kperseg=20)
        """
        k_vec, _, _ = self.k_path(k_nodes, nk, report=False)
        return k_vec, self.solve_ham(k_vec, return_eigvecs=False)

    #TODO: Decide whether to return fin_model or modify in place
    def cut_piece(self, num_cells, periodic_dir, glue_edges=False) -> "TBModel":
        r"""Cut a (d-1)-dimensional piece out of a d-dimensional tight-binding model.

        .. versionchanged:: 2.0.0
            Changed parameter names for clarity: `num` -> `num_cells`, `fin_dir` -> `periodic_dir`.
        
        Constructs a (d-1)-dimensional tight-binding model out of a
        d-dimensional one by repeating the unit cell a given number of
        times along one of the periodic lattice vectors. 
        
        Parameters
        ----------
        num_cells : int
            How many times to repeat the unit cell.

        periodic_dir : int
            Index of the periodic lattice vector along which to make the system finite.

        glue_edges : bool, optional
            If True, allow hoppings from one edge to the other of a cut model.

        Returns
        -------
        fin_model : TBModel
            Object of type :class:`pythtb.TBModel` representing a cutout
            tight-binding model. 

        See Also
        ---------
        :ref:`cubic-slab-hwf-nb` : For an example
        :ref:`three-site-thouless-nb` : For an example

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
        by repeating model along second lattice vector ten times


        >>> A = TBModel(Lattice([[1.0, 0.0, 0.0],
        ...                      [0.0, 1.0, 0.0],
        ...                      [0.0, 0.0, 1.0]], ...))
        >>> B = A.cut_piece(10, 1)

        Further cut two-dimensional model B into one-dimensional model
        A by repeating unit cell twenty times along third lattice
        vector and allow hoppings from one edge to the other

        >>> C = B.cut_piece(20, 2, glue_edges=True)

        """
        if not isinstance(num_cells, int) or num_cells < 1:
            raise ValueError("num_cells must be a positive integer.")
        if not isinstance(periodic_dir, int) or periodic_dir not in self.periodic_dirs:
            raise ValueError(
                "periodic_dir must be an integer corresponding to one of the periodic directions."
            )
        if not isinstance(glue_edges, bool):
            raise ValueError("glue_edges must be a boolean.")
        if self.dim_k == 0:
            raise ValueError("Can't cut a piece out of a finite sample.")
        
        #TODO: Why can't num_cells be 1 if glue_edges is False?
        if num_cells == 1 and glue_edges:
            raise ValueError("Can't have `num=1` and gluing of the edges!")
        
        lat_fin = self.lattice.cut_piece(num_cells, periodic_dir)
        fin_model = TBModel(lat_fin, spinful=self.spinful)

        onsite = []  # store onsite energies
        for _ in range(num_cells):  # go over all cells in finite direction
            for j in range(self.norb):  # go over all orbitals in one cell
                # do the onsite energies at the same time
                onsite.append(self._site_energies[j])
        onsite = np.array(onsite)
        fin_model.set_onsite(onsite, mode="set")

        # remember if came from w90
        fin_model._assume_position_operator_diagonal = (
            self._assume_position_operator_diagonal
        )
        amps, from_idx, to_idx, R_vecs = self._hoppings.components()
        for c in range(num_cells):
            for amp, ind_i, ind_j, ind_R in zip(amps, from_idx, to_idx, R_vecs, strict=True):
                hop_amp = amp.copy() if self._nspin == 2 else complex(amp)
                R_vec = ind_R.copy()
                jump_fin = int(R_vec[periodic_dir])

                hi = int(ind_i) + c * self.norb
                hj = int(ind_j) + (c + jump_fin) * self.norb

                if fin_model.dim_k != 0:
                    R_vec[periodic_dir] = 0
                    R_arg = R_vec
                else:
                    R_arg = None

                to_add = True
                if not glue_edges:
                    if hj < 0 or hj >= self.norb * num_cells:
                        to_add = False
                else:
                    hj = int(hj) % int(self.norb * num_cells)

                if to_add:
                    fin_model.set_hop(
                        hop_amp,
                        hi,
                        hj,
                        R_arg,
                        mode="add",
                        allow_conjugate_pair=True,
                    )

        return fin_model

    def make_finite(
            self, 
            periodic_dirs: list[int], 
            num_cells: list[int], 
            glue_edges: list[bool] = None
            ) -> "TBModel":
        r"""Returns a finite model.

        This function constructs a finite tight-binding model by removing periodicity
        along specified directions. The resulting model has open boundary conditions
        along those directions, with the option to glue edges together.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        periodic_dirs : list[int]
            List of indices of periodic directions along which
            you wish to make the model finite.
        num_cells : list[int]
            Number of unit cells of the sample along each periodic direction.
        glue_edges : list[bool], optional
            If True, allow hoppings from one edge to the other of a cut model
            along the corresponding direction. If None (default), no edge gluing
            is performed along any direction and we have open boundary conditions.

        Returns
        -------
        finite : :class:`pythtb.TBModel`
            A model whose periodic hoppings have been removed (OBC model).

        See Also
        ---------
        `cut_piece` : Cut a lower-dimensional piece out of a higher-dimensional model.
        :ref:`haldane-nb` : For an example
        :ref:`haldane-edge-nb` : For an example
        :ref:`fkm-nb` : For an example
        :ref:`local-chern-nb` : For an example

        Notes
        -----   
        - This function applies `cut_piece` iteratively along each specified direction.
          The order of directions in `dirs` determines the sequence of cuts.
        - Orbitals in the returned model are numbered so that the `i`-th orbital of the `n`-th unit 
          cell along the first direction in `dirs`, the `m`-th unit cell along the second direction in `dirs`, etc., 
          has index ``i + norb * (n + m * num_cells[0] + ...)`` (here `norb` is the number of orbitals in the original model).
        - The real-space lattice vectors of the returned model are the same as those of
          the original model; only the dimensionality of reciprocal space
          is reduced.

        Examples
        --------
        Construct a two-dimensional finite model by removing periodicity
        along both lattice vectors of a two-dimensional model
        >>> lat = Lattice([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0]], periodic_dirs=[0,1])
        >>> tb = TBModel(lat)
        >>> fin_tb = tb.make_finite(periodic_dirs=[0, 1], num_cells=[10, 5])
        >>> fin_tb.dim_k
        0
        >>> fin_tb.norb
        50
        """
        if self.dim_k == 0:
            raise ValueError("Model is already finite!")
        if not all(d in self.periodic_dirs for d in periodic_dirs):
            raise ValueError("All directions in `periodic_dirs` must be periodic.")
        if len(periodic_dirs) != len(set(periodic_dirs)):
            raise ValueError("All directions in `periodic_dirs` must be unique.")
        if len(periodic_dirs) != len(num_cells):
            raise ValueError("Length of `periodic_dirs` must match length of `num_cells`.")
        if not all(n_cell > 0 for n_cell in num_cells):
            raise ValueError("Number of sites along finite direction must be greater than 0")

        if glue_edges is not None:
            if len(glue_edges) != len(num_cells):
                raise ValueError("Length of `glue_edges` must match number of periodic directions.")
        else:
            glue_edges = [False] * self.dim_k

        cut = self
        for idx, n_cell in enumerate(num_cells):
            cut = cut.cut_piece(num_cells=n_cell, periodic_dir=periodic_dirs[idx], glue_edges=glue_edges[idx])

        return cut

    # This function is being deprecated. The preferred way to reduce dimensionality
    # is to use `make_finite` with `num_cells=1` along the desired directions. 
    # This approach is more general and can handle multiple directions at once.
    # If the intention is to keep periodicity along all directions while keeping some
    # k-values fixed, this can be achieved by using the `hamiltonian` method passing the 
    # desired k-values. Explicit manipulation of k-space sampling in the model is discouraged. k-space
    # sampling is managed by 'Mesh' and 'WFArray' classes.
    @deprecated("use `make_finite` with `num_cells=1` instead (since v2.0).", category=FutureWarning)
    def reduce_dim(self) -> "TBModel":
        r"""
        .. deprecated:: 2.0.0
            Use `make_finite` with `num_cells=[1, ...]` along the desired directions instead.
            If the intention is to keep periodicity along all directions while keeping some
            k-values fixed, this can be achieved by using the `hamiltonian` method passing the 
            desired k-values.
        """
        pass

    def change_nonperiodic_vector(
        self, 
        fin_dir: int, 
        new_latt_vec: np.ndarray = None, 
        to_home: bool = True, 
        ):
        """Change non-periodic lattice vector 

        .. versionchanged:: 2.0.0
            Parameter `to_home_supress_warning` has been removed.
            Parameter `np_dir` renamed to `fin_dir` for clarity.

        Changes one of the non-periodic "lattice vectors". Non-periodic lattice vectors 
        are those that are not listed as periodic with the `periodic_dirs` parameter. 
        The orbital vectors are modified accordingly so that the actual (Cartesian) coordinates of 
        orbitals remain unchanged.

        Parameters
        ----------
        fin_dir : int
            Index of non-periodic lattice vector to change.

        new_latt_vec : array_like, optional
            The new non-periodic lattice vector. If None (default), the new
            non-periodic lattice vector is constructed to be orthogonal to all periodic 
            vectors and to have the same length as the original non-periodic vector.

        to_home : bool, optional
            If ``True`` (default), shift all orbitals to the home cell along
            periodic directions. Default behavior is to shift orbitals
            to the home cell.

        See Also
        --------
        :ref:`boron-nitride-nb` : For an example.

        Notes
        -----
        - This function is especially useful after using function cut_piece to create slabs, 
          rods, or ribbons.
        - By default, the new non-periodic vector is constructed
          from the original by removing all components in the periodic
          space. This ensures that the Berry phases computed in the
          periodic space correspond to the usual expectations.
        - For example, after this change, the Berry phase computed for a
          ribbon depends only on the location of the Wannier center
          in the extended direction, not on its location in the
          transverse direction. Alternatively, the new non-periodic
          vector can be set explicitly via the ``new_latt_vec`` parameter.

        Examples
        --------
        Modify slab model so that non-periodic third vector is perpendicular to the slab

        >>> tb.change_nonperiodic_vector(2)
        """
        self._lattice.change_nonperiodic_vector(fin_dir, new_latt_vec)

        if to_home:
            self._shift_hop_to_home()
            self._lattice._shift_orb_to_home()

    def make_supercell(
        self,
        sc_red_lat,
        return_sc_vectors: bool=False,
        to_home: bool=True,
    ) -> "TBModel":
        """Make model on a super-cell.

        .. versionchanged:: 2.0.0
            Parameter `to_home_supress_warning` has been removed.

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
            to the home cell along periodic directions.

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
        geom = self._lattice._prepare_supercell_geometry(sc_red_lat)

        # get super-cell vectors in cartesian coordinates
        sc_vec = geom["translations"]
        num_sc = sc_vec.shape[0]
        

        lat = Lattice(geom["lat_vecs"], geom["orb_vecs"], self.periodic_dirs)
        sc_tb = TBModel(lat, spinful=self.spinful)
        sc_tb._assume_position_operator_diagonal = self._assume_position_operator_diagonal

        for offset in range(num_sc):
            base = offset * self.norb
            for orb_idx, onsite in enumerate(self._site_energies):
                sc_tb.set_onsite(onsite, base + orb_idx)

        sc_index = {tuple(vec.tolist()): idx for idx, vec in enumerate(sc_vec)}
        sc_red_lat = geom["sc_red_lat"]
        red_transform = geom["red_transform"]
        eps = 1e-8

        amps, ind_is, ind_js, R_vecs = self._hoppings.components()

        for offset, cur_sc_vec in enumerate(sc_vec):
            base = offset * self.norb
            for amp, ind_i, ind_j, ind_R in zip(amps, ind_is, ind_js, R_vecs, strict=True):
                R_vec = ind_R.copy()
                total_disp = cur_sc_vec + R_vec
                red_disp = total_disp @ red_transform
                sc_part = np.floor(red_disp + eps).astype(int)
                orig_part = total_disp - sc_part @ sc_red_lat
                pair_idx = sc_index.get(tuple(orig_part.tolist()))
                if pair_idx is None:
                    raise Exception("\n\nDid not find super cell vector!")

                hi = int(ind_i) + base
                hj = int(ind_j) + pair_idx * self.norb

                if self._nspin == 2:
                    amp_use = amp.copy()
                else:
                    amp_use = complex(amp)

                sc_tb.set_hop(
                    amp_use, hi, hj, sc_part, mode="add", allow_conjugate_pair=True
                )

        if to_home:
            #NOTE: These two functions must be called in this order! 
            # First shift hoppings, then orbitals. The hoppings
            # depend on the orbital positions.

            sc_tb._shift_hop_to_home()
            sc_tb._lattice._shift_orb_to_home()

        return sc_tb if not return_sc_vectors else (sc_tb, sc_vec.copy())
                            

    def _shift_hop_to_home(self):
        """Shifts orbital coordinates (along periodic directions) to the home
        unit cell. 
        
        After this function is called reduced coordinates
        (along periodic directions) of orbitals will be between 0 and
        1.

        .. versionchanged:: 1.7.2
            Versions < 1.7.2 shifted orbitals to the home cell even
            along even nonperiodic directions. In later versions, this is
            no longer allowed, as this might produce
            counterintuitive results. Shifting orbitals along nonperiodic
            directions changes physical nature of the tight-binding model.
            This behavior might be especially non-intuitive for
            tight-binding models that came from the `cut_piece` function.
        
        """

        for i in range(self.norb):
            disp_vec = np.zeros(self.dim_r, dtype=int)
            for k in range(self.dim_r):
                shift = int(np.floor(self.orb_vecs[i, k]))
                if k in self.per:
                    disp_vec[k] = shift
                elif shift != 0:
                    logger.warning(
                        f"Orbital {i} has reduced coordinate {self.orb_vecs[i,k]:.4f} along non-periodic direction {k}. "
                        "This orbital will not be shifted to the home cell along this direction."
                    )

            if self.dim_k != 0 and np.any(disp_vec):
                self._hoppings.shift_orbital(i, disp_vec)

    def add_orb(self, orb_pos):
        """Adds a new orbital to the model with the specified coordinates.
        
        The orbital coordinate must be given in reduced
        coordinates, i.e. in units of the real-space lattice vectors
        of the model. The new orbital is added at the end of the list
        of orbitals, and the orbital index is set to the next available
        index.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        orb_pos : array_like, float
            The reduced coordinates of the new orbital of length `dim_r`. If
            ``orb_pos`` is a single float or int, it will be converted to a 1D array
            (`dim_r` must be 1).
        """

        # Append orbital position
        self._lattice.add_orb(orb_pos)

        # Append default site energy and specified flag
        if not self.spinful:
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

        Notes
        -----
        Removing orbitals will reindex the orbitals with indices higher
        than those that are removed. For example, if model has 6 orbitals
        and you remove the 2nd orbital, then the orbitals 3-6 will be
        reindexed to 2-5 (Python counting). Indices of first two orbitals (0 and 1) 
        are unaffected.
         
        Examples
        --------
        If original_model has say 10 orbitals then returned small_model will 
        have only 8 orbitals.

        >>> small_model = original_model.remove_orb([2,5])

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

        self._lattice.remove_orb(orb_index)

        # remove indices one by one
        for _, orb_ind in enumerate(orb_index):
            self._site_energies = np.delete(self._site_energies, orb_ind, 0)
            self._site_energies_specified = np.delete(
                self._site_energies_specified, orb_ind
            )

        self._hoppings.remove_orbitals(orb_index)

    @copydoc(Lattice.k_uniform_mesh)
    def k_uniform_mesh(self, mesh_size):
        return self._lattice.k_uniform_mesh(mesh_size)

    @copydoc(Lattice.k_path)
    def k_path(self, kpts, nk:int, report:bool=True):
        return self._lattice.k_path(kpts, nk, report)   
    
    def ignore_position_operator_offdiagonal(self):
        """Set flag to ignore off-diagonal elements of the position operator.

        Call to this function enables one to approximately compute
        Berry-like objects from tight-binding models that were
        obtained from Wannier90.
        """
        self._assume_position_operator_diagonal = True

    def position_matrix(self, evecs: np.ndarray, dir: int):
        r"""Position operator matrix elements

        Returns matrix elements of the position operator along
        direction `dir` for eigenvectors `evecs` at a single k-point.
        Position operator is defined in reduced coordinates.

        The returned object :math:`X` is

        .. math::

          X_{m n {\bf k}}^{\alpha} = \langle u_{m {\bf k}} \vert
          r^{\alpha} \vert u_{n {\bf k}} \rangle

        Here :math:`r^{\alpha}` is the position operator along direction
        :math:`\alpha` that is selected by `dir`.

        .. versionchanged:: 2.0.0
            Parameter `evec` renamed to `evecs` to clarify that multiple
            eigenvectors can be passed at once.

        Parameters
        ----------
        evecs : np.ndarray
            Eigenvectors for which we are computing matrix
            elements of the position operator.  The shape of this array
            is ``evecs[band, orbital]`` if ``spinful=False`` and
            ``evecs[band, orbital, spin]`` if ``spinful=True``.

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
        if dir in self.per:
            raise ValueError(
                "Can not compute position matrix elements along periodic direction!"
            )
        # make sure direction is not out of range
        if dir < 0 or dir >= self.dim_r:
            raise ValueError("Direction out of range!")

        # check if model came from w90
        if not self._assume_position_operator_diagonal:
            _offdiag_approximation_warning_and_stop()

        # check shape of evec
        if not isinstance(evecs, np.ndarray):
            raise TypeError("evec must be a numpy array.")
        # check number of dimensions of evec
        if not self.spinful:
            if evecs.ndim != 2:
                raise ValueError(
                    "evec must be a 2D array with shape (band, orbital) for spinless models."
                )
        elif self.spinful:
            if evecs.ndim != 3:
                raise ValueError(
                    "evec must be a 3D array with shape (band, orbital, spin) for spinful models."
                )

        # get coordinates of orbitals along the specified direction
        pos_tmp = self.orb_vecs[:, dir]
        # reshape arrays in the case of spinful calculation
        if self.spinful:
            # tile along spin direction if needed
            pos_use = np.tile(pos_tmp, (2, 1)).transpose().flatten()
            evec_use = evecs.reshape(evecs.shape[0], -1) # flatten spin index
        else:
            pos_use = pos_tmp
            evec_use = evecs

        # <u_i | r | u_j> = sum_orb r_orb u_i*(orb) u_j(orb)
        pos_mat = np.einsum("im,m,jm->ij", evec_use.conj(), pos_use, evec_use)

        # make sure matrix is Hermitian
        if not np.allclose(pos_mat, pos_mat.T.conj()):
            raise ValueError("Position matrix is not Hermitian.")

        return pos_mat

    def position_expectation(self, evecs: np.ndarray, dir: int):
        r"""Returns diagonal matrix elements of the position operator.
        
        These elements :math:`X_{n n}` can be interpreted as an
        average position of n-th Bloch state ``evec[n]`` along
        direction `dir`. 

        .. versionchanged:: 2.0.0
            Parameter `evec` renamed to `evecs` to clarify that multiple
            eigenvectors can be passed at once.

        Parameters
        ----------
        evecs : np.ndarray
            Eigenvectors for which we are computing matrix
            elements of the position operator. The shape of this array
            is ``evecs[band, orbital]`` if ``spinful=True`` and
            ``evecs[band, orbital, spin]`` if ``spinful=False``.

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

        pos_exp = self.position_matrix(evecs, dir).diagonal()
        return np.array(np.real(pos_exp), dtype=float)

    def position_hwf(
            self, 
            evecs: np.ndarray, 
            dir: int, 
            hwf_evec=False, 
            basis="orbital"
            ):
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

        .. versionchanged:: 2.0.0
            Parameter `evec` renamed to `evecs` to clarify that multiple
            eigenvectors can be passed at once.

        Parameters
        ----------
        evecs : np.ndarray
            Eigenvectors for which we are computing matrix
            elements of the position operator. The shape of this array
            is ``evecs[band, orbital]`` if ``spinful=True`` and
            ``evecs[band, orbital, spin]`` if ``spinful=False``.
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
            if ``spinful=True``) give the amplitudes of the i-th hybrid Wannier function on
            the specified basis function. 

        Returns
        -------
        hwfc : np.ndarray
            Eigenvalues of the position operator matrix :math:`X`
            (also called hybrid Wannier function centers).
            Length of this vector equals number of bands given in ``evecs``
            input array. Hybrid Wannier function centers are ordered in ascending order.
            Note that in general `n`-th hwfc does not correspond to `n`-th
            state in ``evecs``.
        hwf : np.ndarray
            Eigenvectors of the position operator matrix :math:`X`.
            (also called hybrid Wannier functions).  These are returned only if
            parameter ``hwf_evec = True``.

            The shape of this array is ``[h,x]`` or ``[h,x,s]`` depending on value of
            ``basis`` and ``spinful``.

            - If ``basis`` is "bloch" then ``x`` refers to indices of
              Bloch states.
            - If ``basis`` is "orbital" then ``x`` (or ``x`` and ``s``)
              correspond to orbital index (or orbital and spin index if ``spinful=True``).

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
        average positions of the Bloch states ``evecs``, which are
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
        pos_mat = self.position_matrix(evecs=evecs, dir=dir)

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
                    ret_hwf = np.zeros((hwf.shape[0], self.norb), dtype=complex)
                    # sum over bloch states to get hwf in orbital basis
                    for i in range(ret_hwf.shape[0]):
                        ret_hwf[i] = np.dot(hwf[i], evecs)
                    hwf = ret_hwf
                else:
                    ret_hwf = np.zeros((hwf.shape[0], self.norb * 2), dtype=complex)
                    # get rid of spin indices
                    evec_use = evecs.reshape([hwf.shape[0], self.norb * 2])
                    # sum over states
                    for i in range(ret_hwf.shape[0]):
                        ret_hwf[i] = np.dot(hwf[i], evec_use)
                    # restore spin indices
                    hwf = ret_hwf.reshape([hwf.shape[0], self.norb, 2])
                return (hwfc, hwf)
            else:
                raise ValueError(
                    "Basis must be either 'wavefunction', 'bloch', or 'orbital'"
                )

    def berry_curvature(
        self,
        k_pts,
        evals=None,
        evecs=None,
        occ_idxs=None,
        dirs="all",
        cartesian: bool = False,
        non_abelian: bool = False,
    ):
        r"""Compute the Berry curvature at a list of k-points via Kubo formula.

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
        The Berry curvature is only defined for models with at least 2 k-space dimensions
        (``dim_k >= 2``). The Berry curvature is computed using the Kubo formula, which
        requires knowledge of the velocity operator :math:`\partial_\mu H_k`. This operator
        is computed using the gradient of the Hamiltonian provided by :func:`grad_ham`.
        """

        if self.dim_k < 2:
            raise Exception(
                """
                Berry curvature in this context is only computed for k-space dimensions. 
                Must have dim_k >= 2.
                """
            )

        v_k = self.velocity(k_pts, cartesian=cartesian)  # (Nk, dim_k, n_orb, n_orb)
        # flatten spin axis if present
        new_shape = (v_k.shape[:2]) + (self.nstate, self.nstate)
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

        if not non_abelian:
            b_curv = np.trace(b_curv, axis1=-1, axis2=-2)
        if dirs == "all":
            return b_curv
        else:
            return b_curv[dirs]

    def chern(self, occ_idxs=None, dirs=(0, 1), nk=200):
        r"""Computes Chern number for occupied manifold.

        The Chern number is computed by integrating the Berry curvature
        over a 2d surface in reciprocal space defined by `dirs` parameter.
        The Chern number is given by

        .. math::
            C = \frac{1}{2\pi} \int_{\text{2d surface}} d^2k \, \Omega(k)

        where :math:`\Omega(k)` is the trace of the Berry curvature
        tensor over the occupied bands.

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
        chern : float
            Chern number for the occupied manifold.

        Notes
        -----
        This function only works for models with at least 2 k-space
        dimensions (``dim_k >= 2``). The Chern number is only defined
        for 2d surfaces in k-space, so `dirs` must be a tuple of
        length 2. The Chern number is guaranteed to be an integer
        (within numerical accuracy) if the occupied manifold is
        separated by an energy gap from the unoccupied manifold over
        the entire 2d surface in k-space.
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
            C_i = 4\pi \, \mathrm{Im} \left(P[X,P][Y,P]\right)_{ii},

        where :math:`P` is the projector onto occupied states, :math:`X,Y` are position
        operators, and :math:`i` is the orbital index. The local Chern marker
        is normalized by the unit cell volume, so that its spatial average
        gives the Chern number of the occupied manifold.

        Returns
        -------
        C_local : np.ndarray of shape (N,)
            Per-site local Chern marker.
        """
        if self.dim_k != 0:
            raise ValueError("Local Chern marker is only defined for real-space models (dim_k=0).")
        
        H = self.hamiltonian()
        coords = self.get_orb_vecs(cartesian=True)
        uc_vol = self.lattice.cell_volume

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
        k_nodes,
        k_node_labels=None,
        nk=101,
        fig=None,
        ax=None,
        proj_orb_idx=None,
        proj_spin=False,
        bands_label=None,
        scat_size=3,
        lw=2,
        lc="b",
        ls="solid",
        cmap="plasma",
        cbar=True,
    ):
        """Plot the band structure along a specified path in k-space.

        This function allows for customization of the plot, including projection of orbitals,
        spin projection, figure and axis objects, title, scatter size, line width,
        line color, line style, colormap, and whether to show a color bar.

        Parameters
        ----------
        k_nodes : list[list[float]]
            List of high symmetry points (in reduced units) to plot the bands through. 
            For example, ``[[0,0,0], [0, 1/2, 1/2]]``.
        k_node_labels : list[str], optional
            Labels of high symmetry points. Defaults to None.
        nk : int, optional
            Total number of k-points to sample along the path. Defaults to 101.
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
        scat_size : float, optional
            Size of the scatter points. Defaults to 3. Only relevant if
            `proj_spin` is True or `proj_orb_idx` is not None.
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
        cbar : bool, optional
            Whether to show a color bar. Defaults to True.
            Only relevant if `proj_spin` is True or `proj_orb_idx` is not None.

        Returns:
            fig : matplotlib.figure.Figure
            ax: matplotlib.axes.Axes
        """
        return plot_bands(
            self,
            k_nodes,
            nk=nk,
            ktick_labels=k_node_labels,
            bands_label=bands_label,
            proj_orb_idx=proj_orb_idx,
            proj_spin=proj_spin,
            fig=fig,
            ax=ax,
            scat_size=scat_size,
            lw=lw,
            lc=lc,
            ls=ls,
            cmap=cmap,
            cbar=cbar,
        )


# Backward-compatibility for legacy tb_model constructor
class tb_model(TBModel):
    """Deprecated alias for backward-compatibility with PythTB <= 1.8.

    This class preserves the old constructor signature:
        ``tb_model(dim_k, dim_r, lat=None, orb=None, per=None, nspin=1)``

    Use ``TBModel(lattice, spinful)`` going forward.
    """
    def __init__(self, dim_k, dim_r, lat=None, orb=None, per=None, nspin=1):
        warnings.warn(
            "pythtb.tb_model is deprecated and will be removed in a future release. "
            "Use TBModel instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Build a Lattice from v1-style arguments
        if not isinstance(dim_k, int):
            raise TypeError("dim_k must be an int in tb_model-compatible constructor")
        if not isinstance(dim_r, int):
            raise TypeError("dim_r must be an int in tb_model-compatible constructor")
        if dim_k < 0 or dim_k > 4:
            raise ValueError("dim_k must be between 0 and 4")
        if dim_r < dim_k or dim_r > 4:
            raise ValueError("dim_r must satisfy dim_r >= dim_k and <= 4")

        # Lattice vectors
        if (isinstance(lat, str) and lat == 'unit') or lat is None:
            lat_vecs = np.identity(dim_r, float)
        else:
            lat_vecs = np.array(lat, dtype=float)
            if lat_vecs.shape != (dim_r, dim_r):
                raise ValueError("lat must have shape (dim_r, dim_r)")
            det = np.linalg.det(lat_vecs)
            if abs(det) < 1.0e-12:
                raise ValueError("lattice vectors have near-zero volume")

        # Orbital positions (reduced coordinates)
        if (isinstance(orb, str) and orb == 'bravais') or orb is None:
            orb_vecs = np.zeros((1, dim_r), dtype=float)
        elif isinstance(orb, (int, np.integer)):
            orb_vecs = np.zeros((int(orb), dim_r), dtype=float)
        else:
            orb_vecs = np.array(orb, dtype=float)
            if orb_vecs.ndim != 2 or orb_vecs.shape[1] != dim_r:
                raise ValueError("orb must be (norb, dim_r) in reduced coords")

        # Periodic directions
        if per is None:
            periodic_dirs = list(range(dim_k))
        else:
            periodic_dirs = list(per)
            if len(periodic_dirs) != dim_k:
                raise ValueError("len(per) must equal dim_k")

        # Construct new-style Lattice and delegate to TBModel
        lat_obj = Lattice(lat_vecs, orb_vecs, periodic_dirs=periodic_dirs)
        spinful = nspin == 2
        super().__init__(lattice=lat_obj, spinful=spinful)
