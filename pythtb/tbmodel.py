from collections.abc import Mapping
import copy
import logging
import warnings
import numpy as np
import inspect
from itertools import product
from .plotting import plot_bands, plot_tb_model, plot_tb_model_3d
from .utils import (
    _offdiag_approximation_warning_and_stop, 
    is_Hermitian, 
    deprecated, 
    copydoc, 
    finite_difference,
    levi_civita,
    _maybe_pad
    )
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


def _iter_params_of_callable(f):
        """Yield explicit parameter names (ignore *args/**kwargs)."""
        sig = inspect.signature(f)
        for name, param in sig.parameters.items():
            if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                if param.default is inspect._empty:  # user must supply
                    yield name

def _call_provider(prov, params):
    """Call a provider with only the kwargs it actually declares (unless it has **kwargs)."""
    if not callable(prov):
        raise TypeError("Provider is not callable.")
    sig = inspect.signature(prov)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return prov(**params)  # accepts anything
    kwargs = {k: params[k] for k in _iter_params_of_callable(prov) if k in params}
    return prov(**kwargs)

def _describe_provider(provider):
    if isinstance(provider, str):
        return f"'{provider}'"
    if callable(provider):
        try:
            src = inspect.getsource(provider).strip()
            return src
        except (OSError, TypeError, AttributeError):
            name = getattr(provider, "__qualname__", getattr(provider, "__name__", repr(provider)))
            return f"callable {name} (source unavailable)"
    return repr(provider)

class TBModel:
    r"""Tight-binding model.

    This class is the central entry point for defining and analyzing tight-binding Hamiltonians. 
    In addition to Hamiltonian construction and diagonalization, it provides methods for computing
    topological and quantum-geometric observables such as Berry curvature, quantum metric, 
    the quantum geometric tensor, the Chern number, and the Bianco–Resta local Chern marker.

    .. versionremoved:: 2.0.0
        Parameters ``dim_r`` and ``dim_k`` were removed. Real- and reciprocal-space 
        dimensions are inferred from :class:`Lattice`.

    Parameters
    ----------
    lattice : :class:`Lattice`
        Lattice structure (lattice vectors, orbital positions, periodic directions).
        Create a :class:`Lattice` object separately and pass it to `TBModel`.

        .. versionchanged:: 2.0.0
            Replaces parameters ``lat``, ``orb``, and ``per``.

    spinful : bool, optional
        If True, each orbital carries two spin components (spin-1/2). 
        If False, the model is spinless. Default is False.

        .. versionchanged:: 2.0.0
            Renamed from ``nspin`` to ``spinful`` and changed type to ``bool``.

    Examples
    --------
    Create a model that is two-dimensional in real space but one-dimensional in
    reciprocal space. The first lattice vector is ``[1, 1/2]`` and the second is
    ``[0, 2]``. The second lattice vector is periodic (``periodic_dirs=[1]``). Three
    orbital positions are given in reduced (fractional) coordinates.

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
        self.assume_position_operator_diagonal = True
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
        self._hoptable = HoppingTable(self.dim_r, spinful=spinful)

        self._onsite_param_terms = {}
        self._hopping_param_terms = {}

    def __repr__(self):
        r"""Return a concise representation of the model.

        Returns
        -------
        str
            A string like ``"pythtb.TBModel(dim_r=..., dim_k=..., norb=..., spinful=...)"``.
        """
        return (
            f"pythtb.TBModel(dim_r={self.dim_r}, dim_k={self.dim_k}, "
            f"norb={self.norb}, spinful={self.spinful})"
        )

    def __str__(self):
        r"""Return a human‑readable summary string.

        Returns
        -------
        str
            Same text as :meth:`info` with ``show=False``.
        """
        return self.info(show=False)

    # Property decorators for read-only access to model attributes

    @property
    def lattice(self) -> Lattice:
        """The :class:`Lattice` object associated with the model.

        .. versionadded:: 2.0.0

        Returns
        -------
        Lattice
            A copy of the :class:`Lattice` object associated with the model.
        """
        return copy.copy(self._lattice)
    
    @property
    def dim_r(self) -> int:
        """Dimensionality of real space.

        .. versionadded:: 2.0.0

        Returns
        -------
        int
            Number of Cartesian real-space directions.
        """
        return self.lattice.dim_r

    @property
    def dim_k(self) -> int:
        """Dimensionality of reciprocal space (number of periodic directions).

        .. versionadded:: 2.0.0

        Returns
        -------
        int
            Number of periodic directions.
        """
        return self.lattice.dim_k

    @property
    def nspin(self) -> int:
        """Number of spin components (1 for spinless, 2 for spinful).

        .. versionadded:: 2.0.0

        Returns
        -------
        int
            Number of spin components (1 for spinless, 2 for spinful).
        """
        return self._nspin

    @property
    def spinful(self) -> bool:
        """Whether the model includes spin degrees of freedom.

        .. versionadded:: 2.0.0

        Returns
        -------
        bool
            Whether the model includes spin degrees of freedom.
        """
        return self._nspin == 2

    @property
    def per(self) -> list[int]:
        """Alias of :attr:`periodic_dirs`.

        .. versionadded:: 2.0.0

        Returns
        -------
        list[int]
            Indices of periodic directions.
        """
        return copy.copy(self.periodic_dirs)
    
    @property
    def periodic_dirs(self) -> list[int]:
        """Periodic directions as indices into the lattice vectors.

        .. versionadded:: 2.0.0

        Returns
        -------
        list[int]
            Indices of periodic directions (length equals :attr:`dim_k`).
        """
        return self.lattice.periodic_dirs

    @property
    def norb(self) -> int:
        """Number of tight-binding orbitals per unit cell.

        .. versionadded:: 2.0.0

        Returns
        -------
        int
            Number of tight-binding orbitals per unit cell.
        """
        return copy.copy(self.lattice.norb)

    @property
    def nstate(self) -> int:
        """Total number of electronic states (``norb * nspin``).

        .. versionadded:: 2.0.0

        Returns
        -------
        int
            Total number of electronic states (``norb * nspin``).
        """
        return self.norb * self.nspin
    
    @property
    def orb_vecs(self) -> np.ndarray:
        """Orbital positions in reduced coordinates.

        .. versionadded:: 2.0.0

        Returns
        -------
        np.ndarray
            A copy of the orbital positions in reduced coordinates. 
            Shape is ``(norb, dim_r)``.
        """
        return copy.copy(self.lattice.orb_vecs)

    @property
    def lat_vecs(self) -> np.ndarray:
        """Lattice vectors in Cartesian coordinates.

        .. versionadded:: 2.0.0
        Returns
        -------
        np.ndarray
            A copy of the lattice vectors in Cartesian coordinates.
            Shape is ``(dim_r, dim_r)``.
        """
        return copy.copy(self.lattice.lat_vecs)
    
    @property
    def recip_lat_vecs(self) -> np.ndarray:
        """Reciprocal lattice vectors in inverse Cartesian units.

        .. versionadded:: 2.0.0

        Returns
        -------
        np.ndarray
            A copy of the reciprocal lattice vectors in inverse Cartesian units.
            Shape is ``(dim_k, dim_k)``.
        """
        return copy.copy(self.lattice.recip_lat_vecs)

    @property
    def recip_volume(self) -> float:
        """Volume of the reciprocal unit cell in inverse Cartesian units.

        .. versionadded:: 2.0.0

        Returns
        -------
        float
            Volume of the reciprocal unit cell in inverse Cartesian units.
        """
        return copy.copy(self.lattice.recip_volume)

    @property
    def cell_volume(self) -> float:
        """Volume of the real-space unit cell in Cartesian units.

        .. versionadded:: 2.0.0

        Returns
        -------
        float
            Volume of the real-space unit cell in Cartesian units.
        """
        return copy.copy(self.lattice.cell_volume)

    @property
    def site_energies(self) -> np.ndarray:
        """On-site energies for each orbital. 

        .. versionadded:: 2.0.0

        Returns
        -------
        np.ndarray
            A copy of the on-site energies for each orbital.
            Shape is ``(norb,)`` for spinless models, 
            ``(norb, 2, 2)`` for spinful models.
        """
        return self._site_energies.copy()

    @property
    def hoppings(self) -> list[dict]:
        """Hopping terms defined in the model.

        .. versionadded:: 2.0.0

        Returns
        -------
        list[dict]
            One dictionary per hopping with keys:

            - ``"amplitude"`` : hopping amplitude (complex or 2x2 numpy.ndarray for spinful)
            - ``"from_orbital"``: index of starting orbital (int)
            - ``"to_orbital"``: index of ending orbital (int)
            - ``"lattice_vector"``: lattice vector displacement for ``"to_orbital"`` (list of int)
        """
        amps, i_idx, j_idx, R_vecs = self._hoptable.components()
        formatted: list[dict] = []
        for hop_idx in range(self.nhops):
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
    def nhops(self) -> int:
        """Number of hoppings defined in the model.

        .. versionadded:: 2.0.0

        Returns
        -------
        int
            Number of hoppings defined in the model.
        """
        return len(self._hoptable)

    @property
    def parameters(self):
        """Parameter providers registered on on‑site and hopping terms.

        Returns
        -------
        list[dict]
            Each entry describes a provider with the following fields:

            - ``kind``: ``"onsite"`` or ``"hopping"``
            - ``orbitals``: index (onsite) or ``(i, j)`` tuple (hopping)
            - ``R``: lattice-vector tuple for hoppings
            - ``names``: list[str] of parameter names required by the provider
            - ``source`` (optional): best-effort textual description of the callable
            - ``function`` (optional): the callable itself
        """
        out = []
        for idx, provider in getattr(self, "_onsite_param_terms", {}).items():
            if provider is None:
                continue
            names = self._provider_names(provider, ctx=f"onsite[{idx}]")
            desc = {"kind": "onsite", "orbitals": int(idx), "names": names}
            if callable(provider):
                desc["source"] = _describe_provider(provider)
                desc["function"] = provider
            out.append(desc)

        for (i, j, R), provider in getattr(self, "_hopping_param_terms", {}).items():
            if provider is None:
                continue
            names = self._provider_names(provider, ctx=f"hopping[{i},{j},{tuple(R)}]")
            desc = {"kind": "hopping", "orbitals": (i, j), "R": tuple(R), "names": names}
            if callable(provider):
                desc["source"] = _describe_provider(provider)
                desc["function"] = provider
            out.append(desc)

        return out
    
    @property
    def from_w90(self) -> bool:
        """Whether the model was constructed from :class:`W90`.

        .. versionadded:: 2.0.0

        Returns
        -------
        bool
            Whether the model was constructed from :class:`W90` and
            comes from a Wannier90 calculation.
        """
        return self._from_w90

    @property
    def assume_position_operator_diagonal(self) -> bool:
        """Whether the position operator is assumed to be diagonal in the orbital basis.
        
        Returns
        -------
        bool
            Whether the position operator is assumed to be diagonal in the orbital basis.
        """
        return self._assume_position_operator_diagonal

    @assume_position_operator_diagonal.setter
    def assume_position_operator_diagonal(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("assume_position_operator_diagonal must be a boolean.")
        self._assume_position_operator_diagonal = value

    @deprecated(
        "The 'display()' method is deprecated and will be removed in a future release. " \
        "Use 'print(model)' or 'model.info(show=True)' instead."
    )
    def display(self):
        r"""
        .. deprecated:: 2.0.0
            Use ``print(model)`` or :meth:`info` instead.
        """
        return self.info(show=True)

    def info(self, show: bool = True, short: bool = False):
        r"""Print or return a textual report describing the model.

        Parameters
        ----------
        show : bool, optional
            If True, print to stdout and return ``None``. 
            If False, return the string without printing.
            Default is True.

            .. versionadded:: 2.0.0

        short : bool, optional
            If True, include only a lattice summary.
            If False, also include site-energies and hopping information.
            Default is False.

            .. versionadded:: 2.0.0

        Returns
        -------
        str or None
            The report string if ``show`` is False; otherwise ``None``.

        Notes
        -----
        The report includes lattice vectors, orbital positions, spin information,
        site energies, hopping terms, and hopping distances (when available).

        Examples
        --------
        >>> text = tb.info(show=False, short=True)
        >>> isinstance(text, str)
        True
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
        lat_report.insert(3, f"spinful                     = {self.spinful}")
        lat_report.insert(4, f"number of spin components   = {self.nspin}")
        lat_report.insert(5, f"number of electronic states = {self.nstate}")
        output.extend(lat_report)

        if not short:
            # Print Site Energies
            output.append("Site energies:")
            onsite_params = getattr(self, "_onsite_param_terms", {})

            for i, site in enumerate(self._site_energies):
                onsite_param_term = onsite_params.get(i)
                if onsite_param_term:
                    output.append(f"  # {i:<3} ===> {_describe_provider(onsite_param_term)}")
                    continue

                if self._nspin == 1:
                    energy_str = f"{site:^7.3f}"
                else:
                    energy_str = str(site).replace("\n", " ")

                output.append(f"  # {i:<3} ===> {energy_str}")

            amps, i_idx, j_idx, R_vecs = self._hoptable.components()

            output.append("Hoppings:")
            for hop_idx in range(self.nhops):
                hop_from = int(i_idx[hop_idx])
                hop_to = int(j_idx[hop_idx])
                R_vec = R_vecs[hop_idx]

                coords = ", ".join(f"{value:^5.1f}" for value in R_vec)
                disp = f" + [{coords}] " if self.dim_k else ""
                out_str = f"  <{hop_from:^3}| H |{hop_to:^3}{disp}>  ===> "

                amp = amps[hop_idx]
                if self.spinful:
                    amp_str = str(np.asarray(amp).round(4)).replace("\n", " ")
                else:
                    amp_str = f"{complex(amp):^7.4f}"

                out_str += amp_str
                output.append(out_str)

            param_hops = getattr(self, "_hopping_param_terms", {})
            if param_hops:
                for (hop_from, hop_to, R_tup), provider in param_hops.items():
                    coords = ", ".join(f"{value:^5.1f}" for value in R_tup) if len(R_tup) else ""
                    disp = f" + [{coords}] " if (coords and self.dim_k) else ""
                    output.append(
                        f"  <{hop_from:^3}| H |{hop_to:^3}{disp}>  ===> {_describe_provider(provider)}"
                    )

            output.append("Hopping distances:")
            orb_cart = self.get_orb_vecs(cartesian=True)
            lat_vecs = np.asarray(self.lat_vecs, dtype=float)

            if self.nhops:
                for hop_idx in range(self.nhops):
                    hop_from = int(i_idx[hop_idx])
                    hop_to = int(j_idx[hop_idx])
                    R_vec = R_vecs[hop_idx]

                    pos_i = orb_cart[hop_from]
                    pos_j = orb_cart[hop_to] + R_vec @ lat_vecs

                    coords = ", ".join(f"{value:^5.1f}" for value in R_vec)
                    disp = f" + [{coords}]" if self.dim_k else ""
                    distance = np.linalg.norm(pos_j - pos_i)

                    output.append(
                        f"  | pos({hop_from:^3}) - pos({hop_to:^3}){disp} | = {distance:7.3f}"
                    )

            for (hop_from, hop_to, R_tup), _ in param_hops.items():
                R_vec = np.asarray(R_tup, dtype=float)

                pos_i = orb_cart[hop_from]
                pos_j = orb_cart[hop_to] + R_vec @ lat_vecs

                coords = ", ".join(f"{value:^5.1f}" for value in R_vec)
                disp = f" + [{coords}]" if self.dim_k else ""
                distance = np.linalg.norm(pos_j - pos_i)

                output.append(
                    f"  | pos({hop_from:^3}) - pos({hop_to:^3}){disp} | = {distance:7.3f} (param)"
                )

        if show:
            print("\n".join(output))
        else:
            return "\n".join(output)

    def copy(self) -> "TBModel":
        """Return a deep copy of the model.

        .. versionadded:: 2.0.0

        Returns
        -------
        TBModel
            Independent copy of ``self``.

        Examples
        --------
        >>> tb2 = tb.copy()
        >>> tb2 is tb
        False
        """
        return copy.deepcopy(self)

    def clear_hoppings(self):
        """Remove all hopping terms from the model.

        .. versionadded:: 2.0.0

        Notes
        -----
        Useful for resetting the model to a state without any hoppings.
        """
        self._hoptable.clear()
        logger.info("Cleared all hoppings.")

    def clear_onsite(self):
        """Reset all on-site energies to zero.

        .. versionadded:: 2.0.0

        Notes
        -----
        Also clears the internal flags that track which sites were explicitly set.
        """
        self._site_energies.fill(0)
        self._site_energies_specified.fill(False)
        logger.info("Cleared all on-site energies.")

    @deprecated("Use 'norb' property instead.")
    def get_num_orbitals(self):
        """
        .. deprecated:: 2.0.0
           Use :attr:`norb` instead.
        """
        return self.norb

    @deprecated("Use 'get_orb_vecs' instead.")
    def get_orb(self):
        """
        .. deprecated:: 2.0.0
           Use :meth:`get_orb_vecs` instead.
        """
        return self.get_orb_vecs(cartesian=False)

    def get_orb_vecs(self, cartesian=False):
        """Return orbital positions.

        .. versionchanged:: 2.0.0
            The name was changed from `get_orb` to `get_orb_vecs`.

        Parameters
        ----------
        cartesian : bool, optional
            If True, returns orbital positions in Cartesian coordinates.
            If False, returns reduced coordinates (default).

            .. versionadded:: 2.0.0

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
           Use :meth:`get_lat_vecs` instead.
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
        r"""Define on-site energies for tight-binding orbitals.

        You can set the energy for a single orbital (by specifying ``ind_i``),
        or for all orbitals at once by passing ``onsite_en`` as a list/array 
        of length :attr:`norb`.

        .. deprecated:: 2.0.0
            ``mode="reset"`` is deprecated. Use ``mode="set"`` instead.

        Parameters
        ----------
        onsite_en : float, array_like, (2, 2) numpy.ndarray, str, callable
            .. versionadded:: 2.0.0
                Symbolic expressions and callables for onsite energies.

            On-site energy value(s) to set. When specifying a single orbital
            with ``ind_i``, this may be one of the following:
            
            **Symbolic (spinless or spinful)**

            - String: A symbolic expression for the onsite energy.
            - Callable: A function of a single parameter that returns the onsite energy
              in one of the formats listed below.
            
            **Spinless**  (``spinful=False``)

            - Real scalar.

            **Spinful**  (``spinful=True``)

            - **Scalar** ``a``: interpreted as :math:`a I` (same value for both spins).
            - **4-vector** ``[a, b, c, d]``:
              
              .. math::
                a I + b\,\sigma_x + c\,\sigma_y + d\,\sigma_z =
                \begin{bmatrix}
                a + d & b - i c \\
                b + i c & a - d
                \end{bmatrix}

            - **Full matrix**: a 2 x 2 Hermitian ndarray.

            When specifying all orbitals at once (``ind_i=None``), this should be
            a list or array of length :attr:`norb`, where each entry is one of the
            above types.

        ind_i : int, optional
            Orbital index to update. If unspecified, all orbitals are updated and
            ``onsite_en`` must be a sequence of length :attr:`norb`.

        mode : {'set', 'add'}, optional
            How to apply ``onsite_en``.

            - ``'set'``: replace the value(s). (Default)
            - ``'add'``: add to existing value(s).

        See Also
        --------
        set_hop : Define hopping amplitudes between orbitals.

        Notes
        -----
        - When called multiple times with ``mode='add'``, values accumulate.

        Examples
        --------
        Setting all on-site energies at once:

        >>> tb.set_onsite([0.0, 1.0, 2.0]) 

        Adding an on-site energy to a single orbital:

        >>> tb.set_onsite(100.0, ind_i=1, mode="add")  

        Setting the on-site energy of a single orbital:

        >>> tb.set_onsite(0.0, ind_i=1, mode="set")

        Setting a spinful on-site term using a 4-vector:

        >>> tb.set_onsite([1.0, 0.2, 0.0, -0.1], ind_i=0)

        Parametric onsite energy using a string expression:

        >>> tb.set_onsite("mA", ind_i=0)

        Parametric onsite energy using a callable:

        >>> tb.set_onsite(lambda mA: mA**2, ind_i=0)

        Setting all on-site energies with parametric callables:

        >>> tb.set_onsite([lambda mA: mA, lambda mA: 2*mA, lambda mA: 3*mA])

        This will set a single parameter ``"mA"``, which can be set to a single value
        using :meth:`set_parameters`, or with a list of values in downstream functions
        like :meth:`hamiltonian` or :meth:`velocity`.
        """
        # Handle deprecated 'reset' mode
        mode = mode.lower()
        if mode == "reset":
            logger.warning(
                "The 'reset' mode is deprecated as of v2.0. Use 'set' instead to set the onsite energy."
                "This will be removed in a future version."
            )
            mode = "set"

        def _process_single(val):
            # Accept callables (e.g., lambdas) – defer checks to evaluation time
            if callable(val):
                return ("callable", val)

            # Back-compat string path
            if isinstance(val, str):
                return ("expr", val)

            # Numeric/array/matrix path – convert to canonical onsite block now
            block = self._val_to_block(val)
            if self.nspin == 2 and not is_Hermitian(block):
                raise ValueError("Onsite terms should be real, or Hermitian for spinful models.")
            return ("block", block)

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
            
            items = list(onsite_en)
            indices = np.arange(self.norb)
        else:
            if not (0 <= ind_i < self.norb):
                raise ValueError(
                    "Index ind_i is not within the range of number of orbitals."
                )

            items = [onsite_en]
            indices = [ind_i]

        processed = [_process_single(v) for v in items]

        if mode == "set":
            for idx, (kind, payload) in zip(indices, processed):
                if self._site_energies_specified[idx]:
                    logger.warning(
                        f"Onsite energy for site {idx} was already set; resetting to the specified values."
                    )

                if kind == "block":
                    self._site_energies[idx] = payload
                    self._site_energies_specified[idx] = True
                    # Clear any previous param providers
                    self._onsite_param_terms[idx] = None
                else:
                    # callable/expr -> store for later evaluation
                    self._site_energies[idx] = 0  # numeric placeholder, unused at build time
                    self._site_energies_specified[idx] = False
                    self._onsite_param_terms[idx] = payload  # payload is callable or str

        elif mode == "add":
            for idx, (kind, payload) in zip(indices, processed):
                if kind == "block":
                    self._site_energies[idx] += payload
                    self._site_energies_specified[idx] = True
                    # 'add' with a concrete block keeps any prior callable/expr ignored at build time
                    self._onsite_param_terms[idx] = None
                else:
                    # Adding a callable/expr: we interpret as "replace provider" (cannot 'add' unevaluated safely).
                    logger.warning(
                        f"'add' with a callable/string provider on site {idx} replaces the previous provider."
                    )
                    self._site_energies[idx] = 0
                    self._site_energies_specified[idx] = False
                    self._onsite_param_terms[idx] = payload
        else:
            raise ValueError("Mode should be either 'set' or 'add'.")

    def _get_flattened_indices(self):
        return self._hoptable.flatten_cache(self.norb)

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
        and :math:`| \phi_{\mathbf{R},j} \rangle` is the j-th orbital in a cell shifted by lattice 
        vector :math:`\mathbf{R}`.

        .. deprecated:: 2.0.0
            Using ``"reset"`` for ``mode`` is deprecated, use ``"set"`` instead.

        Parameters
        ----------
        hop_amp : scalar, array-like, (2,2) numpy.ndarray, str, callable
            .. versionadded:: 2.0.0
                Symbolic expressions and callables for hoppings.

            **Symbolic (spinless or spinful)**

            - String: A symbolic expression for the onsite energy.
            - Callable: A callable with a single parameter that returns the onsite energy
              in one of the formats listed below.

            **Spinless** (``spinful=False``):

            - Real or complex scalar.

            **Spinful** (``spinful=True``):

            - Scalar ``a`` multiplied by identity: :math:`a I`.
            - 4-vector ``[a, b, c, d]`` dotted into identity and Pauli matrices: 

            .. math::
                a I + b\,\sigma_x + c\,\sigma_y + d\,\sigma_z =
                \begin{bmatrix}
                    a + d & b - i c \\
                    b + i c & a - d
                \end{bmatrix}

            - Full 2 x 2 Hermitian matrix.

        ind_i : int
            Index of bra orbital (in home unit cell).
        ind_j : int
            Index of ket orbital (in cell shifted by ``ind_R``).
        ind_R : array-like of int, optional
            Lattice vector (integer array, in reduced coordinates)
            pointing to the unit cell where the ket orbital is located.
            The number of coordinates must equal the dimensionality in
            real space (``dim_r``) for consistency, but only the periodic 
            directions of ``ind_R`` are used. If reciprocal space is zero-dimensional
            (as in a molecule), this parameter does not need to be specified.
        mode : {'set', 'add'}, optional
            How to apply ``hop_amp``.

            - ``'set'``: replace the value(s). (Default)
            - ``'add'``: add to existing value(s).

        allow_conjugate_pair : bool, optional
            If True, allows specification of both a hopping and its conjugate pair.
            If False, prevents double-counting.

        See Also
        --------
        set_onsite : Define on-site energies for orbitals.

        Notes
        -----
        - Unlike `set_onsite`, there is no way to bulk set hoppings; each hopping
          must be specified individually.
        - Strictly speaking, this term specifies hopping amplitude for hopping from site `j+R` to site `i`, 
          not vice-versa.
          There is no need to specify hoppings in both :math:`i \rightarrow j+\mathbf{R}` and
          :math:`j \rightarrow i-\mathbf{R}` directions, since the latter is included automatically as

          .. math::
              H_{ji}(-\mathbf{R}) = \left[ H_{ij}(\mathbf{R}) \right]^*
    
        - When called multiple times with ``mode='add'``, values accumulate.

        Examples
        --------
        Setting a hopping amplitude between orbital 0 in the home cell and orbital 1
        in the unit cell shifted by lattice vector ``R = [0, 1]``:

        >>> tb.set_hop(0.3+0.4j, 0, 2, [0, 1], mode="set")

        Adding to an existing hopping amplitude:

        >>> tb.set_hop(100.0, 0, 2, [0, 1], mode="add")

        Setting a spinful hopping using a 4-vector:

        >>> tb.set_hop([0.1, 0.0, 0.2, -0.1], 0, 1, [1, 0])

        Parametric hopping using a string expression:

        >>> tb.set_hop("t1", 0, 1, [1, 0])

        Parametric hopping using a callable:

        >>> tb.set_hop(lambda t1: t1**2, 0, 1, [1, 0])

        This will store a parameter ``"t1"``, which can be set to a single value
        using :meth:`set_parameters`, or with a list of values in downstream functions
        like :meth:`hamiltonian` or :meth:`velocity`.
        """
        #### Prechecks and formatting ####
        mode = mode.lower()
        if mode == "reset":
            logger.warning(
                "The 'reset' mode is deprecated as of v2.0. Use 'set' instead to set the hopping term."
                "This will be removed in a future version."
            )
            mode = "set"

        table = self._hoptable

        ind_i, ind_j, R_vec = table.normalize_entry(
            ind_i,
            ind_j,
            ind_R,
            norb=self.norb,
            dim_k=self.dim_k,
            periodic_dirs=self.periodic_dirs,
        )

        # Do not allow onsite hoppings to be specified here
        if ind_i == ind_j and (self.dim_k == 0 or bool(np.all(R_vec == 0))):
            raise ValueError(
                "Do not use set_hop for onsite terms. Use set_onsite instead."
            )

        key = (ind_i, ind_j, tuple(R_vec.tolist()))
        existing_idx = table.find(ind_i, ind_j, R_vec)

        def _process_amp(val):
            # Accept callables (e.g., lambdas) – defer evaluation to build time
            if callable(val):
                return ("callable", val)
            # string  
            if isinstance(val, str):
                return ("expr", val)
            # Numeric / array / matrix -> convert now
            block = self._val_to_block(val)  # may be complex; no Hermitian requirement for offsite
            return ("block", block)

        kind, payload = _process_amp(hop_amp)
        
        if not allow_conjugate_pair:
            conj_key = (ind_j, ind_i, tuple((-R_vec).tolist()))
            conj_idx = table.find(ind_j, ind_i, -R_vec)
            conj_in_providers = (conj_key in getattr(self, "_hopping_param_terms", {}))
            if conj_idx is not None or conj_in_providers:
                # If we're updating the exact same entry, allow it; otherwise error
                if existing_idx is None and key != conj_key:
                    raise ValueError(
                        f"Conjugate element already specified for i={ind_i}, j={ind_j}, R={R_vec.tolist()}. "
                        "Either avoid double entry or set allow_conjugate_pair=True."
                    )
                
        # Ensure provider dict exists
        if not hasattr(self, "_hopping_param_terms"):
            self._hopping_param_terms = {}

        if kind in ("callable", "expr"):
            # Provider path (deferred evaluation). We don't mix numeric state with parameters.
            if mode == "add":
                raise NotImplementedError(
                    "Adding parametric hopping terms is currently not supported."
                )
            # Remove any prior numeric or provider entry for this key
            if existing_idx is not None:
                table.remove(existing_idx) 
            self._hopping_param_terms[key] = payload  # callable or str
            return

        # Numeric/matrix path
        hop_use = payload  # already canonicalized by _val_to_block

        # If a provider existed at this key, numeric input overrides it
        if key in self._hopping_param_terms:
            logger.warning(f"Overriding existing param-dependent hopping at {key} with a numeric block.")
            self._hopping_param_terms.pop(key, None)

        if mode == "set":
            if existing_idx is not None:
                table.update(existing_idx, amplitude=hop_use, R=R_vec)
            else:
                table.append(hop_use, ind_i, ind_j, R_vec)
        elif mode == "add":
            if existing_idx is not None:
                table.add(existing_idx, hop_use)
            else:
                table.append(hop_use, ind_i, ind_j, R_vec)
        else:
            raise ValueError("Wrong value of mode parameter. Should be either `set` or `add`.")

    def set_shell_hops(self, shell_hops: dict, mode="set"):
        r"""Define n'th nearest-neighbor hoppings.

        This function sets hopping amplitudes for all bonds in specified
        nearest-neighbor shells. The shells are defined based on the distance
        from each orbital, with shell 1 being the nearest neighbors, shell 2
        being the next-nearest neighbors, and so on. All hoppings in a given shell
        are assigned the same value.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        shell_hops : dict[int, array-like]
            Dictionary mapping shell indices (counting from 1) to hopping amplitudes.
            The keys are integers representing the shell number, and the values
            are the hopping amplitudes for that shell.
        mode : {'set', 'add'}, optional
            Specifies how ``shell_hops`` is used
            - ``"set"``: Set the hopping term to the values in ``shell_hops``. (Default)
            - ``"add"``: Add the values in ``shell_hops`` to the previously set values.

        Notes
        -----
        - The hopping amplitudes are applied to all bonds in each shell.

        Examples
        --------
        Setting nearest-neighbor and next-nearest-neighbor hoppings:

        >>> tb.set_nn_hops({1: 1.0, 2: 0.5})

        Setting only nearest-neighbor hoppings:

        >>> tb.set_nn_hops({1: 1.0})

        """
        if not isinstance(shell_hops, dict):
            raise TypeError("shell_hops must be a dictionary mapping shell index to hopping amplitude.")
        
        nn_shells = list(shell_hops.keys())
        if len(nn_shells) == 0:
            raise ValueError("shell_hops must have at least one element.")
        if not all(isinstance(shell, int) and shell > 0 for shell in nn_shells):
            raise ValueError("Each element in nn_shells must be a positive integer.")

        max_shell = max(nn_shells)
        shell_bonds = self.nn_bonds(max_shell)[1]

        for shell_idx, shell in enumerate(shell_bonds):
            amp = shell_hops.get(shell_idx + 1, None)
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
        self._hoptable.extend(
            blocks,
            i_idx.tolist(),
            j_idx.tolist(),
            R_vecs.tolist(),
        )

    def _val_to_block(self, val):
        r"""
        Convert input value to appropriate matrix block for onsite or hopping.

        For spinful=False, returns the value (should be real or complex scalar).
        For spinful=True:
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

    def _clear_param_terms(self, onsite_idx=None, hop_key=None):
        if onsite_idx is not None:
            self._onsite_param_terms.pop(onsite_idx, None)
        if hop_key is not None:
            canonical = tuple(hop_key) if len(hop_key) == 3 else hop_key
            self._hopping_param_terms.pop(canonical, None)
        
    def _provider_names(self, provider, *, ctx):
        if callable(provider):
            params = tuple(_iter_params_of_callable(provider))
            if not params:
                raise ValueError(f"{ctx} callable must declare at least one parameter.")
            return params
        if isinstance(provider, str):
            return (provider,)
        raise TypeError(f"Unsupported {ctx} provider type: {type(provider)}")

    def _check_parameter_exists(self, params: dict):
        """Check if any parameters are defined in the model."""
        required = {name for entry in self.parameters for name in entry["names"]}
        provided = set(params.keys())

        unknown = provided - required
        if unknown:
            raise ValueError("Unknown parameter name(s): " + ", ".join(sorted(unknown)))

    def _check_missing_parameters(self, params: dict):
        """Check for missing parameters in the provided dictionary."""
        required = {name for entry in self.parameters for name in entry["names"]}
        provided = set(params.keys())

        unknown = provided - required
        if unknown:
            raise ValueError("Unknown parameter name(s): " + ", ".join(sorted(unknown)))

        missing = required - provided
        if missing:
            raise ValueError("Missing parameter value(s): " + ", ".join(sorted(missing)))
        
    def _params_to_sweep(self, params: dict):
        """Partition parameters into scalars and sweeps."""
        # Partition params into scalars vs sweeps (1D arrays/lists) 
        sweep_names: list[str] = []
        sweep_axes: list[list[object]] = []  # list of per-parameter lists of scalar values
        scalars: dict[str, object] = {}

        for name, raw in params.items():
            # Treat 1D arrays / lists / tuples as sweep axes; everything else as scalar
            if isinstance(raw, (list, tuple, np.ndarray)):
                raw = np.asarray(raw)

                if raw.ndim == 0:
                    # 0D array -> scalar
                    scalars[name] = raw.item()
                    continue
                elif raw.ndim == 1:
                    # keep raw values as-is so lambdas see original dtype (float/complex/etc.)
                    values = list(raw) if not isinstance(raw, np.ndarray) else [raw[i] for i in range(raw.shape[0])]
                    if len(values) == 0:
                        raise ValueError(f"Parameter sweep '{name}' must provide at least one value.")
                    sweep_names.append(name)
                    sweep_axes.append(values)
                elif raw.ndim == 2:
                    # Single-row array -> single value sweep
                    if raw.shape[0] == 1:
                        scalars[name] = raw[0, :].copy()
                    # Single-column array -> scalar sweep
                    elif raw.shape[1] == 1:
                        scalars[name] = raw[:, 0].copy()
                    # Full 2x2 matrix
                    elif raw.shape == (2, 2):
                        if not self.spinful:
                            raise ValueError(
                                f"Parameter '{name}' is a 2x2 array, but the model is spinless."
                            )
                        scalars[name] = raw.copy()
                    # Pauli 4-vector
                    elif raw.shape[1] == 4:
                        if not self.spinful:
                            raise ValueError(
                                f"Parameter '{name}' has shape {raw.shape}, but the model is spinless."
                            )
                        sweep_names.append(name)
                        sweep_axes.append([raw[i, :].copy() for i in range(raw.shape[0])])

            elif isinstance(raw, (int, float, complex)):
                # single scalar value
                scalars[name] = raw
            else:
                raise TypeError(
                    f"Parameter '{name}' has unsupported type {type(raw)}. "
                    "Expected scalar, list, tuple, or numpy.ndarray."
                )
            
        return scalars, sweep_names, sweep_axes

    def _evaluate_params(self, assignments: dict):
        """Evaluate the model with given parameter assignments.
        
        Parameters
        ----------
        assignments : dict
            Dictionary mapping parameter names to their values.
            The form is expected to match the parameters used in
            the symbolic expressions or callables defined in the model.
            For example, if the model has a parameter 't', then
            assignments should include an entry like {'t': 1.0}.
        
        Returns
        -------
        hop_amps : np.ndarray
            Hopping amplitudes after evaluation.
        i_idx : np.ndarray
            Indices of the 'from' orbitals.
        j_idx : np.ndarray
            Indices of the 'to' orbitals.
        R_vecs : np.ndarray
            Lattice vectors for each hopping.
        site : np.ndarray
            On-site energies after evaluation.
        """
        # ---- copy base hops and onsite (immutable per evaluation) ----
        base_hop_amps, base_i_idx, base_j_idx, base_R_vecs = self._hoptable.components()

        hop_amps = np.array(base_hop_amps, copy=True)      # copy!
        i_idx    = np.array(base_i_idx,    copy=True)
        j_idx    = np.array(base_j_idx,    copy=True)
        R_vecs   = np.array(base_R_vecs,   copy=True)

        site = np.asarray(self._site_energies, dtype=complex).copy()  # copy!
        # ---- onsite providers ----
        for idx, term in getattr(self, "_onsite_param_terms", {}).items():
            if term is None: 
                continue
            if callable(term):
                val   = _call_provider(term, assignments)
                block = self._val_to_block(val)
            elif isinstance(term, str):
                block = self._eval_expr_to_block(term, **assignments)
            else:
                raise TypeError("Unsupported onsite provider type.")
            if self.nspin == 2 and not is_Hermitian(block):
                raise ValueError(f"Onsite callable for site {idx} returned non-Hermitian 2×2.")
            site[idx] = np.asarray(block, dtype=complex)

        # collect param-dependent hops without mutating arrays in-place 
        dyn_blocks = []
        dyn_i = []
        dyn_j = []
        dyn_R = []

        for key, term in getattr(self, "_hopping_param_terms", {}).items():
            if callable(term):
                val   = _call_provider(term, assignments)
                block = self._val_to_block(val)
            elif isinstance(term, str):
                block = self._eval_expr_to_block(term, **assignments)
            else:
                raise TypeError("Unsupported hopping provider type.")
            dyn_blocks.append(np.asarray(block, dtype=complex)[None, ...])
            dyn_i.append(key[0])
            dyn_j.append(key[1])
            dyn_R.append(list(key[2]) if len(key) > 2 else [0]*self.dim_k)

        if dyn_blocks:
            hop_amps = np.concatenate([hop_amps] + dyn_blocks, axis=0)
            i_idx    = np.concatenate([i_idx,    np.asarray(dyn_i, dtype=i_idx.dtype)])
            j_idx    = np.concatenate([j_idx,    np.asarray(dyn_j, dtype=j_idx.dtype)])
            R_vecs   = np.concatenate([R_vecs,   np.asarray(dyn_R, dtype=R_vecs.dtype)], axis=0)

        return hop_amps, i_idx, j_idx, R_vecs, site

    def _normalize_parameter_axis(self, values, *, name, period=None):
        """
        Normalize a 1D parameter sweep and report metadata for finite differences.

        Returns
        -------
        values_unique : np.ndarray
            Copy of the input with any duplicated endpoint removed.
        step : float
            Uniform spacing between samples (computed before trimming so the “true” step is kept).
        is_periodic : bool
            True if the sweep spans a full cycle.
        trimmed : bool
            True when the final element was dropped because it duplicated the first.
        """
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError(f"Parameter '{name}' must be one-dimensional with at least two samples.")

        diffs = np.diff(arr)
        if not np.allclose(diffs, diffs[0]):
            raise ValueError(f"Parameter '{name}' must be uniformly spaced.")
        step = float(diffs[0])

        periodic = False
        trimmed = False
        if period is not None:
            period = float(period)
            span = arr[-1] - arr[0]
            if np.isclose(span, period):
                arr = arr[:-1]
                periodic = True
                trimmed = True
            elif np.isclose(step * arr.size, period):
                periodic = True
        else:
            if np.isclose(arr[-1], arr[0]):
                arr = arr[:-1]
                periodic = True
                trimmed = True

        return arr.copy(), step, periodic, trimmed

    
    def _eval_expr_to_block(self, expr: str, **assignments):
        """Evaluate a string expression with the given parameter values and cast it to a block."""
        env = {"np": np, "numpy": np, "pi": np.pi, "complex": complex, "float": float}
        env.update(assignments)
        try:
            value = eval(expr, {"__builtins__": {}}, env)
        except NameError as exc:
            missing = exc.args[0].split("'")[1]
            raise ValueError(f"Expression '{expr}' needs a value for parameter '{missing}'.") from None
        except Exception as exc:
            raise ValueError(f"Could not evaluate expression '{expr}': {exc}") from exc
        return self._val_to_block(value)

    def set_parameters(self, params=None, /, **kwargs):
        r"""
        Evaluate any parameter-dependent onsite or hopping term at the supplied scalar values.

        This helper is the inverse of declaring a parameterized term via `set_onsite` /
        `set_hop` with a callable or string provider. Each parameter name is mapped to
        the value that should be passed directly to the provider; once evaluated, the resulting
        block is fed back into `set_onsite`/`set_hop` so all standard validation (Hermiticity,
        conjugate handling, etc.) still applies.

        Parameters
        ----------
        params : mapping, optional
            Dictionary that maps parameter names to scalar values. Use this when a parameter
            name is not a valid Python identifier (for example, contains spaces or symbols).
            Mutually compatible with ``kwargs`` — entries from ``params`` are applied first,
            then any keyword arguments override them.
        **kwargs
            Additional parameter/value pairs. These are merged on top of ``params`` and are
            convenient when the parameter names are valid identifiers (e.g. ``model.set_parameters(beta=0.3)``).

        Notes
        -----
        - Only scalar values are accepted; 0-D NumPy arrays are unwrapped via ``.item()``.
        - Providers whose parameter lists are not completely covered by the supplied values
          are left untouched. In other words, you can freeze parameters incrementally.
        - After a parameter is set, the corresponding provider entry is removed so future
          Hamiltonian builds no longer depend on externally supplied values.

        Examples
        --------
        >>> tb.set_onsite(lambda m: [m, -m])  # both onsite terms become parametric
        >>> tb.set_hop(lambda t: t * np.eye(2), 0, 1, [0, 0, 0])
        >>> tb.set_parameters(m=0.4, t=1.2)    # both terms become numeric, providers removed
        """
        merged = {}
        if params is not None:
            if not isinstance(params, Mapping):
                raise TypeError("params must be a mapping of parameter names to values.")
            merged.update(params)
        merged.update(kwargs)
        if not merged:
            return

        cleaned = {}
        for name, value in merged.items():
            if isinstance(value, np.ndarray):
                if value.ndim != 0:
                    raise TypeError(f"Parameter '{name}' must be a scalar.")
                value = value.item()
            if not np.isscalar(value):
                raise TypeError(f"Parameter '{name}' must be a scalar.")
            cleaned[name] = value

        # onsite
        for idx, provider in list(getattr(self, "_onsite_param_terms", {}).items()):
            if provider is None:
                continue
            names = self._provider_names(provider, ctx=f"onsite[{idx}]")
            if any(name not in cleaned for name in names):
                continue
            if callable(provider):
                block = _call_provider(provider, {name: cleaned[name] for name in names})
            else:
                block = cleaned[names[0]]
            self.set_onsite(block, ind_i=idx, mode="set")        # reuse existing validation

        # hoppings
        for key, provider in list(getattr(self, "_hopping_param_terms", {}).items()):
            if provider is None:
                continue
            i, j, R = key
            names = self._provider_names(provider, ctx=f"hopping[{i},{j},{tuple(R)}]")
            if any(name not in cleaned for name in names):
                continue
            if callable(provider):
                block = _call_provider(provider, {name: cleaned[name] for name in names})
            else:
                block = cleaned[names[0]]

            if self.dim_k == 0:
                self.set_hop(block, i, j, mode="set", allow_conjugate_pair=True)
            else:
                self.set_hop(block, i, j, list(R), mode="set", allow_conjugate_pair=True)

    ###### Lattice manipulation #########
    
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
            The reduced coordinates of the new orbital of length ``dim_r``. If
            ``orb_pos`` is a single float or int, it will be converted to a 1D array
            (``dim_r`` must be 1).
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

        self._hoptable.remove_orbitals(orb_index)

    def cut_piece(self, num_cells, periodic_dir, glue_edges=False) -> "TBModel":
        r"""Cut a (d-1)-dimensional piece out of a d-dimensional tight-binding model.
        
        Constructs a (d-1)-dimensional tight-binding model out of a
        d-dimensional one by repeating the unit cell a given number of
        times along one of the periodic lattice vectors. 
        
        Parameters
        ----------
        num_cells : int
            How many times to repeat the unit cell.

            .. versionchanged:: 2.0.0
                Renamed from ``num`` for clarity.

        periodic_dir : int
            Index of the periodic lattice vector along which to make the system finite.

            .. versionchanged:: 2.0.0
                Renamed from ``fin_dir`` for clarity.

        glue_edges : bool, optional
            If True, allow hoppings from one edge to the other of a cut model.

        Returns
        -------
        cut_model : TBModel
            Object of type :class:`pythtb.TBModel` representing a cutout
            tight-binding model. 

        See Also
        ---------
        :ref:`cubic-slab-hwf-nb` : For an example
        :ref:`three-site-thouless-nb` : For an example

        Notes
        -----
        - Orbitals in ``cut_model`` are numbered so that the `i`-th orbital of the `n`-th unit 
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
        cut_model = TBModel(lat_fin, spinful=self.spinful)

        onsite = []  # store onsite energies
        for _ in range(num_cells):  # go over all cells in finite direction
            for j in range(self.norb):  # go over all orbitals in one cell
                # do the onsite energies at the same time
                onsite.append(self._site_energies[j])
        onsite = np.array(onsite)
        cut_model.set_onsite(onsite, mode="set")

        # replicate parameterised onsite providers
        onsite_providers = getattr(self, "_onsite_param_terms", {})
        if onsite_providers:
            for c in range(num_cells):
                base = c * self.norb
                for idx, provider in onsite_providers.items():
                    if provider:
                        cut_model.set_onsite(provider, ind_i=base + idx, mode="set")

        # remember if came from w90
        cut_model.assume_position_operator_diagonal = (
            self.assume_position_operator_diagonal
        )
        cut_model._from_w90 = self._from_w90

        amps, from_idx, to_idx, R_vecs = self._hoptable.components()
        for c in range(num_cells):
            for amp, ind_i, ind_j, ind_R in zip(amps, from_idx, to_idx, R_vecs, strict=True):
                hop_amp = amp.copy() if self._nspin == 2 else complex(amp)
                R_vec = ind_R.copy()
                jump_fin = int(R_vec[periodic_dir])

                hi = int(ind_i) + c * self.norb
                hj = int(ind_j) + (c + jump_fin) * self.norb

                if cut_model.dim_k != 0:
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
                    cut_model.set_hop(
                        hop_amp,
                        hi,
                        hj,
                        R_arg,
                        mode="add",
                        allow_conjugate_pair=True,
                    )

        # replicate parameterised hoppings
        param_hops = getattr(self, "_hopping_param_terms", {})
        if param_hops:
            for c in range(num_cells):
                base = c * self.norb
                for (ind_i, ind_j, R_tuple), provider in param_hops.items():
                    R_vec = np.array(R_tuple, dtype=int)
                    jump_fin = int(R_vec[periodic_dir])

                    hi = int(ind_i) + base
                    hj = int(ind_j) + (c + jump_fin) * self.norb

                    if cut_model.dim_k != 0:
                        R_copy = R_vec.copy()
                        R_copy[periodic_dir] = 0
                        R_arg = R_copy
                    else:
                        R_arg = None

                    if not glue_edges:
                        if hj < 0 or hj >= self.norb * num_cells:
                            continue
                    else:
                        hj = hj % (self.norb * num_cells)

                    cut_model.set_hop(
                        provider,
                        hi,
                        hj,
                        R_arg,
                        mode="set",
                        allow_conjugate_pair=True,
                    )


        return cut_model

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
    @deprecated("use `make_finite` or `cut_piece` with ``num_cells=1`` along the desired directions instead (since v2.0).", category=FutureWarning)
    def reduce_dim(self) -> "TBModel":
        r"""
        .. deprecated:: 2.0.0
            Use :meth:`make_finite` or :meth:`cut_piece` with ``num_cells=1`` along the desired 
            directions instead.
            If the intention is to keep periodicity along all directions while keeping some
            k-values fixed, this can be achieved by using the :meth:`hamiltonian` method 
            passing the desired k-values.
        """
        pass

    def change_nonperiodic_vector(
        self, 
        fin_dir: int, 
        new_latt_vec: np.ndarray = None, 
        to_home: bool = True, 
        ):
        """Change non-periodic lattice vector 
            
        Changes one of the non-periodic "lattice vectors". Non-periodic lattice vectors 
        are those that are *not* listed as periodic in the :attr:`periodic_dirs` parameter. 
        The orbital vectors are modified accordingly so that the actual (Cartesian) coordinates of 
        orbitals remain unchanged.

        .. versionremoved:: 2.0.0
            Parameter `to_home_supress_warning` has been removed.

        Parameters
        ----------
        fin_dir : int
            Index of non-periodic lattice vector to change.

            .. versionchanged:: 2.0.0
                Renamed from ``np_dir`` for clarity and consistency.

        new_latt_vec : array_like, optional
            The new non-periodic lattice vector. If None (default), the new
            non-periodic lattice vector is constructed to be orthogonal to all periodic 
            vectors and to have the same length as the original non-periodic vector.

        to_home : bool, optional
            If ``True`` (default), shift all orbitals to the home cell along
            periodic directions. Default behavior is to shift orbitals
            to the home cell.

            .. versionchanged:: 2.0.0
                This parameter was previously not working as intended and is now fixed.

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

        Constructs a :class:`pythtb.TBModel` representing a super-cell 
        of the current object. This function can be used together with :func:`cut_piece`
        in order to create slabs with arbitrary surfaces.

        By default all orbitals will be shifted to the home cell after
        unit cell has been created. That way all orbitals will have
        reduced coordinates between 0 and 1. If you wish to avoid this
        behavior, you need to set, *to_home* argument to *False*.

        .. versionremoved:: 2.0.0
            Parameter `to_home_supress_warning` has been removed.

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
        sc_tb.assume_position_operator_diagonal = self.assume_position_operator_diagonal

        for offset in range(num_sc):
            base = offset * self.norb
            for orb_idx, onsite in enumerate(self._site_energies):
                sc_tb.set_onsite(onsite, base + orb_idx)

        sc_index = {tuple(vec.tolist()): idx for idx, vec in enumerate(sc_vec)}
        sc_red_lat = geom["sc_red_lat"]
        red_transform = geom["red_transform"]
        eps = 1e-8

        amps, ind_is, ind_js, R_vecs = self._hoptable.components()

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
                    raise Exception("Did not find super cell vector!")

                hi = int(ind_i) + base
                hj = int(ind_j) + pair_idx * self.norb

                if self._nspin == 2:
                    amp_use = amp.copy()
                else:
                    amp_use = complex(amp)

                sc_tb.set_hop(
                    amp_use, hi, hj, sc_part, mode="add", allow_conjugate_pair=True
                )

        param_hops = getattr(self, "_hopping_param_terms", {})
        if param_hops:
            for offset, cur_sc_vec in enumerate(sc_vec):
                base = offset * self.norb
                for (ind_i, ind_j, R_tuple), provider in param_hops.items():
                    R_vec = np.array(R_tuple, dtype=float)
                    total_disp = cur_sc_vec + R_vec
                    red_disp = total_disp @ red_transform
                    sc_part = np.floor(red_disp + eps).astype(int)
                    orig_part = total_disp - sc_part @ sc_red_lat
                    pair_idx = sc_index.get(tuple(orig_part.tolist()))
                    if pair_idx is None:
                        raise RuntimeError("Supercell vector not found for parameterised hopping.")

                    hi = int(ind_i) + base
                    hj = int(ind_j) + pair_idx * self.norb

                    sc_tb.set_hop(
                        provider,
                        hi,
                        hj,
                        sc_part,
                        mode="set",
                        allow_conjugate_pair=True,
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
                self._hoptable.shift_orbital(i, disp_vec)

    @copydoc(Lattice.k_uniform_mesh)
    def k_uniform_mesh(self, mesh_size):
        return self._lattice.k_uniform_mesh(mesh_size)

    @copydoc(Lattice.k_path)
    def k_path(self, k_nodes, nk:int, report:bool=True):
        return self._lattice.k_path(k_nodes, nk, report)

    ############### Observables ####################
    
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
    
    def _H_to_per_gauge(self, H_flat, k_vals):
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

    def _hamiltonian_finite(self, hop_amps, i_idx, j_idx, site_energies, *, flatten_spin: bool):
        norb = self.norb

        if not self.spinful:
            hop_amps = hop_amps.astype(complex)
            ham = np.zeros((norb, norb), dtype=complex)
            if hop_amps.size:
                np.add.at(ham, (i_idx, j_idx), hop_amps)
                np.add.at(ham, (j_idx, i_idx), hop_amps.conj())
            np.fill_diagonal(ham, site_energies)
            return ham

        # spinful
        nspin = self.nspin
        dim_block = norb * nspin  # full matrix dimension after flattening spins
        hop_amps = np.asarray(hop_amps, dtype=complex)

        ham = np.zeros((dim_block, dim_block), dtype=complex)
        if hop_amps.size:
            # For every hopping we have a 2×2 spin block. Pre-compute the spin-pair indices
            # so we can add all blocks in a single vectorised pass.
            hop_blocks = hop_amps.reshape(hop_amps.shape[0], -1)   # (n_hops, nspin^2)
            spin_out = np.repeat(np.arange(nspin), nspin)          # s_out varies slowest
            spin_in = np.tile(np.arange(nspin), nspin)             # s_in varies fastest

            rows = (i_idx[:, None] * nspin + spin_out[None, :]).reshape(-1)
            cols = (j_idx[:, None] * nspin + spin_in[None, :]).reshape(-1)
            contrib = hop_blocks.reshape(-1)

            flat_idx = rows * dim_block + cols
            ham_flat = ham.reshape(-1)
            np.add.at(ham_flat, flat_idx, contrib)

            # Hermiticity: add the conjugate only on the opposite off-diagonal entries
            offdiag = rows != cols
            if np.any(offdiag):
                conj_idx = cols[offdiag] * dim_block + rows[offdiag]
                np.add.at(ham_flat, conj_idx, contrib[offdiag].conj())

        # On-site energies already come as 2×2 spin blocks.  Broadcast them onto the diagonal.
        ham_view = ham.reshape(norb, nspin, norb, nspin)
        diag_orbs = np.arange(norb)
        ham_view[diag_orbs, :, diag_orbs, :] += site_energies

        if flatten_spin:
            return ham
        return ham_view


    def _hamiltonian_periodic(
        self,
        k_vecs: np.ndarray,
        hop_amps,
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
        n_hops = hop_amps.shape[0]

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
            hop_amps = hop_amps.astype(complex)
            ham = np.zeros((n_kpts, norb, norb), dtype=complex)
            if n_hops:
                cache = self._get_flattened_indices()
                order = cache["order"]
                starts = cache["starts"]
                uniq = cache["uniq"]
                cols_transposed = cache["cols_transposed"]

                ham_flat = ham.reshape(n_kpts, -1)
                contrib = phases[:, order] * hop_amps[order]
                sums = np.add.reduceat(contrib, starts, axis=1)
                ham_flat[:, uniq] += sums
                ham_flat[:, cols_transposed] += sums.conj()

            diag = np.arange(norb)
            ham[:, diag, diag] += site_energies
            return ham

        # spinful
        nspin = self.nspin
        M = norb * nspin
        hop_blocks = np.asarray(hop_amps, dtype=complex).reshape(n_hops, -1)

        ham = np.zeros((n_kpts, M, M), dtype=complex)
        if n_hops:
            # flattened indices for every spin pair
            spin_out = np.repeat(np.arange(nspin), nspin) # [0, 0, 1, 1]
            spin_in = np.tile(np.arange(nspin), nspin)  # [0, 1, 0, 1]

            row_flat = (i_idx[:, None] * nspin + spin_out[None, :]).reshape(-1)
            col_flat = (j_idx[:, None] * nspin + spin_in[None, :]).reshape(-1)

            pair_flat = row_flat * M + col_flat
            order = np.argsort(pair_flat, kind="stable")
            uniq, starts = np.unique(pair_flat[order], return_index=True)
            cols_transposed = (uniq % M) * M + (uniq // M)

            ham_flat = ham.reshape(n_kpts, -1)
            contrib = (phases[:, :, None] * hop_blocks[None, :, :]).reshape(n_kpts, -1)
            contrib = contrib[:, order]

            sums = np.add.reduceat(contrib, starts, axis=1)
            ham_flat[:, uniq] += sums
            ham_flat[:, cols_transposed] += sums.conj()

        rows = np.arange(norb * nspin).reshape(norb, nspin)
        ham[:, rows[:, :, None], rows[:, None, :]] += site_energies[None, :, :, :]

        if not flatten_spin:
            ham = ham.reshape(n_kpts, norb, nspin, norb, nspin)
        return ham

    def hamiltonian(
            self, 
            k_pts=None, 
            flatten_spin_axis=False,
            **params
            ):
        r"""Generate the Bloch Hamiltonian of the tight-binding model.

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

        **params : 
            Keyword arguments mapping parameter names to value(s). Each value can be a scalar
            or a 1D array of values. If any values are array-like,
            the Hamiltonian is evaluated at all combinations of parameter values,
            and the final array is stacked with the k-axis leading, followed by each
            parameter axis in the order of given parameter names.

        Returns
        -------
        ham : numpy.ndarray
            Array of Bloch-Hamiltonian matrices defined on the specified k-points. The Hamiltonian is Hermitian by construction.
            The shape of the returned array depends on the presence of k-points and spin:

            - Spinless models: ``(..., norb, norb)``
            - Spinful models: ``(..., norb, nspin, norb, nspin)`` if ``flatten_spin_axis=False``,
              or ``(..., norb*nspin, norb*nspin)`` if ``flatten_spin_axis=True``.
            - ``dim_k > 0``: ``(n_kpts, ...)``
            - ``dim_k = 0``: no k-point axis.
            - Parameter sweeps: ``(n_kpts, n_param1, n_param2, ...)`` parameter axes are added after the k-point axis (if present).

        See Also
        --------
        velocity : Compute the derivatives of the Hamiltonian with respect to k and parameters.
        k_uniform_mesh : Generate a uniform k-point mesh for periodic systems.
        k_path : Generate a k-point path for band structure calculations. 

        Notes
        -----
        - In convention I, the Hamiltonian satisfies:

          .. math::
            H(k) \neq H(k + G), \quad \text{but instead} \quad H(k) = U H(k + G) U^{\dagger}

          where :math:`G` is a reciprocal lattice vector and :math:`U` is a unitary transformation
          relating the two.

        Examples
        --------
        Compute Hamiltonian at a single k-point for a finite model:

        >>> ham = tb_model.hamiltonian()

        Compute Hamiltonian at multiple k-points for a periodic model:

        >>> k_points = tb_model.k_uniform_mesh([10, 10])
        >>> ham_k = tb_model.hamiltonian(k_pts=k_points)

        Compute Hamiltonian while sweeping over a parameter:

        >>> model = TBModel(lattice, spinful=False)
        >>> model.set_hop(-1.0, 0, 1, [0, 0], param_name='t1')
        >>> k_points = model.k_uniform_mesh([5, 5])
        >>> ham_param = tb_model.hamiltonian(k_pts=k_points, t1= [0.0, 1.0, 2.0])     
        """

        # Check params includes all parameters
        if params is not None:
            self._check_missing_parameters(params)

        # Normalize k-points 
        if self.dim_k == 0:
            k_arr = None
        else:
            if k_pts is None:
                k_arr = self._normalize_kpoints(np.zeros((1, self.dim_k)))
            else:
                k_arr = self._normalize_kpoints(k_pts)

        # Partition params into scalars vs sweeps (1D arrays/lists)
        # scalars: dict of param_name -> scalar value
        # sweep_names: list of param names to sweep over
        # sweep_axes: list of arrays/lists of values to sweep over 
        scalars, sweep_names, sweep_axes = self._params_to_sweep(params)

        # No sweep axes, resolve scalar parameters only
        if not sweep_axes:
            hop_amps, i_idx, j_idx, R_vecs, site = self._evaluate_params(scalars)
            if self.dim_k == 0:
                H = self._hamiltonian_finite(hop_amps, i_idx, j_idx, site, flatten_spin=flatten_spin_axis)
            else:
                H = self._hamiltonian_periodic(
                    k_arr, hop_amps, i_idx, j_idx, R_vecs, site, flatten_spin=flatten_spin_axis
                )
            return H

        # param sweeps: cartesian product, then reshape with lambda at the end
        axis_lengths = [len(ax) for ax in sweep_axes]
        blocks, base_shape = [], None
        for multi in product(*[range(n) for n in axis_lengths]):
            assign = scalars.copy()
            for a, name in enumerate(sweep_names):
                assign[name] = sweep_axes[a][multi[a]]

            # ---- assemble H from immutable arrays ----
            hop_amps, i_idx, j_idx, R_vecs, site = self._evaluate_params(assign)
            if self.dim_k == 0:
                H = self._hamiltonian_finite(hop_amps, i_idx, j_idx, site, flatten_spin=flatten_spin_axis)
            else:
                H = self._hamiltonian_periodic(
                    k_arr, hop_amps, i_idx, j_idx, R_vecs, site, flatten_spin=flatten_spin_axis
                )

            if base_shape is None:
                base_shape = H.shape
            blocks.append(H[np.newaxis, ...])   

        stacked = np.concatenate(blocks, axis=0)    # (*Nλ, *base_shape)
        stacked = stacked.reshape(*axis_lengths, *base_shape)

        if axis_lengths and self.dim_k != 0:
            p = len(axis_lengths)
            b = len(base_shape)          # typically (Nk, nstate, nstate)
            perm = (p,) + tuple(range(p)) + tuple(range(p + 1, p + b))
            stacked = np.transpose(stacked, perm)

        self._H = stacked

        return stacked
    
    def _sol_ham(
        self, 
        ham, 
        return_eigvecs=False, 
        flatten_spin_axis=False, 
        tf_speedup=False, 
        use_32_bit=False
        ):
        """Solves Hamiltonian and returns eigenvectors, eigenvalues"""
        # NOTE: this function is separate so that it can be jit-compiled if needed

        if not np.allclose(ham, ham.swapaxes(-1, -2).conj()):
            raise ValueError("Hamiltonian matrix is not Hermitian.")
        
        if tf_speedup:
            import tensorflow as tf

            if use_32_bit:
                ham_tf = tf.convert_to_tensor(ham, dtype=tf.complex64)
            else:
                ham_tf = tf.convert_to_tensor(ham, dtype=tf.complex128)

            evals_tf, evecs_tf = tf.linalg.eigh(ham_tf) 

            if return_eigvecs:
                # return later
                eval, evec = evals_tf.numpy(), evecs_tf.numpy()
            else:
                return evals_tf.numpy()

        else:
            if use_32_bit:
                ham_use = ham.astype(np.complex64)
            else:
                ham_use = ham.astype(np.complex128)
            if return_eigvecs:
                # return later
                eval, evec = np.linalg.eigh(ham_use)
            else:
                return np.linalg.eigvalsh(ham_use)
            
        if return_eigvecs:
            if self.nspin == 1:
                shape_evecs = (*ham.shape[:-2],) + (self.norb, self.norb)
            elif self.nspin == 2:
                shape_evecs = (*ham.shape[:-2],) + (
                    self.nstate,
                    self.norb,
                    self.nspin,
                )

            # transpose matrix eig since otherwise it is confusing
            # now eig[i,:] is eigenvector for eval[i]-th eigenvalue
            evec = evec.swapaxes(-1, -2)
            if not flatten_spin_axis and self.nspin == 2:
                evec = evec.reshape(*shape_evecs)
            return eval, evec

    def solve_ham(
            self, 
            k_pts = None, 
            return_eigvecs: bool = False, 
            flatten_spin_axis: bool = True,
            tf_speedup: bool = False,
            **params
            ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        r"""Diagonalize the Hamiltonian 
        
        Solve for eigenvalues and optionally eigenvectors of the tight-binding model
        at a list of one-dimensional k-vectors.

        .. versionadded:: 2.0.0
            Merged :func:`solve_all` and :func:`solve_one` into :func:`solve_ham`.
            This function will equivalently handle both a single k-point and
            multiple k-points. 

        Parameters
        ----------
        k_pts : array_like, optional
            One-dimensional list or array of k-vectors, each given in reduced coordinates.
            Shape should be ``(Nk, dim_k)``, where ``dim_k`` is the number of periodic directions.
            Should not be specified for systems with zero-dimensional reciprocal space.

            .. versionchanged:: 2.0.0
                Renamed from ``k_list``.

        return_eigvecs : bool, optional
            If True, both eigenvalues and eigenvectors are returned.
            If False (default), only eigenvalues are returned.

            .. versionchanged:: 2.0.0
                Renamed from ``eig_vectors``.

        flatten_spin_axis : bool, optional
            If True (default), the spin axes are flattened into the orbital axes.
            If False, the spin axes are kept separate. This affects the
            shape of the returned eigenvectors for spinful models.

            .. versionadded:: 2.0.0

        tf_speedup : bool, optional
            If True, use TensorFlow to accelerate the diagonalization.
            This requires TensorFlow to be installed. Default is False.

            .. versionadded:: 2.0.0

        **params : 
            Keyword arguments mapping parameter names to value(s). Each value can be a scalar
            or a 1D array of values. If any values are array-like,
            the Hamiltonian is evaluated at all combinations of parameter values,
            and the final array is stacked with the k-axis leading, followed by each
            parameter axis in the order of given parameter names.

            .. versionadded:: 2.0.0

        Returns
        -------
        eval : np.ndarray 
            Array of eigenvalues. Shape is:

            - ``(Nk, nstates)`` for periodic systems
            - ``(nstates,)`` for zero-dimensional (molecular) systems
            - ``(Nk, n_param1, n_param2, ..., nstates)`` if parameter sweeps are performed,
              with parameter axes added after the k-point axis.

        evec : np.ndarray, optional
            Array of eigenvectors (if ``return_eigvecs=True``). The ordering of bands matches that in ``eval``.

            For spinless models the shape is:

            - ``(Nk, nstates, norb)``: periodic systems
            - ``(nstates, norb)``: zero-dimensional systems
            - ``(nstates, norb)``: If only one k-point is provided, the redundant k-axis is removed.

            For spinful models the shape is (``nstates = norb * 2``):

            - ``(..., nstates, norb, 2)``: If ``flatten_spin_axis=False``, an additional spin axis of size 2 is appended at the end.
            - ``(..., nstates, nstates)``: If ``flatten_spin_axis=True``, the spin axes are flattened into the orbital axes.

            If parameter sweeps are performed, parameter axes are added after the k-point axis.

            - ``(Nk, n_param1, n_param2, ..., nstates, norb[, 2])`` or
              ``(Nk, n_param1, n_param2, ..., nstates, nstates)`` depending on ``flatten_spin_axis``.
            
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
        Ham = self.hamiltonian(k_pts, flatten_spin_axis=True, **params)

        logger.debug("Diagonalizing Hamiltonian...")
        if return_eigvecs:
            eigvals, eigvecs = self._sol_ham(
                Ham, return_eigvecs=return_eigvecs, flatten_spin_axis=flatten_spin_axis, tf_speedup=tf_speedup
            )
            if self.dim_k != 0:
                # if only one k_point, remove that redundant axis (reproduces solve_one)
                if eigvals.shape[0] == 1:
                    eigvals = eigvals[0]
                    eigvecs = eigvecs[0]

            return eigvals, eigvecs
        else:
            eigvals = self._sol_ham(Ham, return_eigvecs=return_eigvecs)

            if self.dim_k != 0:
                # if only one k_point, remove that redundant axis (reproduces solve_one)
                if eigvals.shape[0] == 1:
                    eigvals = eigvals[0]
            return eigvals

    @deprecated("use .solve_ham() instead (since v2.0).", category=FutureWarning)
    def solve_one(self, k_list=None, eig_vectors=False):
        """
        .. deprecated:: 2.0.0
            Use :meth:`solve_ham` instead.
        """
        return self.solve_ham(
            k_list=k_list, return_eigvecs=eig_vectors, flatten_spin_axis=False
        )

    @deprecated("use .solve_ham() instead (since v2.0).", category=FutureWarning)
    def solve_all(self, k_list=None, eig_vectors=False):
        """
        .. deprecated:: 2.0.0
            Use :meth:`solve_ham` instead.
        """
        return self.solve_ham(
            k_list=k_list, return_eigvecs=eig_vectors, flatten_spin_axis=False
        )
    
    def _velocity(
            self, 
            k_arr: np.ndarray, 
            hop_amps: np.ndarray,
            i_indices: np.ndarray,
            j_indices: np.ndarray,
            R_vecs: np.ndarray,
            site_energies: np.ndarray = None,
            *,
            cartesian: bool = False,
            flatten_spin_axis: bool = False,
            return_ham: bool = False,
            ) -> np.ndarray:
        
        dim_k = self.dim_k

        norb = self.norb
        per = np.asarray(self.per)
        orb_red = np.asarray(self.orb_vecs)

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
            amps_use = np.asarray(hop_amps, dtype=complex)
            vel = np.zeros((dim_k, k_arr.shape[0], norb, norb), dtype=complex)
            if n_hops:
                cache = self._get_flattened_indices()
                order = cache["order"]
                starts = cache["starts"]
                uniq = cache["uniq"]
                cols_transposed = cache["cols_transposed"]

                vel_flat = vel.reshape(dim_k, k_arr.shape[0], -1)

                if return_ham:
                    ham = np.zeros((k_arr.shape[0], norb, norb), dtype=complex)
                    ham_flat = ham.reshape(k_arr.shape[0], -1)
                    contrib_ham = phases[:, order] * amps_use[order]
                    sums_ham = np.add.reduceat(contrib_ham, starts, axis=1)
                    ham_flat[:, uniq] += sums_ham
                    ham_flat[:, cols_transposed] += sums_ham.conj()

                contrib_sorted = deriv_phase[:, :, order] * amps_use[order]
                sums = np.add.reduceat(contrib_sorted, starts, axis=2)
                vel_flat[..., uniq] += sums
                vel_flat[..., cols_transposed] += sums.conj()

            if return_ham:
                diag = np.arange(norb)
                ham[:, diag, diag] += site_energies
                return vel, ham
            return vel
    
        n_kpts = k_arr.shape[0]
        nspin = self.nspin
        M = norb * nspin

        # Flatten spin blocks (n_hops, nspin, nspin) -> (n_hops, nspin^2)
        # For spin pair index s in [0, ..., nspin^2-1]:
        #   s_out = s // nspin
        #   s_in  = s % nspin
        hop_blocks = np.asarray(hop_amps, dtype=complex).reshape(n_hops, -1)

        vel = np.zeros((dim_k, k_arr.shape[0], M, M), dtype=complex)
        if n_hops:
            # spin_out, spin_in shape: (nspin^2,)
            # Creates all (s_out, s_in) pairs in a fixed order
            spin_out = np.repeat(np.arange(nspin), nspin) # [0,0,1,1] for nspin=2
            spin_in = np.tile(np.arange(nspin), nspin)    # [0,1,0,1] for nspin=2

            # For hop h and spin pair (s_out, s_in), the big M×M index is
            #   row = s_out*norb + i_indices[h]
            #   col = s_in *norb + j_indices[h]
            # Build them as (n_hops * nspin^2,) arrays, then flatten to 1D of 
            # length n_hops*nspin^2
            row_flat = (i_indices[:, None] * nspin + spin_out[None, :]).reshape(-1)
            col_flat = (j_indices[:, None] * nspin + spin_in[None, :]).reshape(-1)

            # Convert (row, col) to a single flat index in [0 .. M*M-1].
            # pair_flat uniquely identifies each matrix element touched by a hop×spin pair.
            pair_flat = row_flat * M + col_flat
            # Sorting by pair_flat lets us use reduceat to sum contiguous groups efficiently.
            order = np.argsort(pair_flat, kind="stable")

            # uniq: the unique flat matrix positions; starts: the start indices of each group.
            uniq, starts = np.unique(pair_flat[order], return_index=True)
            # Hermitian partner flat positions (swap row/col): (r,c) -> (c,r)
            cols_transposed = (uniq % M) * M + (uniq // M)

            if return_ham:
                ham = np.zeros((k_arr.shape[0], M, M), dtype=complex)
                ham_flat = ham.reshape(k_arr.shape[0], -1)

                # Broadcast multiply to get contribution per (hop, spin): shape (Nk, n_hops, S)
                contrib_ham = (phases[:, :, None] * hop_blocks[None, :, :])
                # Flatten the last two axes to match pair_flat ordering, then reorder by 'order'
                contrib_ham = contrib_ham.reshape(n_kpts, -1)  # (Nk, n_hops*nspin^2)
                contrib_ham = contrib_ham[:, order]  # align with pair_sorted

                # Sum within each group of identical matrix elements
                sums_ham = np.add.reduceat(contrib_ham, starts, axis=1)  # (Nk, len(uniq))

                # Scatter once per unique element (and its Hermitian partner)
                ham_flat[:, uniq] += sums_ham
                ham_flat[:, cols_transposed] += sums_ham.conj()

            # Broadcast multiply to get contribution per (hop, spin): shape (dim_k, Nk, n_hops, S)
            terms = (deriv_phase[:, :, :, None] * hop_blocks[None, None, :, :])
            # Flatten the last two axes to match pair_flat ordering, then reorder by 'order'
            terms = terms.reshape(dim_k, n_kpts, -1)  # (dim_k, Nk, n_hops*nspin^2)
            terms = terms[..., order]  # align with pair_sorted

            # Sum within each group of identical matrix elements
            sums = np.add.reduceat(terms, starts, axis=2)  # (dim_k, Nk, len(uniq))

            # Scatter once per unique element (and its Hermitian partner)
            vel_flat = vel.reshape(dim_k, n_kpts, -1) # flatten (M,M) -> M*M as last axis
            vel_flat[:, :, uniq] += sums
            vel_flat[:, :, cols_transposed] += sums.conj()

        if return_ham:
            rows = np.arange(norb * nspin).reshape(norb, nspin)
            ham[:, rows[:, :, None], rows[:, None, :]] += site_energies[None, :, :, :]

            if not flatten_spin_axis:
                ham = ham.reshape(n_kpts, norb, nspin, norb, nspin)
                vel = vel.reshape(dim_k, n_kpts, norb, nspin, norb, nspin)
            
            return vel, ham

        if not flatten_spin_axis:
            vel = vel.reshape(dim_k, n_kpts, norb, nspin, norb, nspin)
        return vel

    def velocity(
            self, 
            k_pts: np.ndarray,
            cartesian: bool = False,
            flatten_spin_axis: bool = False,
            *,
            param_periods: dict[str, float] | None = None,
            diff_scheme: str = "central",
            diff_order: int = 2,
            _return_ham: bool = False,
            **params
            ) -> np.ndarray:
        r"""Generate the velocity operator in the orbital basis.

        The velocity operator is related to the derivative of the Hamiltonian
        with respect to each reciprocal lattice direction, i.e., 

        .. math::
            v_{\mu}(k) = \hbar \frac{\partial H(k)}{\partial k_{\mu}}

        When passing parameter sweeps via ``**params``, the generalized velocity
        operator is computed by appending finite-difference derivatives of the
        Hamiltonian with respect to the swept parameters.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        k_pts : (Nk, dim_k) numpy.ndarray
            Reduced k-points where the velocity operator is evaluated.
        cartesian : bool, optional
            If True, use Cartesian coordinates for the velocity operator, 
            otherwise derivatives are taken with respect to reduced coordinates.
        flatten_spin_axis : bool, optional
            If True, the spin indices are flattened into the orbital indices.
            This results in a velocity operator of shape ``(..., norb*nspin, norb*nspin)``.
            If False (default), the velocity operator has shape ``(..., norb, nspin, norb, nspin)``.
        param_periods : dict[str, float], optional
            Optional map ``{param_name: period}`` for swept parameters. When supplied,
            assumes the parameter is cyclic and trims any duplicated endpoint
            before building finite-difference stencils. Parameters not listed here
            are treated as non-periodic unless their sample list starts and ends
            at the same value.
        diff_scheme : {'central', 'forward'}, optional
            Finite-difference stencil used for parameter derivatives 
            (defaults to ``'central'``).
        diff_order : int, optional
            Order of accuracy for the finite differences (defaults to 2).
        **params : 
            Parameter assignments. Scalars are applied directly; any 1D array/list 
            is treated as a sweep and *automatically* adds a finite-difference derivative
            :math:`\partial_{\lambda} H` for that parameter. The velocity operator is
            evaluated at all combinations of parameter values.

        Returns
        -------
        vel : numpy.ndarray
            Velocity operator in the orbital basis. First axis indexes the cartesian direction 
            if ``cartesian=True``. Otherwise, it indexes the reduced direction. If
            ``include_lambda=True``, the lambda (parameter) derivatives are appended
            after the k-directions along the first axis.
            
            Shape is: 

            - ``(n_dir, Nk, *param_shape, norb, norb)`` for spinless models,
            - ``(n_dir, Nk, *param_shape, norb, 2, norb, 2)`` for spinful models.
            - ``(..., norb*nspin, norb*nspin)`` if ``flatten_spin_axis=True``.

            If ``include_lambda=True``, ``n_dir = dim_k + n_params``, where ``n_params`` is 
            the number of parameters being swept over. Otherwise, ``n_dir = dim_k``.
        
        Notes
        -----
        - We use units where :math:`\hbar = 1`, and thus the velocity operator is simply the derivatives
          of the Hamiltonian with respect to k or parameters. 
        - The velocity operator is computed in tight-binding convention I, which includes phase factors
          associated with orbital positions in the hopping terms.
        - Passing a list/array for a parameter means you want derivatives with respect to that
          parameter. If the intent is simply to evaluate at a specific value, resolve the
          symbol first via :meth:`set_parameters`, or simply pass a scalar value with ``**params``.
        - When passing a list/array for a parameter, the finite difference derivatives are computed 
          explicitly as

          .. math::
            \frac{\partial H}{\partial \lambda} \approx 
            \sum_{m} c_m H(\lambda + m \Delta \lambda)

          where the coefficients :math:`c_m` depend on the finite difference scheme and order.

        Examples
        --------
        Compute the velocity operator at the Gamma point:

        >>> vel = tb.velocity(np.array([[0.0, 0.0]]))

        Compute the velocity operator at several k-points with a parameter sweep. This
        will compute the velocity operator at all values of ``mA = 0.0, 1.0, 2.0``. Without
        ``include_lambda``, only the k-derivatives are included, and the first axis will 
        have length equal to ``dim_k``.

        >>> vel = tb.velocity(np.array([[0.0, 0.0], [0.5, 0.5]]), mA=np.linspace(0, np.pi, 10, endpoint=False))

        Compute the velocity operator including parameter derivatives. This will compute the velocity operator
        at all values of ``mA = 0.0, 1.0, 2.0``, and the first axis will have length equal to ``dim_k + 1``,
        with the last slice corresponding to the finite-difference derivative with respect to ``mA``.

        >>> vel = tb.velocity(np.array([[0.0, 0.0]]), mA=np.linspace(0, np.pi, 10, endpoint=False), include_lambda=True)

        In this case, we must be sure that the Hamiltoian is periodic in ``mA``, i.e., that the
        Hamiltonian at ``mA=0`` is the same as that at ``mA=np.pi``, and that the endpoints are not included
        in the parameter list for accurate finite difference derivatives.
        """        
        # Check params
        if params is not None:
            params = dict(params)
            self._check_missing_parameters(params)
        else:
            params = {}

        # Normalize k-points to correct shape
        if self.dim_k == 0:
            raise NotImplementedError("Velocity operator is not defined for systems with dim_k=0.")
        else:
            k_arr = self._normalize_kpoints(k_pts)

        # Partition params into scalars vs sweeps (1D arrays/lists)
        # scalars: dict of param_name -> scalar value
        # sweep_names: list of param names to sweep over
        # sweep_axes: list of arrays/lists of values to sweep over 
        scalars, sweep_names, sweep_axes = self._params_to_sweep(params)

        param_periods = dict(param_periods or {})

        raw_axes: list[list[float]] = []
        param_meta: dict[str, tuple[float, bool, bool, list[float]]] = {}
        for idx, name in enumerate(sweep_names):
            axis_array = np.asarray(sweep_axes[idx], dtype=float)
            raw_axes.append(axis_array.tolist())  # preserve the user’s grid
            if axis_array.ndim != 1 or axis_array.size < 2:
                continue

            normalized, step, periodic, trimmed = self._normalize_parameter_axis(
                axis_array,
                name=name,
                period=param_periods.get(name),
            )
            param_meta[name] = (step, periodic, trimmed, normalized.tolist())

        # param_meta: dict[str, tuple[float, bool]] = {}
        # for idx, name in enumerate(sweep_names):
        #     try:
        #         axis_array = np.asarray(sweep_axes[idx], dtype=float)
        #     except (TypeError, ValueError):
        #         continue
        #     if axis_array.ndim != 1 or axis_array.size < 2:
        #         continue
    
        #     normalized, step, periodic, trimmed = self._normalize_parameter_axis(
        #         axis_array,
        #         name=name,
        #         period=param_periods.get(name),
        #     )
        #     sweep_axes[idx] = normalized.tolist()
        #     param_meta[name] = (step, periodic, trimmed)

        # Determine if we need to compute Hamiltonian for lambda derivatives
        needs_ham = _return_ham or bool(param_meta)

        # No sweeps: just evaluate in place
        if not sweep_axes:
            hop_amps, i_idx, j_idx, R_vecs, site_energies = self._evaluate_params(scalars)
            v = self._velocity(
                k_arr,
                hop_amps,
                i_idx,
                j_idx,
                R_vecs,
                site_energies=site_energies,
                cartesian=cartesian,
                flatten_spin_axis=flatten_spin_axis,
                return_ham=needs_ham
            )
            if needs_ham:
                vel, ham = v
                return (vel, ham) if _return_ham else vel
            return v

        # Parameter sweeps: cartesian product, 
        # then reshape to (*param_shape, dim_k, Nk, norb, norb) at end
        axis_lengths = [len(ax) for ax in sweep_axes]
        vel_blocks, base_shape = [], None
        ham_blocks, ham_shape = [], None

        for multi in product(*[range(n) for n in axis_lengths]):
            assign = scalars.copy() # copy str -> scalar mappings
            for a, name in enumerate(sweep_names):
                # Evaluate velocity on user grid, not normalized grid
                assign[name] = raw_axes[a][multi[a]]

            # Retrive hoppings, orbital indices, R-vectors
            # for this parameter combination
            hop_amps, i_idx, j_idx, R_vecs, site_energies = self._evaluate_params(assign)

            # Build velocity operator
            v = self._velocity(
                k_arr, 
                hop_amps, 
                i_idx, 
                j_idx, 
                R_vecs, 
                site_energies=site_energies,
                cartesian=cartesian,
                flatten_spin_axis=flatten_spin_axis,
                return_ham=needs_ham
            )

            if needs_ham:
                vel, ham = v
                if ham_shape is None:
                    ham_shape = ham.shape
                ham_blocks.append(ham[np.newaxis, ...])
            else:
                vel = v

            # Record base shape first time e.g. (dim_k, Nk, norb, norb)
            if base_shape is None:
                base_shape = vel.shape

            # Append with new leading axis for stacking later
            vel_blocks.append(vel[np.newaxis, ...])

        # Stack all velocity blocks along new leading axis: (N_param, *base_shape)
        stacked = np.concatenate(vel_blocks, axis=0).reshape(*axis_lengths, *base_shape)  
        # Move param axes to be after k-axis
        if axis_lengths:
            p = len(axis_lengths)
            b = len(base_shape)   # e.g. 4 when base is (dim_k, Nk, norb, norb)
            perm = (
                p,          # dim_k
                p + 1,      # Nk
                *range(p),  # all param axes in original order
                *range(p + 2, p + b)  # remaining matrix axes (e.g. norb, norb)
            )
            stacked = np.transpose(stacked, perm)

        # Final velocity array
        vel_k = stacked

        ham = None
        if needs_ham:
            # Stack all hamiltonian blocks along new leading axis: (N_param, *ham_shape)
            ham_stacked = np.concatenate(ham_blocks, axis=0).reshape(*axis_lengths, *ham_shape) 
            # Move param axes to be after k-axis
            if axis_lengths:
                p = len(axis_lengths)
                b = len(ham_shape)   # e.g. 3 when base is (Nk, norb, norb)
                perm = (p,) + tuple(range(p)) + tuple(range(p + 1, p + b))
                ham_stacked = np.transpose(ham_stacked, perm)
            
            # Final hamiltonian array
            ham = ham_stacked

        if not param_meta:
            return (vel_k, ham) if _return_ham else vel_k

        # The velocity output currently has shape (dim_k, Nk, l1, ..., norb, norb).
        # Append parameter-derivatives after the existing k components.
        vel_components = [vel_k]
        for idx, name in enumerate(sweep_names):
            meta = param_meta.get(name)
            if meta is None:
                continue # non-numeric sweep; skip

            step, periodic, trimmed, _ = meta
            axis = 1 + idx  # axis 0 is Nk, axis 1 begins the param sweeps

            if trimmed:
                logger.debug(
                    "velocity: Trimming endpoint for periodic parameter"
                    f"'{name}' before differentiating Hamiltonian.")
                # drop the repeated endpoint before taking finite differences
                slicer = [slice(None)] * ham.ndim
                slicer[axis] = slice(0, -1)
                ham_fd = ham[tuple(slicer)]
            else:
                ham_fd = ham

            dH = finite_difference(
                ham_fd, 
                axis=axis, 
                delta=step, 
                order=diff_order, 
                mode=diff_scheme,
                periodic=periodic
            )
            
            if trimmed and periodic:
                # re-append the first slice so the derivative array matches the user grid
                first_slice = np.take(dH, indices=0, axis=axis)
                first_slice = np.expand_dims(first_slice, axis=axis)
                dH = np.concatenate([dH, first_slice], axis=axis)

            vel_components.append(dH[np.newaxis, ...])  # prepend new derivative axis

        vel_full = np.concatenate(vel_components, axis=0)
        return (vel_full, ham) if _return_ham else vel_full

    def quantum_geometric_tensor(
        self,
        k_pts,
        occ_idxs = None,
        plane = None,
        *,
        cartesian: bool = False,
        non_abelian: bool = False,
        param_periods: dict[str, float] | None = None,
        diff_scheme: str = "central",
        diff_order: int = 2,
        use_tensorflow: bool = False,
        **params
    ):
        r"""Quantum geometric tensor at a list of k-points via Kubo formula.

        The quantum geometric tensor is computed from the derivatives of the 
        Bloch Hamiltonian :math:`\partial_\mu H_k`, where :math:`\mu` is the 
        direction in k-space, and is given by (when ``non_abelian=True``):

        .. math::

            Q_{\mu \nu;\ mn}(k) = \sum_{l \notin \text{occ}}
            \frac{
                \langle u_{mk} | \partial_{\mu} H_k | u_{lk} \rangle
                \langle u_{lk} | \partial_{\nu} H_k | u_{nk} \rangle
            }{
                (E_{nk} - E_{lk})(E_{mk} - E_{lk})
            }

        The Abelian quantum geometric tensor (when ``non_abelian=False``) 
        is obtained by taking the trace
        over occupied bands:

        .. math::

            Q_{\mu \nu}(k) = \sum_{m \in \text{occ}} Q_{\mu \nu;\ mm}(k)

        By specifying the ``plane`` parameter, we choose a particular :math:`(\mu, \nu)` pair
        of the quantum geometric tensor to return.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        k_pts : (Nk, dim_k) array-like
            Array of k-points with shape (Nk, dim_k), where Nk is the number of points
            and dim_k is the dimensionality of the k-space.
        occ_idxs : 1D array, optional
            Indices of the occupied bands. Defaults to the first half of the states.
        plane : tuple of int, optional
            Tuple of two integers specifying the plane in k-space for which to compute 
            the curvature. If None (default), 
            computes all components of the Berry curvature tensor. This 
            will affect the shape of the returned array.
        cartesian : bool, optional
            If True, computes the velocity operator in Cartesian coordinates.
            Default is False (reduced coordinates).
        non_abelian : bool, optional
            If True, returns the full tensor (non-abelian case).
            If False, returns the band-trace of the tensor (abelian case).
            Default is False.
        param_periods : dict[str, float], optional
            Optional map ``{param_name: period}`` for swept parameters. When supplied,
            assumes the parameter is cyclic and trims any duplicated endpoint
            before building finite-difference stencils. Parameters not listed here
            are treated as non-periodic unless their sample list starts and ends
            at the same value.
        diff_scheme : str, optional
            Finite difference scheme to use for lambda derivatives.
            Options are "central" (default) or "forward".
            This parameter is only relevant if ``include_lambda=True``.
        diff_order : int, optional
            Order of accuracy for finite difference lambda derivatives.
            Must be an even integer for "central" scheme (default is 2),
            and a positive integer for "forward" scheme.
            This parameter is only relevant if ``include_lambda=True``.
        use_tensorflow: bool, optional
            If True, will use tensorflow to speed up linear algebra routines.
        **params : 
            Keyword arguments mapping parameter names to value(s). Each value can be a scalar
            or a 1D array of values. If any values are array-like,
            the QGT is evaluated at all combinations of parameter values,
            and the final array is stacked with the k-axis leading, followed by each
            parameter axis in the order of given parameter names.

        Returns
        -------
        Q : array
            Quantum geometric tensor at the specified k-points.
            If ``plane`` is None, shape is ``(dim_k, dim_k, Nk, n_orb, n_orb)``.
            If ``plane`` is a tuple, shape is ``(Nk, n_orb, n_orb)`` and the returned 
            tensor is restricted to the specified directions. If ``non_abelian=False``,
            returns the band-trace of the quantum geometric tensor and the last 
            two axes are not present. If parameter sweeps are performed via ``params``,
            parameter axes are added after the k-point axis.

        See Also
        --------
        velocity : Computes the velocity operator used in the Kubo formula.
        berry_curvature : Computes the Berry curvature from the quantum geometric tensor.
        quantum_metric : Computes the quantum metric from the quantum geometric tensor.

        Notes
        -----
        - The quantum geometric tensor captures both the Berry curvature (imaginary part)
          and the quantum metric (real part) of the occupied bands.
        - The velocity operator is computed in tight-binding convention I, which includes phase factors
          associated with orbital positions in the hopping terms.
        - The finite difference derivatives with respect to parameters are computed explicitly as
          centered differences using a uniform grid of parameter values.
    
        """
        if self.dim_k < 2:
            raise NotImplementedError(
                """
                Quantum geometric tensor must have dim_k >= 2.
                """
            )

        v_k, ham = self.velocity(
            k_pts, 
            cartesian=cartesian, 
            flatten_spin_axis=True,
            param_periods=param_periods,
            diff_scheme=diff_scheme,
            diff_order=diff_order,
            _return_ham=True,  
            **params
            )  # (dim_k + dim_lam , Nk, *lam_shape, nstate, nstate)
        
        
        eigvals, eigvecs = self._sol_ham(
            ham, return_eigvecs=True, flatten_spin_axis=True, tf_speedup=use_tensorflow
        )

        if self.dim_k != 0:
            # if only one k_point, remove that redundant axis
            if eigvals.shape[0] == 1:
                eigvals = eigvals[0]
                eigvecs = eigvecs[0]
                
        # Identify occupied bands
        n_eigs = eigvecs.shape[-2]
        if occ_idxs is None:
            occ_idxs = np.arange(n_eigs // 2)
        else:
            occ_idxs = np.array(occ_idxs)
        # Identify conduction bands as remainder of band indices (assumes gapped)
        cond_idxs = np.setdiff1d(np.arange(n_eigs), occ_idxs)

        if use_tensorflow:
            # tensorflow optimization
            import tensorflow as tf
            from tensorflow import constant as const

            v_k_tf = const(v_k, dtype=tf.complex64)
            evals_tf, evecs_tf = const(eigvals, dtype=tf.complex64), const(eigvecs, dtype=tf.complex64)

            evecs_T_tf = tf.transpose(evecs_tf, perm=[0, 1, 3, 2])  # (n_kpts, n_beta, n_state, n_state)
            evecs_conj_tf = tf.math.conj(evecs_tf)

            # All pairwise energy differences
            delta_E_tf = evals_tf[..., None, :] - evals_tf[..., :, None]

            # Extract occupied <-> conduction band energy differences
            delta_E_occ_cond_tf = tf.gather(tf.gather(delta_E_tf, occ_idxs, axis=-2), cond_idxs, axis=-1)
            delta_E_cond_occ_tf = tf.gather(tf.gather(delta_E_tf, cond_idxs, axis=-2), occ_idxs, axis=-1)

            # Degeneracy guard: abort if any denominator is (near) zero
            tol = tf.constant(1e-12, dtype=delta_E_tf.dtype.real_dtype)
            if tf.reduce_any(tf.math.abs(delta_E_occ_cond_tf) < tol).numpy():
                raise ZeroDivisionError("Degenerate occupied/conduction bands encountered.")
            if tf.reduce_any(tf.math.abs(delta_E_cond_occ_tf) < tol).numpy():
                raise ZeroDivisionError("Degenerate occupied/conduction bands encountered.")

            inv_delta_E_occ_cond_tf = 1 / delta_E_occ_cond_tf
            inv_delta_E_cond_occ_tf = 1 / delta_E_cond_occ_tf

            # Rotate velocity operators to eigenbasis
            v_k_rot_tf = tf.matmul(
                evecs_conj_tf[None, :, :, :, :],  # (1, n_kpts, n_beta, n_state, n_state)
                tf.matmul(
                    v_k_tf,                       # (dim_k, n_kpts, n_beta, n_state, n_state)
                    evecs_T_tf[None, :, :, :, :]  # (1, n_kpts, n_beta, n_state, n_state)
                )
            )  # (dim_k, n_kpts, n_beta, n_state, n_state)

            v_occ_cond_tf = tf.gather(tf.gather(v_k_rot_tf, occ_idxs, axis=-2), cond_idxs, axis=-1)
            v_cond_occ_tf = tf.gather(tf.gather(v_k_rot_tf, cond_idxs, axis=-2), occ_idxs, axis=-1)
            v_occ_cond_tf = v_occ_cond_tf * inv_delta_E_occ_cond_tf
            v_cond_occ_tf = v_cond_occ_tf * -inv_delta_E_cond_occ_tf

            # Compute Berry curvature
            Q = tf.matmul(v_occ_cond_tf[:, None], v_cond_occ_tf[None, :])

            # Convert final result to NumPy
            Q = Q.numpy()
        else:
            # All pairs of energy differences
            delta_E = eigvals[..., np.newaxis, :] - eigvals[..., :, np.newaxis]  # shape (Nk, nstate, nstate)
            # Energy differences between occupied and conduction bands
            delta_occ_cond = delta_E[:, occ_idxs][:, :, cond_idxs]
            delta_cond_occ = delta_E[:, cond_idxs][:, :, occ_idxs]  # shape (Nk, n_occ, n_con)
            if np.any(np.isclose(delta_occ_cond, 0.0)):
                raise ZeroDivisionError("Degenerate occupied/conduction bands encountered.")
            inv_delta_occ_cond = np.divide(1.0, delta_occ_cond)
            inv_delta_cond_occ = np.divide(1.0, delta_cond_occ)

            # newaxis for Cartesian direction
            evecs_conj = eigvecs.conj()[np.newaxis, ...]
            # transpose for matmul
            evecs_T = eigvecs.swapaxes(-2, -1)[np.newaxis, ...]
            vk_evecT = np.matmul(v_k, evecs_T)  # intermediate array
            # Project vk into energy eigenbasis
            v_k_rot = np.matmul(evecs_conj, vk_evecT)  # (dim_k, n_kpts, n_states, n_states)

            # Extract relevant submatrices
            v_occ_cond = v_k_rot[..., occ_idxs, :][..., :, cond_idxs]  # shape (dim_k, Nk, n_occ, n_con)
            v_cond_occ = v_k_rot[..., cond_idxs, :][..., :, occ_idxs]  # shape (dim_k, Nk, n_con, n_occ)

            # premultiply by energy denominators
            v_occ_cond *= inv_delta_occ_cond
            v_cond_occ *= -inv_delta_cond_occ

            Q = np.matmul(v_occ_cond[:, None], v_cond_occ[None, :])

        if not non_abelian:
            Q = np.trace(Q, axis1=-1, axis2=-2)
        if plane is None:
            return Q
        else:
            if not (isinstance(plane, tuple) and len(plane) == 2):
                raise ValueError("plane must be a tuple of length 2.")
            return Q[plane]

    def berry_curvature(
        self,
        k_pts,
        occ_idxs = None,
        plane = None,
        *,
        cartesian: bool = False,
        non_abelian: bool = False,
        param_periods: dict[str, float] | None = None,
        diff_scheme: str = "central",
        diff_order: int = 2,
        use_tensorflow: bool = False,
        **params
    ):
        r"""Compute the Berry curvature in energy eigenbasis via Kubo formula.

        The Berry curvature is computed as the anti-Hermitian part of the quantum
        geometric tensor :math:`Q_{\mu \nu}(k)` from :meth:`quantum_geometric_tensor`, 
        i.e., in the non-Abelian case (``non_abelian=True``):

        .. math::

            \Omega_{\mu \nu;\ mn}(k) =  i \left( Q_{\mu \nu;\ mn}(k) - Q_{\mu \nu;\ nm}^*(k) \right)

        In the Abelian case (``non_abelian=False``), the Berry curvature is given by the
        band-trace of the above quantity. This reduces to the well-known expression for the
        Berry curvature in terms of the quantum geometric tensor.
        
        .. math:: 

           \Omega_{\mu \nu}(k) = -2 \mathrm{Im} \, Q_{\mu \nu}(k),

        By specifying the ``plane`` parameter, we choose a particular :math:`(\mu, \nu)` pair
        of the Berry curvature tensor to return.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        k_pts : (Nk, dim_k) array-like
            Array of k-points with shape (Nk, dim_k), where Nk is the number of points
            and dim_k is the dimensionality of the k-space.
        occ_idxs : 1D array, optional
            Indices of the occupied bands. Defaults to the first half of the states.
        plane : tuple of int, optional
            Tuple of two integers specifying the plane in k-space for which to compute 
            the curvature. If None (default), 
            computes all components of the Berry curvature tensor. This 
            will affect the shape of the returned array.
        cartesian : bool, optional
            If True, computes the velocity operator in Cartesian coordinates.
            Default is False (reduced coordinates).
        non_abelian : bool, optional
            If True, returns the full Berry curvature tensor (non-abelian case).
            If False, returns the band-trace of the Berry curvature tensor (abelian case).
            Default is False.
        param_periods : dict[str, float], optional
            Optional map ``{param_name: period}`` for swept parameters. When supplied,
            assumes the parameter is cyclic and trims any duplicated endpoint
            before building finite-difference stencils. Parameters not listed here
            are treated as non-periodic unless their sample list starts and ends
            at the same value.
        diff_scheme : str, optional
            Finite difference scheme to use for parameter derivatives.
            Options are "central" (default) or "forward".
            This parameter is only relevant if ``include_lambda=True``.
        diff_order : int, optional
            Order of accuracy for finite difference lambda derivatives.
            Must be an even integer for "central" scheme (default is 2),
            and a positive integer for "forward" scheme.
            This parameter is only relevant if ``include_lambda=True``.
        use_tensorflow: bool, optional
            If True, will use tensorflow to speed up linear algebra routines.
        **params : 
            Keyword arguments mapping parameter names to value(s). Each value can be a scalar
            or a 1D array of values. If any values are array-like,
            the Berry curvature is evaluated at all combinations of parameter values,
            and the final array is stacked with the k-axis leading, followed by each
            parameter axis in the order of given parameter names.

        Returns
        -------
        b_curv : np.ndarray
            Berry curvature tensor. If ``plane`` is None, shape is (dim_k, dim_k, Nk, n_orb, n_orb).
            If ``plane`` is a tuple, shape is (Nk, n_orb, n_orb) and the returned tensor is restricted 
            to the specified directions.
            If ``non_abelian=False``, returns the band-trace of the Berry curvature tensor and the last
            two dimensions are not present. If parameter sweeps are performed via ``params``,
            parameter axes are added after the k-point axis.

        See Also
        --------
        quantum_geometric_tensor : Computes the quantum geometric tensor.
        quantum_metric : Computes the quantum metric tensor.
        velocity : Computes the velocity operator used in the Kubo formula.

        Notes
        -----
        - Specifically, for :math:`(m,n) \in \text{occ}`, the non-Abelian Berry curvature tensor
          is given by (when ``non_abelian=True``):

          .. math::

            \Omega_{\mu \nu;\ mn}(k) =  i\sum_{l \notin \text{occ}}
            \frac{
                \langle u_{mk} | \partial_{\mu} H_k | u_{lk} \rangle
                \langle u_{lk} | \partial_{\nu} H_k | u_{nk} \rangle
                -
                m \leftrightarrow n
            }{
                (E_{nk} - E_{lk})(E_{mk} - E_{lk})
            }
        - This quantity is an anti-symmetric under :math:`\mu \leftrightarrow \nu`. 
        - The Berry curvature is only defined for models with at least 2 k+parameter-space dimensions
          (``dim_k + dim_params >= 2``). 
        - The Berry curvature is computed using the Kubo formula, which
          requires knowledge of the velocity operator :math:`\partial_\mu H_k`. This operator
          is computed using the gradient of the Hamiltonian provided by :func:`velocity`.
        - When using parameter sweeps via ``params``, the Berry curvature is computed
          at all combinations of parameter values, and the resulting array has
          parameter axes added after the k-point axis in the output.
        - When using ``include_lambda=True``, the velocity operator includes
          finite-difference derivatives with respect to parameters, allowing
          computation of Berry curvature components involving parameter directions.
          This should only be used when the parameter values in ``params`` are
          uniformly spaced and form a closed loop in parameter space (endpoints not included)
          for accurate finite difference derivatives.
        """
        Q = self.quantum_geometric_tensor(
            k_pts, 
            occ_idxs=occ_idxs,
            cartesian=cartesian, 
            non_abelian=non_abelian,
            param_periods=param_periods,
            diff_scheme=diff_scheme,
            diff_order=diff_order,
            use_tensorflow=use_tensorflow, 
            **params
        )
        # Berry curvature is the anti-symmetric part of the quantum geometric tensor
        if non_abelian:
            Omega = 1j * (Q - np.swapaxes(Q, -1, -2).conj())
        else:
            Omega = -2 * Q.imag

        if plane is not None:
            # Restrict to specified plane
            mu, nu = plane
            Omega = Omega[mu, nu]
        return Omega

    def quantum_metric(
        self,
        k_pts,
        occ_idxs = None,
        plane = None,
        *,
        cartesian: bool = False,
        non_abelian: bool = False,
        param_periods: dict[str, float] | None = None,
        diff_scheme: str = "central",
        diff_order: int = 2,
        use_tensorflow: bool = False,
        **params
    ):
        r"""Quantum metric in the energy eigenbasis computed via Kubo formula.

        The quantum metric is computed as the Hermitian part of the quantum
        geometric tensor :math:`Q_{\mu \nu}(k)` from :meth:`quantum_geometric_tensor`, 
        i.e., in the non-Abelian case (``non_abelian=True``):

        .. math::

            g_{\mu \nu;\ mn}(k) =  \frac{1}{2} \left( Q_{\mu \nu;\ mn}(k)  + Q_{\mu \nu;\ nm}^*(k) \right)

        In the Abelian case (``non_abelian=False``), the quantum metric is given by the
        band-trace of the above quantity. This reduces to the well-known expression for the
        quantum metric in terms of the quantum geometric tensor.
        
        .. math:: 

           g_{\mu \nu}(k) = \mathrm{Re} \, Q_{\mu \nu}(k),

        By specifying the ``plane`` parameter, we choose a particular :math:`(\mu, \nu)` pair
        of the Berry curvature tensor to return.

        .. versionadded:: 2.0.0


        Parameters
        ----------
        k_pts : (Nk, dim_k) array-like
            Array of k-points with shape (Nk, dim_k), where Nk is the number of points
            and dim_k is the dimensionality of the k-space.
        occ_idxs : 1D array, optional
            Indices of the occupied bands. Defaults to the first half of the states.
        plane : tuple of int, optional
            Tuple of two integers specifying the plane in k-space for which to compute 
            the curvature. If None (default), 
            computes all components of the Berry curvature tensor. This 
            will affect the shape of the returned array.
        cartesian : bool, optional
            If True, computes the velocity operator in Cartesian coordinates.
            Default is False (reduced coordinates).
        non_abelian : bool, optional
            If True, returns the full Berry curvature tensor (non-abelian case).
            If False, returns the band-trace of the Berry curvature tensor (abelian case).
            Default is False.
        param_periods : dict[str, float], optional
            Optional map ``{param_name: period}`` for swept parameters. When supplied,
            assumes the parameter is cyclic and trims any duplicated endpoint
            before building finite-difference stencils. Parameters not listed here
            are treated as non-periodic unless their sample list starts and ends
            at the same value.
        diff_scheme : str, optional
            Finite difference scheme to use for parameter derivatives.
            Options are "central" (default) or "forward".
            This parameter is only relevant if ``include_lambda=True``.
        diff_order : int, optional
            Order of accuracy for finite difference lambda derivatives.
            Must be an even integer for "central" scheme (default is 2),
            and a positive integer for "forward" scheme.
            This parameter is only relevant if ``include_lambda=True``.
        use_tensorflow: bool, optional
            If True, will use tensorflow to speed up linear algebra routines.
        **params : 
            Keyword arguments mapping parameter names to value(s). Each value can be a scalar
            or a 1D array of values. If any values are array-like,
            the quantum metric is evaluated at all combinations of parameter values,
            and the final array is stacked with the k-axis leading, followed by each
            parameter axis in the order of given parameter names.

        Returns
        -------
        g : np.ndarray
            Quantum metric tensor at the specified k-points. If ``plane`` is None,
            returns the full quantum metric tensor. If ``plane`` is specified,
            returns the quantum metric component for that plane. If ``non_abelian=False``,
            returns the band-trace of the quantum metric tensor (abelian case).

        See Also
        --------
        quantum_geometric_tensor
        berry_curvature
        velocity

        Notes
        -----
        - Specifically, for :math:`(m,n) \in \text{occ}`, the non-Abelian quantum metric tensor
          is given by (when ``non_abelian=True``):

          .. math::

            g_{\mu \nu;\ mn}(k) =  \frac{1}{2} \sum_{l \notin \text{occ}}
            \frac{
                \langle u_{mk} | \partial_{\mu} H_k | u_{lk} \rangle
                \langle u_{lk} | \partial_{\nu} H_k | u_{nk} \rangle
                +
                m \leftrightarrow n
            }{
                (E_{nk} - E_{lk})(E_{mk} - E_{lk})
            }

        - This quantity is symmetric under :math:`\mu \leftrightarrow \nu`.
        - The quantum metric is only defined for models with at least 2 k-space dimensions
          (``dim_k >= 2``).
        - The quantum metric is computed using the Kubo formula, which
          requires knowledge of the velocity operator :math:`\partial_\mu H_k`. This operator
          is computed using the gradient of the Hamiltonian provided by :func:`velocity`.
        """
        Q = self.quantum_geometric_tensor(
            k_pts, 
            occ_idxs=occ_idxs, 
            cartesian=cartesian, 
            non_abelian=non_abelian, 
            param_periods=param_periods,
            diff_scheme=diff_scheme,
            diff_order=diff_order,
            use_tensorflow=use_tensorflow, **params
        )

        if non_abelian:
            # Quantum metric is the symmetric part of the quantum geometric tensor
            g = (1/2) * (Q + Q.swapaxes(-1, -2))
        else:
            g = Q.real
        if plane is not None:
            mu, nu = plane
            g = g[mu, nu]
        return g
    
    def axion_angle(
        self,
        nks: tuple[int, int, int] = (20, 20, 20),
        occ_idxs=None,
        *,
        param_periods: dict[str, float] | None = None,
        diff_scheme: str = "central",
        diff_order: int = 4,
        use_tensorflow: bool = False,
        return_second_chern: bool = False,
        **params
    ):
        r"""Chern–Simons axion angle.
        
        Computes the Chern-Simons contribution to the axion angle for 
        a 3D bulk model that depends on a single adiabatic parameter
        :math:`\lambda`. This is computed using the gauge-invariant 4-curvature
        formulation:

        .. math::
            \theta(\lambda) = \frac{1}{16\pi} \int_0^{\lambda} d\lambda'
            \int_{\text{BZ}} d^3k \,
            \epsilon^{\mu\nu\rho\sigma} \mathrm{Tr} \left[
                \mathcal{\Omega}_{\mu\nu}(\mathbf{k}, \lambda')
                \mathcal{\Omega}_{\rho\sigma}(\mathbf{k}, \lambda')
            \right]

        where :math:`\mu, \nu, \rho, \sigma` run over the three reciprocal-space
        directions and the adiabatic parameter :math:`\lambda`, and
        :math:`\mathcal{\Omega}_{\mu\nu}` is the non-Abelian Berry curvature
        tensor over the occupied states.

        When the parameter :math:`\lambda` is periodic (e.g., an angle variable),
        the change in :math:`\theta` over one full cycle is quantized
        in units of :math:`2\pi`, with the integer multiple given by the
        second Chern number :math:`C_2`:

        .. math::
            \Delta \theta = \theta(\lambda + P) - \theta(\lambda)
            = 2\pi C_2,

        where :math:`P` is the period of :math:`\lambda`.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        nks : tuple[int, int, int], optional
            Number of reduced k-points along each reciprocal axis.  All axes are treated
            as periodic and sampled uniformly in ``[0, 1)``.
        occ_idxs : array_like, optional
            Explicit list of occupied band indices.  If omitted, all bands below the gap
            are used, consistent with other bulk invariants.
        param_period : float, optional
            If the adiabatic parameter :math:`\lambda` is periodic, its period. If supplied,
            the parameter sweep is treated as cyclic, and the endpoint is trimmed if it
            matches the starting point plus the period. If not supplied, the parameter
            sweep is treated as non-cyclic unless the supplied parameter values start
            and end at the same value.
        diff_scheme : {"central", "forward"}, optional
            Finite-difference stencil for the adiabatic derivative passed to
            :meth:`berry_curvature`.  Defaults to ``"central"``.
        diff_order : int, optional
            Order of the finite-difference scheme (must be even for ``"central"`` stencils).
        use_tensorflow : bool, optional
            Forwarded to :meth:`berry_curvature`; set ``True`` to accelerate large grids on
            GPU if TensorFlow is installed.
        return_second_chern : bool, optional
            If ``True``, return the second Chern number :math:`C_2` alongside :math:`\theta(\lambda)`.

        Returns
        -------
        lambdas : np.ndarray
            The :math:`\lambda` samples (including the closing point).
        theta : float
            Axion angle at each :math:`\lambda` modulo :math:`2\pi` (folded into :math:`[-\pi, \pi)`).
        c2 : float, optional
            Second Chern number. Only returned when ``return_second_chern=True``.

        Notes
        -----
        - Requires a fully periodic three-dimensional model (``dim_k == 3``).
        - All onsite/hopping providers must still depend on ``param``; if the parameter
          has already been frozen via :meth:`set_parameters`, this routine raises an error.
        - The Berry-curvature tensor is evaluated with ``include_lambda=True``; the mesh
          and finite differences are built internally following the reference notebook
          :ref:`axion-fkm-nb`.
        """
        if self.dim_k != 3:
            raise ValueError("axion_angle requires a three-dimensional periodic model (dim_k == 3).")
        
        if not params:
            raise ValueError("axion_angle requires sweeping a single adiabatic parameter; supply param and lambda_values.")
        
        params = dict(params)
        self._check_missing_parameters(params)

        scalars, sweep_names, sweep_axes = self._params_to_sweep(params)

        # Take first and only swept parameter as adiabatic axis
        if len(sweep_names) != 1:
            raise ValueError("axion_angle expects exactly one swept (array-valued) parameter.")
        sweep_name = sweep_names[0]
        lambda_vals_raw = np.asarray(sweep_axes[0], dtype=float)

        # Trim end points if they duplicate the start (periodic)
        period_dict = param_periods or {}
        logger.debug("axion_angle: normalizing adiabatic parameter axis '%s'.", sweep_name)
        lambda_vals, step_lambda, is_cyclic, trimmed = self._normalize_parameter_axis(
            lambda_vals_raw,
            name=sweep_name,
            period=period_dict.get(sweep_name, None),
        )
        sweep_axes[0] = list(lambda_vals)
        
        nkx, nky, nkz = map(int, nks)
        if min(nkx, nky, nkz) < 2:
            raise ValueError("Each k-axis must contain at least two points.")
        
        # Build uniform reduced k-grid in [0, 1)
        k_axes = [np.linspace(0.0, 1.0, n, endpoint=False) for n in (nkx, nky, nkz)]
        k_grid = np.stack(np.meshgrid(*k_axes, indexing="ij"), axis=-1).reshape(-1, self.dim_k)

        # Assemble kwargs for berry_curvature (scalars + the sweep)
        bc_kwargs = {name: value for name, value in scalars.items()}
        bc_kwargs[sweep_name] = lambda_vals

        # evaluate the non-Abelian 4D Berry curvature
        logger.debug("axion_angle: computing Berry curvature on %d x %d x %d x %d grid", nkx, nky, nkz, lambda_vals.size)
        curvature = self.berry_curvature(
            k_grid,
            occ_idxs=occ_idxs,
            non_abelian=True,
            param_periods=param_periods,
            diff_scheme=diff_scheme,
            diff_order=diff_order,
            use_tensorflow=use_tensorflow,
            **bc_kwargs,
        )

        epsilon = levi_civita(4, 4).astype(curvature.dtype, copy=False)
        c2_density = np.einsum("ijkl,ij...mn,kl...nm->...", epsilon, curvature, curvature).real
        c2_density = c2_density.reshape(nkx, nky, nkz, lambda_vals.size)

        # sum over the k-space slab to obtain per-lambda slices
        d_lambda = c2_density.sum(axis=(0, 1, 2))

        # integration weights
        delta_k = 1.0 / (nkx * nky * nkz)
        # 4-volume element: (1/Nk)^3 * delta_lambda (k sampled in reduced units)
        volume_element = delta_k * step_lambda

        # second Chern number from full integral over closed 4D manifold
        # NOTE: shouldn't include endpoints (trimmed already)
        c2 = (volume_element / (32.0 * np.pi**2)) * d_lambda.sum()

        # cumulative trapezoid along parameter
        pref = volume_element / (16.0 * np.pi)
        cumulative = np.empty_like(d_lambda, dtype=float)
        cumulative[0] = 0.0
        if d_lambda.size > 1:
            mids = 0.5 * (d_lambda[:-1] + d_lambda[1:])
            cumulative[1:] = np.cumsum(mids) * pref
        
        if is_cyclic and trimmed:
            logger.debug("axion_angle: Appending endpoint to axion angle for trimmed cyclic parameter sweep.")
            theta_ep = cumulative[-1] + pref * 0.5 * (d_lambda[-1] + d_lambda[0])
            lambdas_total = np.append(lambda_vals, lambda_vals[0] + step_lambda * lambda_vals.size)
            theta_total = np.append(cumulative, theta_ep)
        else:
            lambdas_total = lambda_vals.copy()
            theta_total = cumulative

        # wrap theta into [0, 2*pi)
        thetas_wrapped = np.unwrap(theta_total, period=2.0 * np.pi)

        outputs = [lambdas_total, thetas_wrapped]
        if return_second_chern:
            outputs.append(c2.real)
       
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    #TODO: Allow chern number on parameter planes
    def chern_number(
            self, 
            occ_idxs=None, 
            plane: tuple = (0, 1), 
            k_mesh: tuple = (100, 100),
            diff_scheme: str = "central",
            diff_order: int = 2,
            **params
            ):
        r"""Computes Chern number for occupied manifold.

        The Chern number is computed by integrating the Berry curvature
        over a 2d surface in reciprocal space defined by `plane` parameter.
        The Chern number is given by

        .. math::
            C = \frac{1}{2\pi} \int_{\text{2d surface}} d^2k \, \Omega(k)

        where :math:`\Omega(k)` is the trace of the Berry curvature
        tensor over the occupied bands.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        occ_idxs : array-like, optional
            Occupied band indices. If none are provided, 
            the lower half bands are considered occupied.
        plane : tuple[int, int], optional
            Indices for reciprocal space directions defining
            2D surface to integrate Berry flux.
        k_mesh : tuple[int, int], optional
            Tuple ``(nk1, nk2)`` of number of k-points along each direction 
            in the 2D surface for performing the integration. 
            Default is ``(100, 100)``.

        Returns
        -------
        chern_num : float
            Chern number for the occupied manifold.

        Notes
        -----
        This function only works for models with at least 2 k-space
        dimensions (``dim_k >= 2``). The Chern number is only defined
        for 2D surfaces in k-space, so `plane` must be a tuple of
        length 2. The Chern number is guaranteed to be an integer
        (within numerical accuracy) if the occupied manifold is
        separated by an energy gap from the unoccupied manifold over
        the entire 2D surface in k-space.
        """
        if self.dim_k < 2:
            raise ValueError("Chern number requires at least 2 k-space dimensions (dim_k >= 2).")
        if not (isinstance(plane, tuple) and len(plane) == 2):
            raise ValueError("plane must be a tuple of length 2.")
        # check k_mesh is tuple of all ints and length dim_k
        if not (isinstance(k_mesh, tuple) and len(k_mesh) == 2 and all(isinstance(nk, int) for nk in k_mesh)):
            raise ValueError("k_mesh must be a tuple of length 2 with integer values.")

        nks = k_mesh
        k_grid = self.k_uniform_mesh(nks)
        k_flat = k_grid.reshape(-1, 2)
        
        Omega = self.berry_curvature(
            k_flat, 
            occ_idxs=occ_idxs, 
            plane=plane,
            diff_scheme=diff_scheme,
            diff_order=diff_order, 
            **params
        )

        if plane[0] >= self.dim_k or plane[1] >= self.dim_k:
            raise NotImplementedError(
                "Chern number can only be computed for planes within k-space dimensions."
            )

        Nk = Omega.shape[0]
        dk_sq = 1 / Nk
        chern_num = np.sum(Omega) * dk_sq / (2 * np.pi)

        return chern_num.real

    #TODO: Handle params lists
    def local_chern_marker(
            self, 
            occ_idxs=None,
            return_bulk_avg: bool = False,
            trim_cells: int = 4,
            **params
            ):
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

        .. versionadded:: 2.0.0

        Parameters
        ----------
        occ_idxs : array-like, optional
            Indices of the occupied bands. If none are provided, 
            the lower half bands are considered occupied.
        return_bulk_avg : bool, optional
            If True, also returns the bulk-averaged Chern number
            computed from the local Chern marker. Default is False.
        trim_cells : int, optional
            Number of unit cells to trim from each edge when computing
            the bulk-averaged Chern number. Default is 4.

        Returns
        -------
        C_local : np.ndarray of shape (norb,)
            Per-site local Chern marker.
        C_bulk_avg : float, optional
            Bulk-averaged Chern number computed from local Chern marker.
            Returned only if `return_bulk_avg` is True.
        """
        if self.dim_k != 0:
            raise ValueError("Local Chern marker is only defined for real-space models (dim_k=0).")
        if self.dim_r != 2:
            raise NotImplementedError("Local Chern marker is only defined for 2D models (dim_r=2).")
        
        H = self.hamiltonian(flatten_spin_axis=True, **params)  # (..., N, N) dense
        N = H.shape[-1]
        r_cart = self.get_orb_vecs(cartesian=True) # (N, 2)
        if r_cart.ndim != 2 or r_cart.shape[1] != 2 or r_cart.shape[0] != N:
            raise ValueError("Could not get orbital coordinates in Cartesian basis.")
        x = r_cart[:, 0]
        y = r_cart[:, 1]

        # Dense eigensolve and projector
        _, evecs = np.linalg.eigh(H)  # returns sorted ascending
        if occ_idxs is None:
            # Default to half filling (robust for particle-hole symmetric models like Haldane).
            occ_idxs = np.arange(N // 2)
        else:
            occ_idxs = np.asarray(occ_idxs, int)

        Uocc = evecs[..., occ_idxs]  # (N, k_occ)
        P = Uocc @ Uocc.conj().T  # (N,N) dense projector

        DX = x[:, None] - x[None, :]
        DY = y[:, None] - y[None, :]
        CX = DX * P
        CY = DY * P

        # A = P [X,P][Y,P]
        A = P @ (CX @ CY)

        # Local marker from diagonal of A
        A_cell = self.cell_volume
        C_local = (4 * np.pi / A_cell) * np.imag(np.diag(A))

        if not return_bulk_avg:
            return C_local

        Lx = self.lattice.nsuper[0]
        Ly = self.lattice.nsuper[1]

        # number of orbitals per Bravais cell (handles spin implicitly via N)
        if N % (Lx * Ly) != 0:
            raise ValueError(f"N={N} not divisible by Lx*Ly={Lx*Ly}; "
                            "cannot aggregate per-cell markers.")
        norb_cell = self.norb // (Lx * Ly)

        # Per-cell marker by summing orbitals belonging to the same cell
        # (use fractional positions to infer integer cell index robustly)
        r_frac = self.get_orb_vecs(cartesian=False) # (N, 2)

        # Per-cell fractional centers (average over orbitals of each cell)
        # We don't assume any orbitals ordering; we infer cell indices by rounding.
        # Compute approximate cell indices using fractional coords:
        # Step 1: reshape a best-guess to estimate the internal offset
        # If ordering is arbitrary, estimate offset from all orbitals:
        offset = np.mod(r_frac, 1.0).mean(axis=0)
        ij_float = r_frac - offset  # ~ integers (ix, iy) per orbital
        ij = np.rint(ij_float).astype(int)
        ix = np.mod(ij[:, 0], Lx)
        iy = np.mod(ij[:, 1], Ly)
        lin = ix + Lx * iy  # linear cell index in C-order (ix fastest)

        # Aggregate per-cell local marker
        marker_cell = np.zeros(Lx * Ly, dtype=C_local.dtype)
        np.add.at(marker_cell, lin, C_local)

        # Normalize trim argument
        if isinstance(trim_cells, int):
            tx = ty = int(trim_cells)
        else:
            tx, ty = map(int, trim_cells)
        tx = max(0, tx)
        ty = max(0, ty)
        if 2 * tx >= Lx or 2 * ty >= Ly:
            raise ValueError(f"trim_cells={trim_cells} too large for grid {(Lx, Ly)}")

        # Build mask over interior cells (C-order indexing)
        IX, IY = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing="xy")
        mask = (IX.ravel() >= tx) & (IX.ravel() < Lx - tx) & \
            (IY.ravel() >= ty) & (IY.ravel() < Ly - ty)

        C_bulk_avg = marker_cell[mask].mean()

        return C_local, C_bulk_avg
    
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

        Parameters
        ----------
        evecs : np.ndarray
            Eigenvectors for which we are computing matrix
            elements of the position operator.  The shape of this array
            is ``evecs[band, orbital]`` if ``spinful=False`` and
            ``evecs[band, orbital, spin]`` if ``spinful=True``.

            .. versionchanged:: 2.0.0
                Parameter ``evec`` renamed to ``evecs`` to clarify that multiple
                eigenvectors can be passed at once.

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
        if not self.assume_position_operator_diagonal:
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

        Parameters
        ----------
        evecs : np.ndarray
            Eigenvectors for which we are computing matrix
            elements of the position operator. The shape of this array
            is ``evecs[band, orbital]`` if ``spinful=True`` and
            ``evecs[band, orbital, spin]`` if ``spinful=False``.

            .. versionchanged:: 2.0.0
                Parameter ```evec`` renamed to ``evecs`` to clarify that multiple
                eigenvectors can be passed at once.

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
        if not self.assume_position_operator_diagonal:
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

        Parameters
        ----------
        evecs : np.ndarray
            Eigenvectors for which we are computing matrix
            elements of the position operator. The shape of this array
            is ``evecs[band, orbital]`` if ``spinful=True`` and
            ``evecs[band, orbital, spin]`` if ``spinful=False``.

            .. versionchanged:: 2.0.0
                Parameter ``evec`` renamed to ``evecs`` to clarify that multiple
                eigenvectors can be passed at once.

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
        if not self.assume_position_operator_diagonal:
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

        .. versionchanged:: 2.0.0
            Visualization appearance has been updated.

        Parameters
        ----------
        proj_plane : tuple or list of two integers
            Cartesian coordinates to be used for plotting. For example,
            if ``proj_plane=(0,1)`` then x-y projection of the model is
            drawn. This only should be specified if `dim_r` > 2.

            .. versionchanged:: 2.0.0
                Replaced previous parameters ``dir_first`` and ``dir_second``.

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

        .. versionadded:: 2.0.0

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

        .. versionadded:: 2.0.0

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
