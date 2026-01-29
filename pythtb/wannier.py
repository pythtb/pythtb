import numpy as np
import logging
from .wfarray import WFArray
from .visualization import plot_centers, plot_decay, plot_density
from .mesh import Mesh
from .utils import mat_exp, copydoc
from itertools import product
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass

__all__ = ["Wannier"]


class Wannier:
    r"""Construct Wannier functions using the projection method.

    This class provides methods to build Wannier functions from Bloch eigenstates and to
    evaluate their quadratic spreads. It realizes the Marzari-Vanderbilt maximal localization
    program on a **toroidal k-mesh without endpoints** by combining

    - **Single-shot projection** via SVD alignment to user-specified trial orbitals, producing
      Bloch-like states :math:`\tilde{\psi}_{n\mathbf{k}}`.
    - **Disentanglement** of entangled bands following the Souza-Marzari-Vanderbilt [2]_ scheme,
      where :meth:`disentangle` optimizes projectors within user-defined outer and inner windows
      to minimize the gauge-invariant spread :math:`\Omega_I`.
    - **Maximal localization** thorugh the Marzari-Vanderbilt scheme [1]_ using
      :meth:`maxloc`, which iteratively rotates the
      disentangled subspace to minimize the gauge-dependent spread
      :math:`\widetilde{\Omega}=\Omega_{\mathrm{OD}}+\Omega_{\mathrm{D}}`.

    Together these steps construct Wannier functions from Bloch energy eigenstates and minimize
    their spreads through disentanglement and maximal localization.

    Parameters
    ----------
    bloch_states : WFArray
        The Bloch wavefunctions to be Wannierized. This should be initialiazed on a toroidal
        k-mesh without endpoints (open k-axes). They should also have their values filled, either
        by setting manually or by calling :meth:`WFArray.solve_model`. These can be energy eigenstates
        or any other set of Bloch-like states on the mesh.

    Notes
    -----
    - The k-mesh must be a torus (no endpoints included).
    - :attr:`tilde_states` must be set before calling routines that compute Wannier functions
      or spreads.
    - **Disentanglement workflow**: :meth:`disentangle` uses outer (candidate) and inner
      (frozen) windows to optimize projectors that define a smooth subspace with minimal
      gauge-invariant spread.
    - **Maximal localization**: :meth:`maxloc` performs a steepest-descent update of
      the unitary gauge to reduce the gauge-dependent spread and produce maximally localized
      Wannier functions.
    - **Wannier construction**: We form Wannier functions by discrete inverse Fourier transform
      of the Bloch-like states,

      .. math::
           w_{n\mathbf{R}} = \frac{1}{\sqrt{N_k}} \sum_{\mathbf{k}}
           e^{i\mathbf{k}\cdot\mathbf{R}}\,\tilde{\psi}_{n\mathbf{k}} .

      In code this is done by ``np.fft.ifftn`` over the k-axes.
    - **Overlap matrices**: For nearest-neighbor k-shell displacements
      :math:`\{\mathbf{b}\}` with weights :math:`\{w_b\}`, the code constructs the
      discrete overlaps

      .. math::
           M_{mn}^{(\mathbf{b})}(\mathbf{k}) \equiv
           \langle u_{m\mathbf{k}} \mid u_{n,\mathbf{k}+\mathbf{b}}\rangle ,

      using the cell-periodic parts of the Bloch-like states stored in :attr:`tilde_states`.
    - **Spread decomposition (MV97)**: The total spread
      :math:`\Omega=\Omega_I+\widetilde{\Omega}` is decomposed into the gauge-invariant
      part :math:`\Omega_I` and the gauge-dependent part
      :math:`\widetilde{\Omega}=\Omega_{\mathrm{OD}}+\Omega_{\mathrm{D}}`. The code computes
      these using the diagonal and full elements of :math:`M^{(\mathbf{b})}` as in
      [1]_.

    - **Centers and phases**: The Wannier-center vector for band :math:`n` is obtained from
      the phases of the diagonal overlaps,

      .. math::
           \mathbf{r}_n \;=\; -\frac{w_b}{N_k}
           \sum_{\mathbf{k}} \operatorname{Im}\!\left[\ln M_{nn}^{(\mathbf{b})}(\mathbf{k})\right]
           \, \mathbf{b} ,

    References
    ----------
    .. [1] Marzari, N., & Vanderbilt, D. Maximally localized generalized
        Wannier functions for composite energy bands. Phys. Rev. B 56, 12847 (1997).
    .. [2] Souza, I., Marzari, N., & Vanderbilt, D. Maximally localized
        Wannier functions for entangled energy bands. Phys. Rev. B 65, 035109 (2001).

    Examples
    --------
    Initialize the Bloch :class:`WFArray` on a toroidal k-mesh without endpoints

    >>> from pythtb import TBModel, Lattice, Mesh, WFArray, Wannier
    >>> lat_vecs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> orb_vecs = [[0, 0, 0]]
    >>> lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1, 2])
    >>> model = TBModel(lattice=lat, spinful=False)
    >>> model.set_onsite(0.0)
    >>> model.set_hop(-1.0, 0, 0, [1, 0, 0]) # along with other hops...
    >>> mesh = Mesh(dim_k=3, axis_types=['k', 'k', 'k'])
    >>> mesh.build_grid(shape=(8, 8, 8), k_endpoints=False)
    >>> wfa = WFArray(lattice=lat, mesh=mesh, nstates=model.nbands, spinful=False)
    >>> wfa.solve_model(model)
    >>> wan = Wannier(bloch_states=wfa)

    Set trial wavefunctions and perform single-shot projection

    >>> twf_list = [[(0, 1.0)]]  # single trial wf on orbital 0
    >>> wan.project(twf_list)

    Perform maximal localization

    >>> wan.maxloc(num_iter=100, conv_tol=1e-6)

    Compute and print Wannier centers and spreads

    >>> wan.info(precision=6)

    Plot Wannier centers

    >>> wan.plot_centers(wan_idx=0)
    """

    def __init__(self, bloch_states: WFArray):
        self._wfa: WFArray = bloch_states

        if not self.mesh.is_k_torus:
            raise ValueError(
                "Mesh is not a torus. The Wannier class requires a toroidal k-mesh."
                "To construct a toroidal k-space mesh use `Mesh.build_grid`"
            )
        for ax in self.mesh.k_axes:
            if ax.has_endpoint:
                raise ValueError(
                    f"Detected a closed k-axis: {ax}. The endpoints of the Brillouin zone "
                    f"must not be included."
                )

        ranges = [np.arange(-nk // 2, nk // 2) for nk in self.nks]
        mesh = np.meshgrid(*ranges, indexing="ij")
        # used for real space looping of WFs
        self.supercell = np.stack(mesh, axis=-1).reshape(  # (..., len(nks))
            -1, len(self.nks)
        )  # (product, dims)

    @property
    def mesh(self) -> Mesh:
        """Mesh object associated with the Wannier functions."""
        return self.bloch_states.mesh

    @property
    def lattice(self):
        """Lattice object associated with the Wannier functions."""
        return self._wfa.lattice

    @property
    def bloch_states(self) -> WFArray:
        """WFArray object associated with the Bloch states."""
        return self._wfa

    @property
    def tilde_states(self) -> WFArray:
        r"""WFArray corresponding to the Bloch-like (tilde) states.

        These are the Bloch-like states that are Fourier transformed to
        form the Wannier functions. They are related to the original energy
        eigenstates via the (semi-) unitary transformation

        .. math::
            |\tilde{\psi}_{n\mathbf{k}} \rangle = \sum_{m=1}^{N}
            U_{mn}^{(\mathbf{k})} |\psi_{m\mathbf{k}} \rangle
        """
        if not hasattr(self, "_tilde_states"):
            raise ValueError(
                "Bloch-like states have not been set. "
                "Use `set_tilde_states` or `project`."
            )
        return getattr(self, "_tilde_states", None)

    @property
    def nks(self) -> list:
        """Number of k-points in each dimension."""
        return self.mesh.shape_k

    @property
    def wannier(self) -> np.ndarray:
        r"""Wannier functions.

        The Wannier functions are the discrete Fourier transform of the
        Bloch-like states :math:`\tilde{\psi}`

        .. math::
            w_{n\mathbf{R}} = \frac{1}{\sqrt{N_k}} \sum_{\mathbf{k}} e^{i\mathbf{k} \cdot \mathbf{R}}
            \tilde{\psi}_{n\mathbf{k}}

        where :math:`N_k` is the number of k-points, :math:`\mathbf{R}` is a
        lattice vector conjugate to the discrete k-mesh.
        """
        if not self.tilde_states.filled:
            raise ValueError("Tilde states are not initialized.")
        return getattr(self, "_wannier", None)

    @property
    def spread(self) -> list[float]:
        r"""Quadratic spread for each Wannier function.

        .. math::
            \Omega_n = \langle \mathbf{0} n | r^2 | \mathbf{0} n \rangle
            - \langle \mathbf{0} n | \mathbf{r} | \mathbf{0} n \rangle^2

        where :math:`|\mathbf{0} n\rangle` are the Wannier functions in the home unit cell
        and :math:`\Omega = \sum_n \Omega_n` is the total spread.
        """
        if not self.tilde_states.filled:
            raise ValueError("Tilde states are not initialized.")
        return getattr(self, "_spread", None)

    @property
    def Omega_OD(self) -> float:
        r"""Off-diagonal part of gauge-dependent spread.

        Part of the decomposition of the quadratic spread into gauge-invariant (:math:`\widetilde{\Omega}`) and
        gauge-dependent (:math:`\widetilde{\Omega}`) parts,

        .. math::
            \Omega = \widetilde{\Omega} + \Omega_I =  \Omega_{\rm OD} + \Omega_{\rm D}
            + \Omega_I

        The off-diagonal part :math:`\Omega_{\rm OD}` is computed via

        .. math::
            \Omega_{\rm OD} = \frac{1}{N_k} \sum_{\mathbf{k}, \mathbf{b}} w_b
            \sum_{m\neq n} |M_{mn}^{(\mathbf{b})}(\mathbf{k})|^2
        """
        if not self.tilde_states.filled:
            raise ValueError("Tilde states are not initialized.")
        return getattr(self, "_omega_od", None)

    @property
    def Omega_D(self) -> float:
        r"""Off-diagonal part of gauge-dependent spread.

        Part of the decomposition of the quadratic spread into gauge-invariant (:math:`\widetilde{\Omega}`) and
        gauge-dependent (:math:`\widetilde{\Omega}`) parts,

        .. math::
            \Omega = \widetilde{\Omega} + \Omega_I = \Omega_{\rm OD} + \Omega_{\rm D}
            + \Omega_I

        The diagonal part :math:`\Omega_{\rm D}` is computed via

        .. math::
            \Omega_{\rm D} = \frac{1}{N_k} \sum_{\mathbf{k}, \mathbf{b}} w_b
            \sum_n \left( -\operatorname{Im}\!\left[\ln M_{nn}^{(\mathbf{b})}(\mathbf{k})\right]
            - \mathbf{b}\cdot\mathbf{r}_n \right)^2
        """
        if not self.tilde_states.filled:
            raise ValueError("Tilde states are not initialized.")
        return getattr(self, "_omega_d", None)

    @property
    def Omega_I(self) -> float:
        r"""Gauge-independent quadratic spread.

        Part of the decomposition of the quadratic spread into gauge-invariant (:math:`\widetilde{\Omega}`) and
        gauge-dependent (:math:`\widetilde{\Omega}`) parts,

        .. math::
            \Omega = \widetilde{\Omega} + \Omega_I = \Omega_{\rm OD} + \Omega_{\rm D} + \Omega_I

        The gauge-invariant part :math:`\Omega_I` is independent of the choice of Wannier gauge. It is
        computed via

        .. math::
            \Omega_I = \frac{1}{N_k} \sum_{\mathbf{k}, \mathbf{b}} w_b
            \left( N_{\rm bands} - \sum_{m,n} |M_{mn}^{(\mathbf{b})}(\mathbf{k})|^2 \right)

        """
        if not self.tilde_states.filled:
            raise ValueError("Tilde states are not initialized.")
        return getattr(self, "_omega_i", None)

    @property
    def centers(self) -> np.ndarray:
        r"""Centers of the Wannier functions in Cartesian coordinates.

        The Wannier center for band :math:`n` is obtained from the phases of the
        diagonal overlaps,

        .. math::
            \mathbf{r}_n \;=\; -\frac{1}{N_k}
            \sum_{\mathbf{k}, \mathbf{b}} w_b \mathbf{b} \operatorname{Im}\!\left[\ln M_{nn}^{(\mathbf{b})}(\mathbf{k})\right]
            \, ,
        """
        if not self.tilde_states.filled:
            raise ValueError("Tilde states are not set.")
        return getattr(self, "_centers", None)

    @property
    def trial_wfs(self) -> np.ndarray:
        """Trial wavefunctions to project onto."""
        return getattr(self, "_trial_wfs", None)

    @property
    def num_twfs(self) -> int:
        """Number of trial wavefunctions."""
        if self.trial_wfs is None:
            raise ValueError("Trial wavefunctions are not set.")
        return self.trial_wfs.shape[0]

    @property
    def Amn(self) -> np.ndarray:
        r"""Overlap matrix between energy eigenstates and trial wavefunctions.

        The overlap matrix is defined as

        .. math::

            A(\mathbf{k})_{mn} = \langle \psi_{m \mathbf{k}} | t_{n} \rangle

        where :math:`|\psi_{n\mathbf{k}}\rangle` are the Bloch energy eigenstates and
        :math:`|t_j\rangle` are the trial wavefunctions.
        """
        return getattr(self, "_A", None)

    def info(self, precision=8):
        """Report of Wannier centers and spreads.

        Parameters
        ----------
        precision : int
            The number of decimal places to include in the report.
        """

        if not getattr(self.tilde_states, "filled", False):
            raise ValueError("Tilde states are not set.")

        spreads = np.asarray(self.spread, float)
        centers = np.atleast_2d(np.asarray(self.centers, float))

        n, d = centers.shape
        lines = ["Wannier Function Report"]

        # individual WF rows
        for i, (c, s) in enumerate(zip(centers, spreads), 1):
            c_str = ", ".join(f"{x:.{precision}f}" for x in c)
            lines.append(f"WF {i}: center = [{c_str}]  Omega     = {s:.{precision}f}")

        # totals
        sum_c = centers.sum(axis=0)
        sum_s = spreads.sum()
        sum_c_str = ", ".join(f"{x:.{precision}f}" for x in sum_c)
        lines.append(f"Sum : center = [{sum_c_str}]  Omega tot = {sum_s:.{precision}f}")

        # Omegas
        Omega_I = float(getattr(self, "Omega_I", np.nan))
        Omega_D = float(getattr(self, "Omega_D", np.nan))
        Omega_OD = float(getattr(self, "Omega_OD", np.nan))
        Omega_tot = Omega_I + Omega_D + Omega_OD

        lines += [
            f"Omega I   = {Omega_I:.{precision}f}",
            f"Omega D   = {Omega_D:.{precision}f}",
            f"Omega OD  = {Omega_OD:.{precision}f}",
            f"Omega tot = {Omega_tot:.{precision}f}",
        ]

        # determine longest line
        maxlen = max(len(line) for line in lines)
        divider = "=" * maxlen
        sub_div = "-" * maxlen

        # insert dividers at appropriate places
        lines.insert(1, divider)
        lines.insert(len(lines) - 4, sub_div)

        out = "\n".join(lines)
        print(out)

    def get_centers(self, cartesian=False):
        """Get the centers of the Wannier functions.

        Parameters
        ----------
        cartesian : bool, optional
            If True, return the centers in Cartesian coordinates.
            If False, return the centers in fractional coordinates.

        Returns
        -------
        np.ndarray
            The centers of the Wannier functions.
        """
        if cartesian:
            return self.centers
        else:
            return self.centers @ np.linalg.inv(self.lattice.lat_vecs)

    def _get_trial_wfs(self, twf_list=None):
        if twf_list is None:
            return self._trial_wfs

        # number of trial functions to define
        num_tf = len(twf_list)
        if self.bloch_states.spinful:
            tfs = np.zeros(
                [num_tf, self.lattice.norb, self.bloch_states.nspin], dtype=complex
            )
            for j, tf in enumerate(twf_list):
                assert isinstance(tf, (list, np.ndarray)), (
                    "Trial function must be a list of tuples"
                )
                for orb, spin, amp in tf:
                    tfs[j, orb, spin] = amp
                tfs[j] /= np.linalg.norm(tfs[j])
        else:
            # initialize array containing tfs = "trial functions"
            tfs = np.zeros([num_tf, self.lattice.norb], dtype=complex)
            for j, tf in enumerate(twf_list):
                assert isinstance(tf, (list, np.ndarray)), (
                    "Trial function must be a list of tuples"
                )
                for site, amp in tf:
                    tfs[j, site] = amp
                tfs[j] /= np.linalg.norm(tfs[j])

        return tfs

    def set_trial_wfs(self, tf_list):
        r"""Set trial wavefunctions for Wannierization.

        Parameters
        ----------
        tf_list: list[list[tuple]]
            List of trial wavefunctions. Each trial wavefunction is
            a list of the form ``[(orb, amp), ...]``, where `orb` is the orbital index
            and `amp` is the amplitude of the trial wavefunction on that tight-binding
            orbital. If spin is included, then the form is ``[(orb, spin, amp), ...]``.
            The states are normalized internally, only the relative weights matter.

        Examples
        --------
        For a system with 4 orbitals and no spin, the following defines two trial wavefunctions:

        >>> twf_list = [[(0, 1.0), (2, 1.0)], [(1, 1.0), (3, -1.0)]]
        >>> wan.set_trial_wfs(twf_list)

        This defines two trial wavefunctions: the first is an equal superposition of orbitals
        0 and 2, and the second is an equal superposition of orbitals 1 and 3 with a relative
        minus sign.
        """
        self._trial_wfs = self._get_trial_wfs(tf_list)
        self._tilde_states: WFArray = WFArray(
            self.lattice,
            self.mesh,
            nstates=self.num_twfs,
            spinful=self.bloch_states.spinful,
        )

    def set_tilde_states(self, states, is_cell_periodic=True, is_spin_axis_flat=False):
        r"""Set the Bloch-like states for the Wannier functions.

        These states are Fourier transformed to form the Wannier functions.
        They are related to the original energy eigenstates via the (semi-) unitary transformation

        .. math::

            |\tilde{\psi} \rangle = \sum_{m=1}^{N}
            U_{mn}^{(\mathbf{k})} |\psi_{m\mathbf{k}} \rangle

        Parameters
        ----------
        states : np.ndarray
            The states to set as Bloch-like states. Must have the shape
            ``(nk1, ..., nstates, n_orbs[, n_spins])``.
        cell_periodic : bool, optional
            Whether to treat the ``states`` as cell-periodic, by default False.
        spin_flattened : bool, optional
            Whether the spin dimension is flattened into the orbital dimension.
            If True, ``states`` must have shape ``(nk1, ..., nstates, n_orbs*n_spins)``.
            If False, ``states`` must have shape ``(nk1, ..., nstates, n_orbs, n_spins)``.
            By default False.

        Raises
        ------
        ValueError
            If the input states are not a numpy array or have an invalid shape.

        Notes
        -----
        - If ``cell_periodic`` is True, the states are treated as cell-periodic parts of
          Bloch functions :math:`u_{n\mathbf{k}}`, otherwise as full Bloch functions
          :math:`\psi_{n\mathbf{k}}`.
        - If ``spin_flattened`` is False and the wavefunctions have spin, the states are reshaped
          to flatten the spin dimension into the orbital dimension.
        - The Wannier functions, spreads, and centers are computed upon setting the
          Bloch-like states.
        """
        if not isinstance(states, np.ndarray):
            raise ValueError("Bloch-like states must be a numpy array.")

        if not is_spin_axis_flat and (
            states.ndim != self.mesh.nk_axes + 2 + (self.bloch_states.nspin - 1)
        ):
            raise ValueError(
                f"Bloch-like states must have shape (nk1, ..., nstates, n_orbs[, n_spins]), "
                f"but got {states.shape}."
            )
        elif is_spin_axis_flat and (states.ndim != self.mesh.nk_axes + 2):
            raise ValueError(
                f"Bloch-like states must have shape (nk1, ..., nstates, n_orbs*n_spins), "
                f"but got {states.shape}."
            )

        if self.bloch_states.spinful and not is_spin_axis_flat:
            states = states.reshape((*states.shape[:-2], -1))

        logger.info("Setting Bloch-like states...")
        self.tilde_states.set_states(
            states,
            is_cell_periodic=is_cell_periodic,
            is_spin_axis_flat=is_spin_axis_flat,
        )

        # Fourier transform Bloch-like states to set WFs
        psi_nk = self.tilde_states.psi_nk
        nk_axes = self.mesh.nk_axes

        # FFT NOTE: A non-repeating grid is required for consistent inverse FFTs.
        self._wannier = self.WFs = np.fft.ifftn(
            psi_nk, axes=[i for i in range(nk_axes)], norm=None
        )

        # set spreads
        spread = self._spread_recip(decomp=True)
        self._spread = spread[0][0]
        self._omega_i = spread[0][1]
        self._omega_d = spread[0][2]
        self._omega_od = spread[0][3]
        self._centers = spread[1]

    def _compute_Amn(self, psi_nk, twfs, band_idxs):
        r"""Overlap matrix between Bloch states and trial wavefunctions.

         The overlap matrix is defined as

         .. math::

             A_{k, n, j} = <psi_{n,k} | t_{j}>

         where :math:`|\psi_{n\mathbf{k}}\rangle` are the Bloch energy eigenstates and
         :math:`|t_j\rangle` are the trial wavefunctions.

        Parameters
        -----------
        psi_nk : np.ndarray, optional
            The Bloch states to form the overlap matrix with. By default this will
            choose the energy eigenstates. Shape: ``(*shape_mesh, states, orbs*n_spin])``
        twfs : np.ndarray
            Trial wavefunctions, shape: (n_trial_wfs, orbs[, n_spin])
        band_idxs : list
            Indices of energy bands to project.

        Returns
        --------
        A : np.ndarray
            Overlap matrix with shape ``(*shape_mesh, n_bands, n_trial_wfs)``
        """

        if psi_nk is None:
            # get Bloch psi_nk energy eigenstates
            _, psi_nk = self.bloch_states.states(
                flatten_spin_axis=True, return_psi=True
            )

        # only keep band_idxs
        psi_nk = np.take(psi_nk, band_idxs, axis=-2)

        trial_wfs = twfs
        # flatten along spin dimension in case spin is considered
        trial_wfs = trial_wfs.reshape((*trial_wfs.shape[:1], -1))

        A_k = np.einsum("...ij, kj -> ...ik", psi_nk.conj(), trial_wfs)
        return A_k

    def _single_shot_project(self, psi_nk, twfs, state_idx):
        """
        Performs optimal alignment of psi_nk with trial wavefunctions.

        Parameters
        ----------
        psi_nk : np.ndarray
            Bloch states to be projected, shape: (*mesh_shape, states, orbs*n_spin])
        twfs : np.ndarray
            Trial wavefunctions, shape: (n_trial_wfs, orbs[, n_spin])
        state_idx : list
            Indices of energy bands to project.
        """
        A_k = self._compute_Amn(psi_nk, twfs, state_idx)
        V_k, _, Wh_k = np.linalg.svd(A_k, full_matrices=False)

        # take only state_idxs
        psi_nk = np.take(psi_nk, state_idx, axis=-2)

        # optimal alignment
        psi_tilde = np.einsum(
            "...mn, ...mj -> ...nj", V_k @ Wh_k, psi_nk
        )  # shape: (*mesh_shape, states, orbs*n_spin])

        return psi_tilde

    def project(self, tf_list: list = None, band_idxs: list = None, use_tilde=False):
        r"""Perform Wannierization via optimal alignment with trial functions (single-shot SVD).

        Constructs Bloch-like states :math:`\tilde{\psi}_{n\mathbf{k}}` by maximizing their overlap
        with user-specified trial functions using a per-:math:`\mathbf{k}` singular-value decomposition.
        Specifically, for the overlap matrix

        .. math::

             A_{n j}(\mathbf{k}) \;=\; \langle \psi_{n\mathbf{k}} \mid t_j \rangle ,

        we compute :math:`A(\mathbf{k}) = V(\mathbf{k}) \Sigma(\mathbf{k}) W^\dagger(\mathbf{k})`
        and rotate the selected energy eigenstates by :math:`U(\mathbf{k}) \equiv V(\mathbf{k})
        W^\dagger(\mathbf{k})`:

        .. math::

             \tilde{\psi}_{n\mathbf{k}}
             \;=\; \sum_{m\in \texttt{band_idxs}} U_{nm}(\mathbf{k}) \, \psi_{m\mathbf{k}} .

        This realizes the optimal alignment described in [1]_ (Sec. II) and used
        in the disentanglement initialization of [2]_.

        Sets the Wannier functions in home unit cell with associated spreads, centers, trial functions
        and Bloch-like (tilde) states using the single shot projection method.

        Parameters
        ----------
        tf_list : list, optional
            Trial wavefunctions. If omitted, previously set trials are used.
        band_idxs : list, optional
            Indices of energy bands to project (defaults to occupied manifold).
        use_tilde : bool, optional
            If True, project onto the current Bloch-like subspace instead of the original
            energy eigenstates. This can be used to re-wannierize a set of Bloch-like states
            with a different set of trial functions. By default False.

        Returns
        -------
        w_0n : np.ndarray
            Wannier functions in the home unit cell, obtained by inverse FFT of
            :math:`\tilde{\psi}_{n\mathbf{k}}`.

        Notes
        -----
        This routine does **not** perform iterative minimization of
        :math:`\Omega`; it provides a high-quality initial guess via SVD
        alignment.

        References
        ----------
        .. [1] Marzari, N., & Vanderbilt, D. Maximally localized generalized
            Wannier functions for composite energy bands. Phys. Rev. B 56, 12847 (1997).
        .. [2] Souza, I., Marzari, N., & Vanderbilt, D. Maximally localized
            Wannier functions for entangled energy bands. Phys. Rev. B 65, 035109 (2001).
        """
        if tf_list is None:
            if self.trial_wfs is None:
                raise ValueError(
                    "Trial wavefunctions must be set before Wannierization."
                )
        else:
            self.set_trial_wfs(tf_list)

        twfs = self.trial_wfs

        if use_tilde:
            # projecting back onto tilde states
            if band_idxs is None:  # assume we are projecting onto all tilde states
                band_idxs = list(range(self.tilde_states.nstates))

            psi_til = self.tilde_states.states(flatten_spin_axis=True, return_psi=True)[
                1
            ]
            psi_til_til = self._single_shot_project(psi_til, twfs, state_idx=band_idxs)
            self.set_tilde_states(
                psi_til_til, is_cell_periodic=False, is_spin_axis_flat=True
            )

        else:
            # projecting onto Bloch energy eigenstates
            if band_idxs is None:  # assume we are Wannierizing occupied bands
                n_occ = int(self.bloch_states.nstates / 2)  # assuming half filled
                band_idxs = list(range(0, n_occ))

            psi_nk = self.bloch_states.states(flatten_spin_axis=True, return_psi=True)[
                1
            ]

            # shape: (*nks, states, orbs*n_spin])
            psi_tilde = self._single_shot_project(psi_nk, twfs, state_idx=band_idxs)
            self.set_tilde_states(
                psi_tilde, is_cell_periodic=False, is_spin_axis_flat=True
            )

    def _spread_recip(self, decomp=False):
        r"""Compute quadratic spreads and their MV97 decomposition on a discrete k-shell.

        Computes per-band spreads and (optionally) the decomposition
        :math:`(\Omega_I,\Omega_{\mathrm{D}},\Omega_{\mathrm{OD}})` using discrete overlaps
        on the first nearest-neighbor k-shell.

        Parameters
        ----------
        decomp : bool, optional
            If True, also return the components :math:`\Omega_I`, :math:`\Omega_{\mathrm{D}}`,
            and :math:`\Omega_{\mathrm{OD}}`.

        Returns
        -------
        If ``decomp=False``:
            spread_n, r_n, rsq_n
        If ``decomp=True``:
            [spread_n, Omega_I, Omega_D, Omega_OD], r_n, rsq_n

        Notes
        -----
        The implementation currently uses a **single k-shell** (``n_shell=1``) and assumes a
        **uniform Monkhorst–Pack mesh**.
        """
        M = self.tilde_states.Mmn
        w_b, k_shell, _ = self.lattice.k_shell_weights(self.mesh.shape_k, n_shell=1)
        w_b, k_shell = w_b[0], k_shell[0]  # Assume only one shell for now

        n_states = self.tilde_states.nstates
        nks = self.nks
        k_axes = tuple(self.mesh.k_axis_indices)
        Nk = np.prod(nks)

        diag_M = np.diagonal(M, axis1=-1, axis2=-2)
        log_diag_M_imag = np.log(diag_M).imag
        abs_diag_M_sq = abs(diag_M) ** 2

        r_n = -(1 / Nk) * w_b * np.sum(log_diag_M_imag, axis=k_axes).T @ k_shell
        rsq_n = (
            (1 / Nk)
            * w_b
            * np.sum(
                (1 - abs_diag_M_sq + log_diag_M_imag**2), axis=k_axes + tuple([-2])
            )
        )
        spread_n = rsq_n - np.array(
            [np.vdot(r_n[n, :], r_n[n, :]) for n in range(r_n.shape[0])]
        )

        if decomp:
            Omega_i = w_b * n_states * k_shell.shape[0] - (1 / Nk) * w_b * np.sum(
                abs(M) ** 2
            )

            Omega_d = (
                (1 / Nk) * w_b * (np.sum((-log_diag_M_imag - k_shell @ r_n.T) ** 2))
            )

            Omega_od = (1 / Nk) * w_b * (+np.sum(abs(M) ** 2) - np.sum(abs_diag_M_sq))
            return [spread_n, Omega_i, Omega_d, Omega_od], r_n, rsq_n

        else:
            return spread_n, r_n, rsq_n

    def _get_omega_til(self, Mmn, wb, k_shell):
        nks = self.nks
        Nk = np.prod(nks)
        k_axes = tuple(self.mesh.k_axis_indices)

        diag_M = np.diagonal(Mmn, axis1=-1, axis2=-2)
        log_diag_M_imag = np.log(diag_M).imag
        abs_diag_M_sq = abs(diag_M) ** 2

        r_n = -(1 / Nk) * wb * np.sum(log_diag_M_imag, axis=k_axes).T @ k_shell

        Omega_tilde = (
            (1 / Nk)
            * wb
            * (
                np.sum((-log_diag_M_imag - k_shell @ r_n.T) ** 2)
                + np.sum(abs(Mmn) ** 2)
                - np.sum(abs_diag_M_sq)
            )
        )
        return Omega_tilde

    def _get_omega_d(self, Mmn, wb, k_shell):
        nks = self.nks
        Nk = np.prod(nks)
        k_axes = tuple(self.mesh.k_axis_indices)

        diag_M = np.diagonal(Mmn, axis1=-1, axis2=-2)
        log_diag_M_imag = np.log(diag_M).imag
        r_n = -(1 / Nk) * wb * np.sum(log_diag_M_imag, axis=k_axes).T @ k_shell

        Omega_d = (1 / Nk) * wb * (np.sum((-log_diag_M_imag - k_shell @ r_n.T) ** 2))
        return Omega_d

    def _get_omega_od(self, Mmn, wb):
        Nk = np.prod(self.nks)
        diag_M = np.diagonal(Mmn, axis1=-1, axis2=-2)
        abs_diag_M_sq = abs(diag_M) ** 2

        Omega_od = (1 / Nk) * wb * (+np.sum(abs(Mmn) ** 2) - np.sum(abs_diag_M_sq))
        return Omega_od

    def _get_omega_i(self, Mmn, wb, k_shell):
        Nk = np.prod(self.tilde_states.mesh.shape_k)
        n_states = self.tilde_states.nstates
        Omega_i = wb * n_states * k_shell.shape[0] - (1 / Nk) * wb * np.sum(
            abs(Mmn) ** 2
        )
        return Omega_i

    def _get_omega_i_k(self):
        r"""Calculate the gauge-independent quadratic spread for the Wannier functions.

        This function computes the quadratic spread :math:`\Omega_I`
        of the Wannier functions as a function of `k`. This is related to the
        real part of the quantum metric.
        """
        P = self.tilde_states.projectors()
        _, Q_nbr = self.tilde_states._nbr_projectors(return_Q=True)

        nks = self.nks
        Nk = np.prod(nks)
        w_b, _, idx_shell = self.lattice.k_shell_weights(self.mesh.shape_k, n_shell=1)
        num_nnbrs = idx_shell[0].shape[0]

        T_kb = np.zeros((*nks, num_nnbrs), dtype=complex)
        for nbr_idx in range(num_nnbrs):  # nearest neighbors
            T_kb[..., nbr_idx] = np.trace(
                P[..., :, :] @ Q_nbr[..., nbr_idx, :, :], axis1=-1, axis2=-2
            )

        return (1 / Nk) * w_b[0] * np.sum(T_kb, axis=-1)

    ####### Maximally Localized WF #######

    def _optimal_subspace(
        self,
        n_wfs=None,
        inner_window="occupied",
        outer_window="all",
        iter_num=1000,
        verbose=True,
        tol=1e-10,
        beta=1,
        tf_speedup=False,
    ):
        # useful constants
        nks = self.nks
        Nk = np.prod(nks)
        n_orb = self.bloch_states.norb
        n_states = self.bloch_states.nstates
        n_occ = int(n_states / 2)

        # eigenenergies and eigenstates for inner/outer window
        energies = self.bloch_states.energies
        u_nk = self.bloch_states.states(flatten_spin_axis=True)

        # number of states in target manifold
        if n_wfs is None:
            n_wfs = self.tilde_states.nstates

        ########### Setting energy windows ############

        #### outer window ####
        if isinstance(outer_window, str):
            if outer_window.lower() != "occupied" and outer_window.lower() != "all":
                raise ValueError(
                    "If outer_window is a string, it must be 'occupied' or 'all'."
                )

            outer_window_type = "bands"

            if outer_window.lower() == "all":
                outer_band_idxs = list(range(n_states))
                outer_energy_range = [np.min(energies), np.max(energies)]
            elif outer_window.lower() == "occupied":
                outer_band_idxs = list(range(n_occ))
                outer_band_energies = energies[..., outer_band_idxs]
                outer_energy_range = [
                    np.min(outer_band_energies),
                    np.max(outer_band_energies),
                ]

        elif isinstance(outer_window, dict):
            if list(outer_window.keys())[0].lower() not in ["bands", "energy"]:
                raise ValueError(
                    "If outer_window is a dict, it must have keys 'bands' or 'energy'."
                )

            outer_window_type = list(outer_window.keys())[0].lower()

            if outer_window_type == "bands":
                outer_band_idxs = list(outer_window.values())[0]
                outer_band_energies = energies[..., outer_band_idxs]
                outer_energy_range = [
                    np.min(outer_band_energies),
                    np.max(outer_band_energies),
                ]
            elif outer_window_type == "energy":
                outer_energy_range = np.sort(list(outer_window.values())[0])

        elif (
            isinstance(outer_window, (list, tuple))
            and len(outer_window) == 2
            and all(isinstance(x, (int, float, np.floating)) for x in outer_window)
        ):
            outer_window_type = "energy"
            outer_energy_range = [float(outer_window[0]), float(outer_window[1])]

        ######## inner window ########
        if inner_window is None:
            N_inner = 0
            inner_window_type = outer_window_type
            inner_band_idxs = None
            # make inner energies such that no states are found inside
            inner_energies = [np.inf, -np.inf]

        elif isinstance(inner_window, str):
            if inner_window.lower() != "occupied":
                raise ValueError("If inner_window is a string, it must be 'occupied'.")

            inner_window_type = "bands"
            inner_band_idxs = list(range(n_occ))
            inner_band_energies = energies[..., inner_band_idxs]
            inner_energies = [np.min(inner_band_energies), np.max(inner_band_energies)]

        elif isinstance(inner_window, dict):
            if list(inner_window.keys())[0].lower() not in ["bands", "energy"]:
                raise ValueError(
                    "If inner_window is a dict, it must have keys 'bands' or 'energy'."
                )

            inner_window_type = list(inner_window.keys())[0].lower()

            if inner_window_type == "bands":
                inner_band_idxs = list(inner_window.values())[0]
                inner_band_energies = energies[..., inner_band_idxs]
                inner_energies = [
                    np.min(inner_band_energies),
                    np.max(inner_band_energies),
                ]

            elif inner_window_type == "energy":
                inner_energies = np.sort(list(inner_window.values())[0])

        elif (
            isinstance(inner_window, (list, tuple))
            and len(inner_window) == 2
            and all(isinstance(x, (int, float, np.floating)) for x in inner_window)
        ):
            inner_window_type = "energy"
            inner_energies = [float(inner_window[0]), float(inner_window[1])]

        if inner_window_type == outer_window_type == "bands":
            logger.debug(
                "Both inner and outer windows specified via band indices. "
                "Using the faster optimal_subspace_bands method instead."
            )
            # defer to the faster function
            return self._optimal_subspace_bands(
                n_wfs=n_wfs,
                frozen_bands=inner_band_idxs,
                disentang_bands=outer_band_idxs,
                iter_num=iter_num,
                verbose=verbose,
                tol=tol,
                beta=beta,
                tf_speedup=tf_speedup,
            )

        # create array of nans for masking
        nan = np.empty(u_nk.shape)
        nan.fill(np.nan)

        # mask out states outside outer window
        states_sliced = np.where(
            np.logical_and(
                energies[..., np.newaxis] >= outer_energy_range[0],
                energies[..., np.newaxis] <= outer_energy_range[1],
            ),
            u_nk,
            nan,
        )
        mask_outer = np.isnan(states_sliced)
        masked_outer_states = np.ma.masked_array(states_sliced, mask=mask_outer)

        # mask out states outside inner window
        states_sliced = np.where(
            np.logical_and(
                energies[..., np.newaxis] >= inner_energies[0],
                energies[..., np.newaxis] <= inner_energies[1],
            ),
            u_nk,
            nan,
        )
        mask_inner = np.isnan(states_sliced)
        masked_inner_states = np.ma.masked_array(states_sliced, mask=mask_inner)

        # minimization manifold
        if inner_window is not None:
            # states in outer manifold and outside inner manifold
            min_mask = ~np.logical_and(~mask_outer, mask_inner)
            min_states = np.ma.masked_array(u_nk, mask=min_mask)
            min_states = np.ma.filled(min_states, fill_value=0)

            N_inner = (~masked_inner_states.mask).sum(axis=(-1, -2)) // n_orb
            if np.any(N_inner > n_wfs):
                mesh = self.mesh
                bad_kpts = np.where(N_inner > n_wfs)
                bad_kpts = [mesh.grid[idx] for idx in zip(*bad_kpts)]
                raise ValueError(
                    f"Number of states in inner window exceeds n_wfs at k-points {bad_kpts}."
                )

            num_keep = n_wfs - N_inner  # matrix of integers

            keep_mask = (
                np.arange(min_states.shape[-2])
                >= (num_keep[:, :, np.newaxis, np.newaxis])
            )
            keep_mask = keep_mask.repeat(min_states.shape[-2], axis=-2)
            keep_mask = np.swapaxes(keep_mask, axis1=-1, axis2=-2)
        else:
            min_states = np.ma.filled(masked_outer_states, fill_value=0)

            # keep all the states from minimization manifold
            num_keep = np.full(min_states.shape, n_wfs)
            keep_mask = np.arange(min_states.shape[-2]) >= num_keep
            keep_mask = np.swapaxes(keep_mask, axis1=-1, axis2=-2)

        # Assumes only one shell for now
        w_b, _, _ = self.lattice.k_shell_weights(self.mesh.shape_k, n_shell=1)
        w_b = w_b[0]  # Assume only one shell for now

        # Projector of initial tilde subspace at each k-point
        init_states = self.tilde_states
        P = init_states.projectors(return_Q=False)
        P_nbr, Q_nbr = init_states._nbr_projectors(return_Q=True)
        T_kb = np.einsum("...ij, ...kji -> ...k", P, Q_nbr)

        omega_I_prev = (1 / Nk) * w_b * np.sum(T_kb)
        logger.info(f"Initial Omega_I: {omega_I_prev.real}")
        if verbose:
            print(f"Initial Omega_I: {omega_I_prev.real}")

        P_min = np.copy(P)  # for start of iteration
        P_nbr_min = np.copy(P_nbr)  # for start of iteration
        Q_nbr_min = np.copy(Q_nbr)  # for start of iteration

        if tf_speedup:
            try:
                import tensorflow as tf
            except ImportError:
                raise ImportError(
                    "TensorFlow must be installed to use tf_speedup option."
                )

        #### Start of minimization iteration ####
        for i in range(iter_num):
            P_avg = np.sum(w_b * P_nbr_min, axis=-3)
            Z = min_states.conj() @ P_avg @ np.transpose(min_states, axes=(0, 1, 3, 2))

            if tf_speedup:
                Z_tf = tf.convert_to_tensor(Z, dtype=tf.complex64)
                eigvals, eigvecs = tf.linalg.eigh(Z_tf)  # [..., val, idx]
                eigvals = eigvals.numpy()
                eigvecs = eigvecs.numpy()
            else:
                eigvals, eigvecs = np.linalg.eigh(Z)  # [..., val, idx]

            eigvecs = np.swapaxes(eigvecs, axis1=-1, axis2=-2)  # [..., idx, val]

            # eigvals = 0 correspond to states outside the minimization manifold. Mask these out.
            zero_mask = eigvals.round(10) == 0
            non_zero_eigvals = np.ma.masked_array(eigvals, mask=zero_mask)
            non_zero_eigvecs = np.ma.masked_array(
                eigvecs,
                mask=np.repeat(
                    zero_mask[..., np.newaxis], repeats=eigvals.shape[-1], axis=-1
                ),
            )

            # sort eigvals and eigvecs by eigenvalues in descending order excluding eigvals=0
            sorted_eigvals_idxs = np.argsort(-non_zero_eigvals, axis=-1)
            # sorted_eigvals = np.take_along_axis(non_zero_eigvals, sorted_eigvals_idxs, axis=-1)
            sorted_eigvecs = np.take_along_axis(
                non_zero_eigvecs, sorted_eigvals_idxs[..., np.newaxis], axis=-2
            )
            sorted_eigvecs = np.ma.filled(sorted_eigvecs, fill_value=0)

            states_min = np.einsum("...ji, ...ik->...jk", sorted_eigvecs, min_states)
            keep_states_ma = np.ma.masked_array(states_min, mask=keep_mask)

            # need to concatenate with frozen states
            if inner_window is not None:
                min_states_ma = np.ma.concatenate(
                    (keep_states_ma, masked_inner_states), axis=-2
                )
                min_states_sliced = min_states_ma[np.where(~min_states_ma.mask)]
                min_states_sliced = min_states_sliced.reshape((*nks, n_wfs, n_orb))
                states_min = np.array(min_states_sliced)
            else:
                min_states_sliced = keep_states_ma[np.where(~keep_states_ma.mask)]
                min_states_sliced = min_states_sliced.reshape((*nks, n_wfs, n_orb))
                states_min = np.array(min_states_sliced)

            # update projectors
            min_wfa = WFArray(
                self.lattice,
                self.mesh,
                nstates=states_min.shape[-2],
                spinful=self.bloch_states.spinful,
            )
            min_wfa.set_states(
                states_min, is_cell_periodic=True, is_spin_axis_flat=True
            )
            P_new = min_wfa.projectors()
            P_nbr_new = min_wfa._nbr_projectors(return_Q=False)

            if beta != 1:
                # for next iteration
                P_min = beta * P_new + (1 - beta) * P_min
                P_nbr_min = beta * P_nbr_new + (1 - beta) * P_nbr_min
            else:
                # for next iteration
                P_min = P_new
                P_nbr_min = P_nbr_new

            Q_nbr_min = np.eye(P_nbr_min.shape[-1]) - P_nbr_min
            T_kb = np.einsum("...ij, ...kji -> ...k", P_min, Q_nbr_min)
            omega_I_new = (1 / Nk) * w_b * np.sum(T_kb)
            delta = omega_I_new - omega_I_prev
            logger.info(
                f"iter {i:4d} | Ω_I = {omega_I_new.real:12.9e} | ΔΩ = {delta.real:10.5e}"
            )

            if verbose:
                print(
                    f"iter {i:4d} | Ω_I = {omega_I_new.real:12.9e} | ΔΩ = {delta.real:10.5e}"
                )

            if abs(delta) <= tol:
                logger.info(
                    f"Converged within tolerance in {i} iterations. Breaking the loop."
                )
                if verbose:
                    print(
                        f"Converged within tolerance in {i} iterations. Breaking the loop."
                    )
                break

            if omega_I_new > omega_I_prev:
                beta = max(beta - 0.01, 0)
                logger.warning(f"Warning: Ω_I is increasing. Reducing beta to {beta}.")
                if verbose:
                    print(f"Warning: Ω_I is increasing. Reducing beta to {beta}.")

            omega_I_prev = omega_I_new

        return states_min

    def _optimal_subspace_bands(
        self,
        n_wfs: int | None = None,
        frozen_bands: list | None = None,
        disentang_bands: list | str = "occupied",
        iter_num: int = 1000,
        tol: float = 1e-10,
        beta: float = 1,
        verbose: bool = True,
        tf_speedup: bool = False,
    ):
        r"""Obtain the subspace that minimizes the gauge-independent spread.

        This function utilizes the 'disentanglement' technique to find the subspaces
        throughout the BZ that minimizes the gauge-independent spread.

        Parameters
        ----------
        n_wfs : int | None
            Number of states in the optimal subspace. If ``None``, the number
            of trial wavefunctions is used.
        frozen_bands : list | None, optional
            List of band indices defining the 'frozen window', specifying
            the states totally included within the optimized subspace.
            Defaults to `None`, in which case no bands are frozen.
        disentang_bands : list | str, optional
            List of band indices defining 'disentanglement window' where
            states are borrowed in order to minimize the gauge independent spread.
            If "occupied", all occupied bands are disentangled. Defaults to "occupied".
        iter_num : int, optional
            Maximum number of optimization iterations. Defaults to 100.
        tol : float, optional
            Convergence tolerance for the optimization. Defaults to 1e-10.
        beta : float, optional
            Mixing parameter for the optimization. If 1, the current step is taken fully.
            Lower values result in a percentage ``beta`` of the previous step being mixed into
            the result. Defaults to 1.
        verbose : bool, optional
            If True, print detailed information during optimization.
        tf_speedup : bool, optional
            If True, uses the ``tensorflow`` package for faster linear algebra operations.

        Returns
        -------
        states_min : np.ndarray
            The states spanning the optimized subspace that minimizes the gauge-independent spread.
        """
        nks = self.nks
        Nk = np.prod(nks)
        n_orb = self.lattice.norb
        n_occ = int(n_orb / 2)

        # Assumes only one shell for now
        w_b, _, _ = self.lattice.k_shell_weights(self.mesh.shape_k, n_shell=1)
        w_b = w_b[0]  # Assume only one shell for now

        # initial subspace
        u_nk = self.bloch_states.states(flatten_spin_axis=True)
        # u_wfs_til = init_states.states(flatten_spin_axis=True)

        if n_wfs is None:
            # assume number of states in the subspace is number of tilde states
            n_wfs = self.tilde_states.nstates

        if isinstance(disentang_bands, str) and disentang_bands == "occupied":
            disentang_bands = list(range(n_occ))

        # Projector of initial tilde subspace at each k-point
        if frozen_bands is None:
            N_inner = 0
            init_states = self.tilde_states

            # manifold from which we borrow states to minimize omega_i
            comp_states = u_nk.take(disentang_bands, axis=-2)
        else:
            N_inner = len(frozen_bands)
            inner_states = u_nk.take(frozen_bands, axis=-2)
            P_inner = np.swapaxes(inner_states, -1, -2) @ inner_states.conj()
            Q_inner = np.eye(P_inner.shape[-1]) - P_inner

            P_tilde = self.tilde_states.projectors()

            # chosing initial subspace as highest eigenvalues
            MinMat = Q_inner @ P_tilde @ Q_inner
            _, eigvecs = np.linalg.eigh(MinMat)
            eigvecs = np.swapaxes(eigvecs, -1, -2)

            init_evecs = eigvecs[..., -(n_wfs - N_inner) :, :]
            init_states = WFArray(
                self.lattice,
                self.mesh,
                nstates=init_evecs.shape[-2],
                spinful=self.bloch_states.spinful,
            )
            init_states.set_states(
                init_evecs, is_cell_periodic=False, is_spin_axis_flat=True
            )

            comp_bands = list(np.setdiff1d(disentang_bands, frozen_bands))
            comp_states = u_nk.take(comp_bands, axis=-2)

        P = init_states.projectors(return_Q=False)
        P_nbr, Q_nbr = init_states._nbr_projectors(return_Q=True)

        T_kb = np.einsum("...ij, ...kji -> ...k", P, Q_nbr)
        omega_I_prev = (1 / Nk) * w_b * np.sum(T_kb)
        logger.info(f"Initial Omega_I: {omega_I_prev.real}")
        if verbose:
            print(f"Initial Omega_I: {omega_I_prev.real}")

        P_min = np.copy(P)  # for start of iteration
        P_nbr_min = np.copy(P_nbr)  # for start of iteration

        if tf_speedup:
            try:
                import tensorflow as tf
            except ImportError:
                raise ImportError(
                    "TensorFlow must be installed to use tf_speedup option."
                )

        for i in range(iter_num):
            # states spanning optimal subspace minimizing gauge invariant spread
            P_avg = w_b * np.sum(P_nbr_min, axis=-3)
            Z = comp_states.conj() @ P_avg @ np.swapaxes(comp_states, -1, -2)

            if tf_speedup:
                Z_tf = tf.convert_to_tensor(Z, dtype=tf.complex64)
                _, eigvecs_tf = tf.linalg.eigh(Z_tf)
                eigvecs = eigvecs_tf.numpy()
            else:
                _, eigvecs = np.linalg.eigh(Z)  # [val, idx]

            evecs_keep = eigvecs[..., -(n_wfs - N_inner) :]
            comp_min = np.swapaxes(evecs_keep, -1, -2) @ comp_states

            if frozen_bands is not None:
                states_min = np.concatenate((inner_states, comp_min), axis=-2)
            else:
                states_min = comp_min

            min_wfa = WFArray(
                self.lattice,
                self.mesh,
                nstates=states_min.shape[-2],
                spinful=self.bloch_states.spinful,
            )
            min_wfa.set_states(
                states_min, is_cell_periodic=True, is_spin_axis_flat=True
            )
            P_new = min_wfa.projectors()
            P_nbr_new = min_wfa._nbr_projectors(return_Q=False)

            if beta != 1:
                # for next iteration
                P_min = beta * P_new + (1 - beta) * P_min
                P_nbr_min = beta * P_nbr_new + (1 - beta) * P_nbr_min
            else:
                # for next iteration
                P_min = P_new
                P_nbr_min = P_nbr_new

            Q_nbr_min = np.eye(P_nbr_min.shape[-1]) - P_nbr_min
            T_kb = np.einsum("...ij, ...kji -> ...k", P_min, Q_nbr_min)
            omega_I_new = (1 / Nk) * w_b * np.sum(T_kb)
            delta = omega_I_new - omega_I_prev
            logger.info(
                f"iter {i:4d} | Ω_I = {omega_I_new.real:12.9e} | ΔΩ = {delta.real:10.5e}"
            )

            if verbose:
                print(
                    f"iter {i:4d} | Ω_I = {omega_I_new.real:12.9e} | ΔΩ = {delta.real:10.5e}"
                )

            if abs(delta) <= tol:
                logger.info(
                    f"Converged within tolerance in {i} iterations. Breaking the loop."
                )
                if verbose:
                    print(
                        f"Converged within tolerance in {i} iterations. Breaking the loop."
                    )
                break

            if omega_I_new > omega_I_prev:
                beta = max(beta - 0.01, 0)
                logger.warning(
                    f"Warning: Ω_I is increasing. Reducing beta from {beta + 0.01} to {beta}."
                )
                if verbose:
                    print(f"Warning: Ω_I is increasing. Reducing beta to {beta}.")

            omega_I_prev = omega_I_new

        return states_min

    def _max_loc_unitary(
        self, alpha=1 / 2, iter_num=100, verbose=False, tol=1e-10, grad_min=1e-3
    ):
        r"""
        Finds the unitary that minimizes the gauge dependent part of the spread.

        Parameters
        ----------
        eps : float
            Step size for gradient descent
        iter_num : int
            Number of iterations
        verbose : bool
            Whether to print the spread at each iteration
        tol : float
            If difference of spread is lower that tol for consecutive iterations,
            the loop breaks

        Returns
        ---------
        U : np.ndarray
            The unitary matrix that rotates the tilde states to minimize
            the gauge-dependent spread.
        """
        M = self.tilde_states.Mmn
        w_b, k_shell, idx_shell = self.lattice.k_shell_weights(
            self.mesh.shape_k, n_shell=1
        )
        # Assumes only one shell for now
        w_b, k_shell, idx_shell = w_b[0], k_shell[0], idx_shell[0]
        k_axes = tuple(self.mesh.k_axis_indices)
        nks = self.nks
        shape_mesh = self.mesh.shape_axes
        Nk = np.prod(nks)
        num_state = self.tilde_states.nstates

        U = np.zeros(
            (*shape_mesh, num_state, num_state), dtype=complex
        )  # unitary transformation
        U[...] = np.eye(num_state, dtype=complex)  # initialize as identity
        M0 = np.copy(M)  # initial overlap matrix
        M = np.copy(M)  # new overlap matrix

        # initializing
        omega_tilde_prev = self._get_omega_til(M, w_b, k_shell)
        grad_mag_prev = 0
        for i in range(iter_num):
            r_n = (
                -(1 / Nk)
                * w_b
                * np.sum(
                    log_diag_M_imag := np.log(np.diagonal(M, axis1=-1, axis2=-2)).imag,
                    axis=k_axes,
                ).T
                @ k_shell
            )
            q = log_diag_M_imag + (k_shell @ r_n.T)
            R = np.multiply(
                M, np.diagonal(M, axis1=-1, axis2=-2)[..., np.newaxis, :].conj()
            )
            T = np.multiply(
                np.divide(M, np.diagonal(M, axis1=-1, axis2=-2)[..., np.newaxis, :]),
                q[..., np.newaxis, :],
            )
            A_R = (R - np.swapaxes(R, axis1=-1, axis2=-2).conj()) / 2
            S_T = (T + np.swapaxes(T, axis1=-1, axis2=-2).conj()) / (2j)
            G = 4 * w_b * np.sum(A_R - S_T, axis=-3)
            U = np.einsum(
                "...ij, ...jk -> ...ik",
                U,
                mat_exp((alpha / (4 * k_shell.shape[0] * w_b)) * G),
            )

            for idx, idx_vec in enumerate(idx_shell):
                M[..., idx, :, :] = (
                    np.swapaxes(U, -1, -2).conj()
                    @ M0[..., idx, :, :]
                    @ np.roll(
                        U, shift=tuple(-idx_vec), axis=tuple(self.mesh.k_axis_indices)
                    )
                )

            grad_mag = np.linalg.norm(np.sum(G, axis=tuple(self.mesh.k_axis_indices)))
            omega_tilde_new = self._get_omega_til(M, w_b, k_shell)
            delta = omega_tilde_new - omega_tilde_prev
            logger.info(
                f"iter {i:4d} | Ω_tilde = {omega_tilde_new.real:12.9e} | ΔΩ = {delta.real:12.5e} | ‖∇‖ = {grad_mag:10.5e}"
            )

            if verbose:
                print(
                    f"iter {i:4d} | Ω_tilde = {omega_tilde_new.real:12.9e} | ΔΩ = {delta.real:12.5e} | ‖∇‖ = {grad_mag:10.5e}"
                )

            # Check for convergence
            if abs(grad_mag) <= grad_min and abs(delta) <= tol:
                logger.info(
                    f"Converged within tolerance in {i} iterations. Breaking the loop."
                )
                if verbose:
                    print(
                        f"Converged within tolerance in {i} iterations. Breaking the loop."
                    )
                break

            if grad_mag_prev < grad_mag and i != 0:
                logger.warning("Warning: Gradient increasing.")
                if verbose:
                    print("Warning: Gradient increasing.")
                # Reduce step size to try and stabilize
                # eps *= 0.9

            grad_mag_prev = grad_mag
            omega_tilde_prev = omega_tilde_new

        return U

    def disentangle(
        self,
        n_wfs: int | None = None,
        outer_window: str | tuple | list | dict = "all",
        frozen_window: str | tuple | list | dict | None = None,
        max_iter: int = 1000,
        tol: float = 1e-10,
        mix: float = 1.0,
        tf_speedup: bool = False,
        verbose: bool = True,
    ):
        r"""Disentanglement of a subspace that minimizes gauge-independent spread.

        This procedure implements the Souza–Marzari–Vanderbilt (SMV)
        disentanglement algorithm [1]_. The goal is to
        select an ``n_wfs``-dimensional optimal subspace from a larger set of
        Bloch eigenstates in a specified "``outer window``," such that the
        gauge-independent part of the Wannier spread :math:`\Omega_I` is
        minimized. The procedure is iterative, updating the subspace at each
        k-point until self-consistency is achieved.

        Parameters
        ----------
        n_wfs : int | None
            Number of states in the optimal subspace. If ``None``, the number
            of trial wavefunctions is used.
        outer_window : str | tuple | list | dict, optional
            Defines the "disentanglement window," i.e. the set of candidate
            states from which the optimal subspace is chosen. States outside
            this window are ignored. Options:

            - ``"occupied"``: All states below the Fermi level.
            - ``"all"``: All available states.
            - ``(Emin, Emax)``: Energy range in eV.
            - ``{"bands": [i1, i2, ...]}``: Explicit band indices.
            - ``{"energy": (Emin, Emax)}``: Energy window.

            Defaults to ``"all"``.
        frozen_window : str | tuple | list | dict | None, optional
            Defines the "frozen window," i.e. states that must be exactly
            included in the subspace. This ensures that, for example, the
            occupied manifold is preserved while disentangling higher states.
            Options follow the same conventions as ``outer_window``. If
            ``None``, no states are frozen. Defaults to ``None``.
        max_iter : int, optional
            Maximum number of optimization iterations. Defaults to 1000.
        tol : float, optional
            Convergence tolerance for the optimization. Defaults to 1e-10.
        mix : float, optional
            Mixing parameter for iterative updates. ``mix=1`` uses the new step
            fully, while smaller values blend the new and old projectors.
            Defaults to 1.
        tf_speedup : bool, optional
            If True, use the ``tensorflow`` package for accelerated linear
            algebra. Defaults to False.
        verbose : bool, optional
            If True, print detailed iteration to the logger. Defaults to True.

        Notes
        -----
        - The disentanglement algorithm iteratively refines the projectors
          onto the optimal subspace by solving the eigenvalue problem

          .. math::
              \left[\sum_{\mathbf{b}} w_b \,\hat{\mathcal{P}}_{\mathbf{k}+\mathbf{b}}^{(i)}\right]
              | u_{m\mathbf{k}}^{(i)} \rangle
              = \lambda_{m\mathbf{k}}^{(i)} | u_{m\mathbf{k}}^{(i)} \rangle,

          where :math:`\hat{\mathcal{P}}_{\mathbf{k}+\mathbf{b}}^{(i)}` is the
          projector from the previous iteration. The states with the largest
          :math:`n_\text{wfs}` eigenvalues are selected to form the new subspace.

        - The role of the **outer window** is to provide flexibility: states
          above or below the frozen region can be borrowed to reduce
          :math:`\Omega_I`. The **frozen window** ensures that crucial states
          (e.g. fully occupied bands) are exactly included regardless of the
          optimization outcome.

        - After convergence, the ``.tilde_states`` attribute stores the
          disentangled wavefunctions spanning the optimized subspace.

        References
        ----------
        .. [1] Souza, I., Marzari, N., & Vanderbilt, D. Maximally localized
            Wannier functions for entangled energy bands. Phys. Rev. B 65, 035109 (2001).
        """
        # if we haven't done single-shot projection yet (set tilde states)
        if not hasattr(self.tilde_states, "_u_nk"):
            # check if we have trial wavefunctions
            if not hasattr(self, "_trial_wfs"):
                # we use energy eigenstates tilde states
                self.set_tilde_states(
                    self.bloch_states.states(flatten_spin_axis=True),
                    is_cell_periodic=True,
                )
            else:
                # we initialize tilde states with previous trial wavefunctions
                n_occ = int(self.bloch_states.nstates / 2)  # assuming half filled
                band_idxs = list(range(0, n_occ))  # project onto occ manifold
                psi_nk = self.bloch_states.states(
                    flatten_spin_axis=True, return_psi=True
                )[1]
                self._single_shot_project(psi_nk, self._twfs, state_idx=band_idxs)

        # Minimizing Omega_I via disentanglement
        util_min = self._optimal_subspace(
            n_wfs=n_wfs,
            inner_window=frozen_window,
            outer_window=outer_window,
            iter_num=max_iter,
            verbose=verbose,
            beta=mix,
            tol=tol,
            tf_speedup=tf_speedup,
        )

        self.set_tilde_states(util_min, is_cell_periodic=True, is_spin_axis_flat=True)

    def maxloc(
        self, alpha=1 / 2, max_iter=1000, tol=1e-5, grad_min=1e-3, verbose=False
    ):
        r"""Unitary transformation to minimize the gauge-dependent spread.

        This procedure implements the Marzari-Vanderbilt maximal localization
        algorithm [1]_. Given a (disentangled) subspace
        (``.tilde_states``), it finds the optimal unitary transformation that
        minimizes the gauge-dependent part of the Wannier spread
        :math:`\widetilde{\Omega}`. The algorithm proceeds iteratively, applying
        gradient-descent updates to the unitary matrices at each k-point until
        convergence.

        Parameters
        ----------
        alpha : float, optional
            Step size for gradient descent. Typical values are between 0 and 1.
        max_iter : int, optional
            Maximum number of iterations for the optimization. Default is 1000.
        tol : float, optional
            Convergence tolerance for the change in spread. Default is 1e-5.
        grad_min : float, optional
            Minimum gradient magnitude for convergence. Default is 1e-3.
        verbose : bool, optional
            If True, print progress messages. Default is False.

        Notes
        -----
        - The gauge-dependent contribution to the total Wannier spread is

          .. math::
              \widetilde{\Omega} = \Omega - \Omega_I,

          where :math:`\Omega` is the total quadratic spread functional and
          :math:`\Omega_I` is the gauge-invariant part obtained during the
          disentanglement step.

        - The minimization is achieved by unitary rotations of the form

          .. math::
              | u_{m\mathbf{k}}^{\text{new}} \rangle
              = \sum_{n} U_{nm}(\mathbf{k}) \,
                | u_{n\mathbf{k}}^{\text{old}} \rangle,

          with :math:`U(\mathbf{k}) \in U(N)`, where :math:`N` is the dimension
          of the disentangled subspace at each k-point.

        - The gradient of :math:`\widetilde{\Omega}` with respect to an infinitesimal
          anti-Hermitian generator :math:`A(\mathbf{k})` is computed, and the unitary
          matrices are updated via

          .. math::
              U(\mathbf{k}) \;\to\; \exp[-\epsilon A(\mathbf{k})] \, U(\mathbf{k}),

          where :math:`\epsilon = \alpha/4 \sum_{\mathbf{b}}w_b` is the step size (given ``alpha``).

        - Iteration proceeds until the gradient norm falls below ``grad_min`` and the change in spread is smaller
          than ``tol``, or the maximum number of iterations is reached.

        References
        ----------
        .. [1] Marzari, N., & Vanderbilt, D. Maximally localized generalized
            Wannier functions for composite energy bands. Phys. Rev. B 56, 12847 (1997).
        """

        U = self._max_loc_unitary(
            alpha=alpha, iter_num=max_iter, verbose=verbose, tol=tol, grad_min=grad_min
        )

        u_tilde_wfs = self.tilde_states.states(flatten_spin_axis=True)
        util_max_loc = np.einsum("...ji, ...jm -> ...im", U, u_tilde_wfs)

        self.set_tilde_states(
            util_max_loc, is_cell_periodic=True, is_spin_axis_flat=True
        )

    def min_spread(
        self,
        outer_window="all",
        inner_window=None,
        twfs_2=None,
        n_wfs=None,
        max_iter=1000,
        max_iter_dis=1000,
        alpha=1 / 2,
        tol_max_loc=1e-5,
        tol_dis=1e-10,
        grad_min=1e-3,
        mix=1,
        verbose=False,
    ):
        r"""Find the maximally localized Wannier functions using the projection method.

        This method performs three steps:

        1. Calls :meth:`disentangle` to find the optimal subspace that minimizes the gauge-independent spread.

        2. Applies a second projection using ``twfs_2`` if provided, or the original trial wavefunctions otherwise,
           to refine the states within the optimal subspace. This step ensures that the states are well-aligned
           with the desired chemical character before localization. It uses the :meth:`project` method
           for this projection.

        3. Calls :meth:`maxloc` to find the unitary transformation that minimizes the gauge-dependent spread,
           resulting in maximally localized Wannier functions.

        Parameters
        ----------
        outer_window : str | tuple | list | dict, optional
            Defines the "disentanglement window," i.e. the set of candidate
            states from which the optimal subspace is chosen. States outside
            this window are ignored. Options:

            - ``"occupied"``: All states below the Fermi level.
            - ``"all"``: All available states.
            - ``(Emin, Emax)``: Energy range in eV.
            - ``{"bands": [i1, i2, ...]}``: Explicit band indices.
            - ``{"energy": (Emin, Emax)}``: Energy window.

            Defaults to ``"all"``.
        inner_window : str | tuple | list | dict, optional
            Defines the "frozen window," i.e. states that must be exactly
            included in the subspace. This ensures that, for example, the
            occupied manifold is preserved while disentangling higher states.
            Options follow the same conventions as ``outer_window``. If
            ``None``, no states are frozen. Defaults to ``None``.
        twf_list_second : list[list[tuple]], optional
            A second set of trial wavefunctions for the projection step after
            disentanglement. If ``None``, the original trial wavefunctions are
            used. Defaults to ``None``.
        n_wfs : int | None, optional
            Number of states in the optimal subspace. If ``None``, the number
            of trial wavefunctions is used. Defaults to ``None``.
        max_iter : int, optional
            Maximum number of iterations for the maximal localization step.
            Default is 1000.
        max_iter_dis : int, optional
            Maximum number of iterations for the disentanglement step.
            Default is 1000.
        alpha : float, optional
            Step size for gradient descent in the maximal localization step.
            Typical values are between 0 and 1. Default is 1/2.
        tol_max_loc : float, optional
            Convergence tolerance for the change in spread in the maximal
            localization step. Default is 1e-5.
        tol_dis : float, optional
            Convergence tolerance for the disentanglement step. Default is 1e-10.
        grad_min : float, optional
            Minimum gradient magnitude for convergence in the maximal
            localization step. Default is 1e-3.
        mix : float, optional
            Mixing parameter for iterative updates in the disentanglement step.
            ``mix=1`` uses the new step fully, while smaller values blend the
            new and old projectors. Defaults to 1.
        verbose : bool, optional
            If True, print detailed iteration information to the logger. Default is False.

        Notes
        -----
        - This method combines disentanglement and maximal localization to
          produce maximally localized Wannier functions from a set of Bloch
          states. It first identifies an optimal subspace, then refines the
          states via projection, and finally minimizes the gauge-dependent spread.
        - The resulting Wannier functions are stored in the ``.tilde_states``
          attribute after the procedure completes.
        """

        ### Subspace selection ###
        self.disentangle(
            outer_window=outer_window,
            inner_window=inner_window,
            n_wfs=n_wfs,
            max_iter=max_iter_dis,
            tol=tol_dis,
            mix=mix,
            verbose=verbose,
        )

        ### Second projection ###
        # if we need a smaller number of twfs b.c. of subspace selec
        if twfs_2 is not None:
            twfs = self._get_trial_wfs(twfs_2)
            psi_til = self.tilde_states.states(flatten_spin_axis=True, return_psi=True)[
                1
            ]
            psi_til_til = self._single_shot_project(
                psi_til,
                twfs,
                state_idx=list(range(self.tilde_states.nstates)),
            )
        # choose same twfs as in subspace selection
        else:
            psi_til = self.tilde_states.states(flatten_spin_axis=True, return_psi=True)[
                1
            ]
            psi_til_til = self._single_shot_project(
                psi_til,
                self.trial_wfs,
                state_idx=list(range(self.tilde_states.nstates)),
            )

        self.set_tilde_states(
            psi_til_til, is_cell_periodic=False, is_spin_axis_flat=True
        )

        ### Finding optimal gauge with maxloc ###
        self.maxloc(
            alpha=alpha,
            iter_num=max_iter,
            tol=tol_max_loc,
            grad_min=grad_min,
            verbose=verbose,
        )

    def interp_bands(
        self, k_nodes, n_interp: int = 20, wan_idxs=None, ret_eigvecs=False
    ):
        r"""Wannier interpolate the band structure along a k-path.

        This method uses the Wannier functions to interpolate the band structure
        along a specified k-path. It constructs a tight-binding Hamiltonian in the
        Wannier basis, diagonalizes it, and Fourier transforms back to k-space
        along the k-path defined by ``k_nodes``.

        Parameters
        ----------
        k_nodes : np.ndarray
            Array of k-points defining the path in reciprocal space.
        n_interp : int, optional
            Number of interpolated k-points between each pair of nodes in ``k_nodes``.
            Defaults to 20.
        wan_idxs : list | np.ndarray | None, optional
            Indices of Wannier functions to include in the interpolation. If None, all Wannier functions
            are used. Defaults to None.
        ret_eigvecs : bool, optional
            If True, return the eigenvectors along with the eigenvalues. Defaults to False.
        """
        u_tilde = self.tilde_states.states(flatten_spin_axis=False)
        if wan_idxs is not None:
            u_tilde = np.take_along_axis(u_tilde, wan_idxs, axis=-2)

        k_mesh = self.mesh.get_k_points()
        k_flat = k_mesh.reshape(-1, k_mesh.shape[-1])
        H_k = self.bloch_states.model.hamiltonian(k_flat)
        H_k = H_k.reshape(k_mesh.shape[:-1] + H_k.shape[1:])
        if self.bloch_states.spinful:
            new_shape = H_k.shape[:-4] + (
                self.bloch_states.nstates,
                self.bloch_states.nstates,
            )
            H_k = H_k.reshape(*new_shape)

        H_rot_k = u_tilde.conj() @ H_k @ np.swapaxes(u_tilde, -1, -2)
        eigvals, eigvecs = np.linalg.eigh(H_rot_k)
        eigvecs = np.einsum("...ij, ...ik->...kj", u_tilde, eigvecs)
        # eigvecs = np.swapaxes(eigvecs, -1, -2)

        nks = self.nks
        idx_grid = np.indices(nks, dtype=int)
        k_idx_arr = idx_grid.reshape(len(nks), -1).T
        Nk = np.prod([nks])

        supercell = list(
            product(
                *[
                    range(-int((nk - nk % 2) / 2), int((nk - nk % 2) / 2) + 1)
                    for nk in nks
                ]
            )
        )

        # Fourier transform to real space
        # H_R = np.zeros((len(supercell), H_rot_k.shape[-1], H_rot_k.shape[-1]), dtype=complex)
        # u_R = np.zeros((len(supercell), u_tilde.shape[-2], u_tilde.shape[-1]), dtype=complex)
        eval_R = np.zeros((len(supercell), eigvals.shape[-1]), dtype=complex)
        evecs_R = np.zeros(
            (len(supercell), eigvecs.shape[-2], eigvecs.shape[-1]), dtype=complex
        )
        for idx, r in enumerate(supercell):
            for k_idx in k_idx_arr:
                R_vec = np.array([*r])
                phase = np.exp(-1j * 2 * np.pi * np.vdot(k_mesh[*k_idx], R_vec))
                # H_R[idx, :, :] += H_rot_k[k_idx] * phase / Nk
                # u_R[idx] += u_tilde[k_idx] * phase / Nk
                eval_R[idx] += eigvals[*k_idx] * phase / Nk
                evecs_R[idx] += eigvecs[*k_idx] * phase / Nk

        # interpolate to arbitrary k
        k_path, _, _ = self.lattice.k_path(k_nodes, nk=n_interp, report=False)

        # H_k_interp = np.zeros((k_path.shape[0], H_R.shape[-1], H_R.shape[-1]), dtype=complex)
        # u_k_interp = np.zeros((k_path.shape[0], u_R.shape[-2], u_R.shape[-1]), dtype=complex)
        eigvals_k_interp = np.zeros((k_path.shape[0], eval_R.shape[-1]), dtype=complex)
        eigvecs_k_interp = np.zeros(
            (k_path.shape[0], evecs_R.shape[-2], evecs_R.shape[-1]), dtype=complex
        )

        for k_idx, k in enumerate(k_path):
            for idx, r in enumerate(supercell):
                R_vec = np.array([*r])
                phase = np.exp(1j * 2 * np.pi * np.vdot(k, R_vec))
                # H_k_interp[k_idx] += H_R[idx] * phase
                # u_k_interp[k_idx] += u_R[idx] * phase
                eigvals_k_interp[k_idx] += eval_R[idx] * phase
                eigvecs_k_interp[k_idx] += evecs_R[idx] * phase

        # eigvals, eigvecs = np.linalg.eigh(H_k_interp)
        # eigvecs = np.einsum('...ij, ...ik -> ...kj', u_k_interp, eigvecs)
        # # normalizing
        # eigvecs /= np.linalg.norm(eigvecs, axis=-1, keepdims=True)
        eigvecs_k_interp /= np.linalg.norm(eigvecs_k_interp, axis=-1, keepdims=True)

        if ret_eigvecs:
            return eigvals_k_interp.real, eigvecs_k_interp
        else:
            return eigvals_k_interp.real

    def _get_sc_centers(self):
        r"""Get the positions of the Wannier function centers in the supercell.
        This method computes the positions of the Wannier function centers
        in the supercell defined by ``self.supercell``. It returns a dictionary
        containing the x and y coordinates of the Wannier function centers for
        all sites and home cell sites.

        Returns
        -------
        positions : dict
            A dictionary with keys 'centers all' and 'centers home', each containing
            sub-dictionaries with keys 'xs' and 'ys' for x-coordinates and y-coordinates,
            respectively.
        """
        lat_vecs = self.lattice.lat_vecs
        centers = self.centers

        # Initialize arrays to store positions and weights
        positions = {
            "centers all": {
                "xs": [[] for _ in range(centers.shape[0])],
                "ys": [[] for _ in range(centers.shape[0])],
            },
            "centers home": {"xs": [], "ys": []},
        }

        for j in range(centers.shape[0]):
            for tx, ty in self.supercell:
                center = centers[j] + tx * lat_vecs[0] + ty * lat_vecs[1]
                positions["centers all"]["xs"][j].append(center[0])
                positions["centers all"]["ys"][j].append(center[1])

                if tx == ty == 0:
                    positions["centers home"]["xs"].append(center[0])
                    positions["centers home"]["ys"].append(center[1])

        # Convert lists to numpy arrays (batch processing for cleanliness)
        for key, data in positions.items():
            for sub_key in data:
                positions[key][sub_key] = np.array(data[sub_key])
        return positions

    def _get_sc_weights(self, wan_idx, special_sites=None):
        r"""Get the positions and weights of the Wannier functions in the supercell.
        This method computes the positions and weights of the Wannier functions
        in the supercell defined by ``self.supercell``. It returns a dictionary
        containing the x and y coordinates, radial distances from the center,
        and weights of the Wannier functions for all sites, home cell sites, and
        optionally special sites.

        Parameters
        ----------
        wan_idx : int
            Index of the Wannier function to analyze.
        special_sites : list | None, optional
            List of orbital indices considered as special sites. If provided,
            the method will also compute positions and weights for these sites.
            Defaults to None.
        Returns
        -------
        positions : dict
            A dictionary with keys 'all', 'home', and optionally 'special',
            each containing sub-dictionaries with keys 'xs', 'ys', 'r', and 'wt'
            for x-coordinates, y-coordinates, radial distances, and weights,
            respectively.
        """
        w0 = self.WFs
        center = self.centers[wan_idx]
        orbs = self.lattice.orb_vecs
        lat_vecs = self.lattice.lat_vecs

        # Initialize arrays to store positions and weights
        positions = {
            "all": {"xs": [], "ys": [], "r": [], "wt": []},
            "home": {"xs": [], "ys": [], "r": [], "wt": []},
        }

        for tx, ty in self.supercell:
            for i, orb in enumerate(orbs):
                # Extract relevant parameters
                wf_value = w0[tx, ty, wan_idx, i]
                wt = np.sum(np.abs(wf_value) ** 2)
                pos = (
                    orb[0] * lat_vecs[0]
                    + tx * lat_vecs[0]
                    + orb[1] * lat_vecs[1]
                    + ty * lat_vecs[1]
                )
                rel_pos = pos - center
                x, y, rad = pos[0], pos[1], np.sqrt(rel_pos[0] ** 2 + rel_pos[1] ** 2)

                # Store values in 'all'
                positions["all"]["xs"].append(x)
                positions["all"]["ys"].append(y)
                positions["all"]["r"].append(rad)
                positions["all"]["wt"].append(wt)

                # Handle special sites if applicable
                if special_sites is not None and i in special_sites:
                    positions["special"]["xs"].append(x)
                    positions["special"]["ys"].append(y)
                    positions["special"]["r"].append(rad)
                    positions["special"]["wt"].append(wt)

                if tx == ty == 0:
                    positions["home"]["xs"].append(x)
                    positions["home"]["ys"].append(y)
                    positions["home"]["r"].append(rad)
                    positions["home"]["wt"].append(wt)

        # Convert lists to numpy arrays (batch processing for cleanliness)
        for key, data in positions.items():
            for sub_key in data:
                positions[key][sub_key] = np.array(data[sub_key])

        return positions

    @copydoc(plot_centers)
    def plot_centers(
        self,
        center_scale=200,
        section_home_cell=True,
        color_home_cell=True,
        translate_centers=False,
        show=False,
        legend=False,
        pmx=4,
        pmy=4,
        center_color="r",
        center_marker="*",
        lat_home_color="b",
        lat_color="k",
        fig=None,
        ax=None,
    ):
        return plot_centers(
            self,
            center_scale=center_scale,
            section_home_cell=section_home_cell,
            color_home_cell=color_home_cell,
            translate_centers=translate_centers,
            show=show,
            legend=legend,
            pmx=pmx,
            pmy=pmy,
            center_color=center_color,
            center_marker=center_marker,
            lat_home_color=lat_home_color,
            lat_color=lat_color,
            fig=fig,
            ax=ax,
        )

    @copydoc(plot_decay)
    def plot_decay(self, wan_idx, fig=None, ax=None, show=False):
        return plot_decay(self, wan_idx=wan_idx, fig=fig, ax=ax, show=show)

    @copydoc(plot_density)
    def plot_density(
        self,
        wan_idx,
        mark_home_cell=False,
        mark_center=False,
        show_lattice=False,
        dens_size=40,
        lat_size=2,
        show=False,
        fig=None,
        ax=None,
        cbar=True,
    ):
        return plot_density(
            self,
            wan_idx=wan_idx,
            mark_home_cell=mark_home_cell,
            mark_center=mark_center,
            show_lattice=show_lattice,
            show=show,
            dens_size=dens_size,
            lat_size=lat_size,
            fig=fig,
            ax=ax,
            cbar=cbar,
        )
