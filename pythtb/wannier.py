import numpy as np
from .wf_array import WFArray
from .plotting import plot_centers, plot_decay, plot_density
from .mesh import Mesh
from .utils import mat_exp
from itertools import product
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tb_model import TBModel

__all__ = ["Wannier"]

class Wannier:
    """Class for constructing Wannier functions from Bloch energy eigenstates.

    This class provides methods to compute Wannier functions from Bloch energy eigenstates
    and minimize their spreads through disentanglement and maximal localization. 

    Parameters
    ----------
    model : TBModel
        The tight-binding model associated with these Wannier functions.
    energy_eigstates : WFArray
        The Bloch wavefunctions corresponding to the energy eigenstates.
    """
    def __init__(self, model: "TBModel", energy_eigstates: WFArray):
        self._model: "TBModel" = model
        self._energy_eigstates: WFArray = energy_eigstates

        if not self.mesh.is_k_torus:
            raise ValueError(
                "Mesh is not a torus. The Wannier class requires a toroidal k-mesh."
                "To construct a toroidal k-space mesh use `Mesh.build_grid`"
                )
        for ax in self.mesh.k_axes:
            if self.mesh.is_axis_closed(ax):
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

    def report(self, precision=8):
        """Concise report of Wannier centers and spreads."""

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
        Omega_I  = float(getattr(self, "Omega_I",  np.nan))
        Omega_D  = float(getattr(self, "Omega_D",  np.nan))
        Omega_OD = float(getattr(self, "Omega_OD", np.nan))
        Omega_tot = Omega_I + Omega_D + Omega_OD

        lines += [
            f"Omega I   = {Omega_I:.{precision}f}",
            f"Omega D   = {Omega_D:.{precision}f}",
            f"Omega OD  = {Omega_OD:.{precision}f}",
            f"Omega tot = {Omega_tot:.{precision}f}",
        ]

         # determine longest line
        maxlen = max(len(l) for l in lines)
        divider = "=" * maxlen
        sub_div = "-" * maxlen

        # insert dividers at appropriate places
        lines.insert(1, divider)
        lines.insert(len(lines) - 4, sub_div)

        out = "\n".join(lines)
        print(out) 

    @property
    def model(self) -> "TBModel":
        """TBModel object associated with the Wannier functions."""
        return self.energy_eigstates.model

    @property
    def mesh(self) -> Mesh:
        """Mesh object associated with the Wannier functions."""
        return self.energy_eigstates.mesh

    @property
    def energy_eigstates(self) -> WFArray:
        """WFArray object associated with the energy eigenstates."""
        return self._energy_eigstates

    @property
    def tilde_states(self) -> WFArray:
        r"""WFArray object associated with the Bloch-like states.

        These are the Bloch-like states that are Fourier transformed to 
        form the Wannier functions. They are related to the original energy
        eigenstates via the (semi-) unitary transformation

        .. math::
            |\tilde{\psi} \rangle = \sum_{m=1}^{N} 
            U_{mn}^{(\mathbf{k})} |\psi_{m\mathbf{k}} \rangle
        """
        if not hasattr(self, "_tilde_states"):
            raise ValueError(
                "Bloch-like states have not been set. " \
                "Use `set_bloch_like_states` or `single_shot_projection`.")
        return getattr(self, "_tilde_states", None)

    @property
    def nks(self) -> list:
        """Number of k-points in each dimension."""
        return self.mesh.shape_k

    @property
    def wannier(self) -> np.ndarray:
        r"""Wannier functions .

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
            \Omega = \sum_n \langle \mathbf{0} n | r^2 | \mathbf{0} n \rangle 
            - \langle \mathbf{0} n | \mathbf{r} | \mathbf{0} n \rangle^2

        where :math:`|\mathbf{0} n\rangle` are the Wannier functions in the home unit cell.
        """
        if not self.tilde_states.filled:
            raise ValueError("Tilde states are not initialized.")
        return getattr(self, "_spread", None)

    @property
    def Omega_OD(self) -> float:
        r"""Off-diagonal part of gauge-dependent spread :math:`\Omega_{\rm OD}`

        Part of the decomposition of the quadratic spread into gauge-invariant and
        gauge-dependent parts,

        .. math::
            \Omega = \widetilde{\Omega} + \Omega_I \, 
            \quad \widetilde{\Omega} = \Omega_{\rm OD} + \Omega_{\rm D}
        """
        if not self.tilde_states.filled:
            raise ValueError("Tilde states are not initialized.")
        return getattr(self, "_omega_od", None)

    @property
    def Omega_D(self) -> float:
        r"""Off-diagonal part of gauge-dependent spread :math:`\Omega_{\rm D}`

        Part of the decomposition of the quadratic spread into gauge-invariant and
        gauge-dependent parts,

        .. math::
            \Omega = \widetilde{\Omega} + \Omega_I \, 
            \quad \widetilde{\Omega} = \Omega_{\rm OD} + \Omega_{\rm D}
        """
        if not self.tilde_states.filled:
            raise ValueError("Tilde states are not initialized.")
        return getattr(self, "_omega_d", None)

    @property
    def Omega_I(self) -> float:
        r"""Gauge-independent quadratic spread :math:`\Omega_I`

        Part of the decomposition of the quadratic spread into gauge-invariant and
        gauge-dependent parts,

        .. math::
            \Omega = \widetilde{\Omega} + \Omega_I
        
        """
        if not self.tilde_states.filled:
            raise ValueError("Tilde states are not initialized.")
        return getattr(self, "_omega_i", None)
    
    @property
    def centers(self) -> np.ndarray:
        """Centers of the Wannier functions in Cartesian coordinates.
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

    def get_centers(self, cartesian=False):
        if cartesian:
            return self.centers
        else:
            return self.centers @ np.linalg.inv(self.model.lat_vecs)
        
    def _get_trial_wfs(self, twf_list=None):
        if twf_list is None:
            return self._trial_wfs
        
        # number of trial functions to define
        num_tf = len(twf_list)
        if self.model.nspin == 2:
            tfs = np.zeros([num_tf, self.model.norb, 2], dtype=complex)
            for j, tf in enumerate(twf_list):
                assert isinstance(
                    tf, (list, np.ndarray)
                ), "Trial function must be a list of tuples"
                for orb, spin, amp in tf:
                    tfs[j, orb, spin] = amp
                tfs[j] /= np.linalg.norm(tfs[j])

        elif self.model.nspin == 1:
            # initialize array containing tfs = "trial functions"
            tfs = np.zeros([num_tf, self.model.norb], dtype=complex)
            for j, tf in enumerate(twf_list):
                assert isinstance(
                    tf, (list, np.ndarray)
                ), "Trial function must be a list of tuples"
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
        """
        self._trial_wfs = self._get_trial_wfs(tf_list)
        self._tilde_states: WFArray = WFArray(self.model, self.mesh, nstates=self.num_twfs)

    def _set_bloch_like_states(self, bloch_states, cell_periodic=False):
        r"""Internal function to set Bloch-like states with spins flattened.

        From a user facing end, the API is having the additional
        spin axis in the wavefunctions. This is so we ensure that the 
        wavefunctions are defined with the proper basis ordering.

        Internally, it is best to flatten the spin axis, this way the algorithm
        stays spin independent.
        """
        print("Setting Bloch-like states...")
        # set Bloch-like states
        self.tilde_states._set_wfs(
            bloch_states, cell_periodic=cell_periodic, 
            spin_flattened=True
        )

        # Fourier transform Bloch-like states to set WFs
        psi_nk = self.tilde_states.psi_nk
        num_k_axes = self.mesh.num_k_axes

        # FFT NOTE: A non-repeating grid is required for consistent inverse FFTs.
        self._wannier = self.WFs = np.fft.ifftn(psi_nk, axes=[i for i in range(num_k_axes)], norm=None)

        # set spreads
        spread = self._spread_recip(decomp=True)
        self._spread = spread[0][0]
        self._omega_i = spread[0][1]
        self._omega_d = spread[0][2]
        self._omega_od = spread[0][3]
        self._centers = spread[1]


    def set_bloch_like_states(self, states, cell_periodic=False):
        r"""Set the Bloch-like states for the Wannier functions.

        Parameters
        ----------
        states : np.ndarray
            The states to set as Bloch-like states. Must have the shape
            ``(nk1, ..., nstates, n_orbs[, n_spins])``.
        cell_periodic : bool, optional
            Whether to treat the ``states`` as cell-periodic, by default False.
        """
        if not isinstance(states, np.ndarray):
            raise ValueError("Bloch-like states must be a numpy array.")

        if states.ndim != self.mesh.num_k_axes + 2 + (self.model.nspin-1):
            raise ValueError(
                f"Bloch-like states must have shape (nk1, ..., nstates, n_orbs[, n_spins]), "
                f"but got {states.shape}."
            )
        
        if self.model.nspin == 2:
            states = states.reshape((*states.shape[:-2], -1))

        self._set_bloch_like_states(states, cell_periodic=cell_periodic)


    def _compute_Amn(self, psi_nk, twfs, band_idxs):
        r"""Overlap matrix between Bloch states and trial wavefunctions.

        The overlap matrix is defined as

        .. math::

            A_{k, n, j} = <psi_{n,k} | t_{j}> 
            
        where :math:`|\psi_{n\mathbf{k}}\rangle` are the Bloch energy eigenstates and 
        :math:`|t_j\rangle` arethe trial wavefunctions.

       Parameters
       -----------
        band_idxs : list
            Band indices to take from the Bloch states. May choose a subset of bands.
        psi_nk : np.ndarray, optional
            The Bloch states to form the overlap matrix with. By default this will
            choose the energy eigenstates.

        Returns
        --------
        A : np.ndarray
            Overlap matrix with shape ``(*shape_mesh, n_bands, n_trial_wfs)``
        """
        
        if psi_nk is None:
            # get Bloch psi_nk energy eigenstates
            _, psi_nk = self.energy_eigstates.get_states(flatten_spin=True, return_psi=True)

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
        """
        A_k = self._compute_Amn(psi_nk, twfs, state_idx)
        V_k, _, Wh_k = np.linalg.svd(A_k, full_matrices=False)

        if self.model.nspin == 2:
            # flatten spin dimensions
            psi_nk = psi_nk.reshape((*psi_nk.shape[:-2], -1))

        # take only state_idxs
        psi_nk = np.take(psi_nk, state_idx, axis=-2)
        
        # optimal alignment
        psi_tilde = np.einsum(
            "...mn, ...mj -> ...nj", V_k @ Wh_k, psi_nk
        )  # shape: (*mesh_shape, states, orbs*n_spin])

        return psi_tilde

    def single_shot_projection(
        self, 
        tf_list: list = None, 
        band_idxs: list = None, 
        use_tilde=False
    ):
        r"""Perform Wannierization via optimal alignment with trial functions.

        Sets the Wannier functions in home unit cell with associated spreads, centers, trial functions
        and Bloch-like (tilde) states using the single shot projection method.

        Parameters
        ----------
        tf_list : list, optional
            List of tuples with sites and weights. Can be un-normalized.
        band_idxs : list, optional
            Band indices to Wannierize. Defaults to occupied bands (lower half).

        Returns
        -------
        w_0n : np.array
            Wannier functions in home unit cell
        """
        if tf_list is None:
            if self.trial_wfs is None:
                raise ValueError("Trial wavefunctions must be set before Wannierization.")
        else:
            self.set_trial_wfs(tf_list)
        
        twfs = self.trial_wfs

        if use_tilde:
            # projecting back onto tilde states
            if band_idxs is None:  # assume we are projecting onto all tilde states
                band_idxs = list(range(self.tilde_states.nstates))

            psi_til_til = self._single_shot_project(
                self.tilde_states.psi_nk, twfs, state_idx=band_idxs
            )
            self._set_bloch_like_states(psi_til_til, cell_periodic=False)

        else:
            # projecting onto Bloch energy eigenstates
            if band_idxs is None:  # assume we are Wannierizing occupied bands
                n_occ = int(self.energy_eigstates.nstates / 2)  # assuming half filled
                band_idxs = list(range(0, n_occ))

            # shape: (*nks, states, orbs*n_spin])
            psi_tilde = self._single_shot_project(
                self.energy_eigstates.psi_nk, twfs, state_idx=band_idxs
            )
            self._set_bloch_like_states(psi_tilde, cell_periodic=False)


    def _spread_recip(self, decomp=False):
        r"""
        Args:
            decomp (bool, optional):
                Whether to compute and return decomposed spread. Defaults to False.

        Returns:
            spread | [spread, Omega_i, Omega_tilde], expc_rsq, expc_r_sq :
                quadratic spread, the expectation of the position squared,
                and the expectation of the position vector squared
        """
        M = self.tilde_states.Mmn
        w_b, k_shell, _ = self.energy_eigstates.get_shell_weights()
        w_b, k_shell = w_b[0], k_shell[0]  # Assume only one shell for now

        n_states = self.tilde_states.nstates
        nks = self.nks
        k_axes = tuple(self.mesh.k_axes)
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
                (1 / Nk)
                * w_b
                * (
                    np.sum((-log_diag_M_imag - k_shell @ r_n.T) ** 2)
                )
            )

            Omega_od = (
                (1 / Nk)
                * w_b
                * (
                    + np.sum(abs(M) ** 2)
                    - np.sum(abs_diag_M_sq)
                )
            )
            return [spread_n, Omega_i, Omega_d, Omega_od], r_n, rsq_n

        else:
            return spread_n, r_n, rsq_n

    def _get_omega_til(self, Mmn, wb, k_shell):
        nks = self.nks
        Nk = np.prod(nks)
        k_axes = tuple([i for i in range(len(nks))])

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
        k_axes = tuple([i for i in range(len(nks))])

        diag_M = np.diagonal(Mmn, axis1=-1, axis2=-2)
        log_diag_M_imag = np.log(diag_M).imag
        r_n = -(1 / Nk) * wb * np.sum(log_diag_M_imag, axis=k_axes).T @ k_shell

        Omega_d = (
            (1 / Nk)
            * wb
            * (
                np.sum((-log_diag_M_imag - k_shell @ r_n.T) ** 2)
            )
        )
        return Omega_d


    def _get_omega_od(self, Mmn, wb):
        Nk = np.prod(self.nks)
        diag_M = np.diagonal(Mmn, axis1=-1, axis2=-2)
        abs_diag_M_sq = abs(diag_M) ** 2

        Omega_od = (
            (1 / Nk)
            * wb
            * (
                + np.sum(abs(Mmn) ** 2)
                - np.sum(abs_diag_M_sq)
            )
        )
        return Omega_od
    
    def _get_omega_i(self, Mmn, wb, k_shell):
        Nk = np.prod(self.tilde_states.mesh.shape_k)
        n_states = self.tilde_states.nstates
        Omega_i = wb * n_states * k_shell.shape[0] - (1 / Nk) * wb * np.sum(
            abs(Mmn) ** 2
        )
        return Omega_i

    def get_omega_i_k(self):
        r"""Calculate the gauge-independent quadratic spread for the Wannier functions.

        This function computes the quadratic spread :math:`\Omega_I`
        of the Wannier functions as a function of `k`. This is related to the
        real part of the quantum metric.
        """
        P = self.tilde_states.get_projector()
        Q_nbr = self.tilde_states._Q_nbr

        nks = self.nks
        Nk = np.prod(nks)
        w_b, _, idx_shell = self.tilde_states.get_shell_weights(n_shell=1)
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
        n_wfs : int | None = None,
        frozen_bands: list | None = None,
        disentang_bands: list | str ="occupied",
        iter_num: int = 100,
        tol: float = 1e-10,
        beta: float = 1,
        verbose: bool = False,
        tf_speedup: bool = False
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
        nks = self.energy_eigstates.mesh.shape_k
        # k_axes = self.energy_eigstates.mesh.k_axes
        Nk = np.prod(nks)
        n_orb = self.model.norb
        n_occ = int(n_orb / 2)

        # Assumes only one shell for now
        w_b, _, _ = self.energy_eigstates.get_shell_weights(n_shell=1)
        w_b = w_b[0]  # Assume only one shell for now

        # initial subspace
        energy_eigstates = self.energy_eigstates
        u_nk = energy_eigstates.get_states(flatten_spin=True)
        # u_wfs_til = init_states.get_states(flatten_spin=True)

        if n_wfs is None:
            # assume number of states in the subspace is number of tilde states
            N_wfs = self.tilde_states.nstates

        if disentang_bands == "occupied":
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

            P_tilde = self.tilde_states.get_projectors()

            # chosing initial subspace as highest eigenvalues
            MinMat = Q_inner @ P_tilde @ Q_inner
            _, eigvecs = np.linalg.eigh(MinMat)
            eigvecs = np.swapaxes(eigvecs, -1, -2)

            init_evecs = eigvecs[..., -(N_wfs - N_inner) :, :]
            init_states = WFArray(self.model, self.mesh, nstates=init_evecs.shape[-2])
            init_states._set_wfs(init_evecs, cell_periodic=False, spin_flattened=True)

            comp_bands = list(np.setdiff1d(disentang_bands, frozen_bands))
            comp_states = u_nk.take(comp_bands, axis=-2)

            # min_states = np.einsum(
            #     "...ij, ...ik->...jk", eigvecs[..., -(N_wfs - N_inner) :], outer_states
            # )

          
        P, Q = init_states.get_projectors(return_Q=True)
        P_nbr, Q_nbr = init_states._P_nbr, init_states._Q_nbr

        T_kb = np.einsum("...ij, ...kji -> ...k", P, Q_nbr)
        omega_I_prev = (1 / Nk) * w_b * np.sum(T_kb)
        if verbose:
            print(f"Initial Omega_I: {omega_I_prev.real}")
        
        P_min = np.copy(P)  # for start of iteration
        P_nbr_min = np.copy(P_nbr)  # for start of iteration

        if tf_speedup:
            from tensorflow import convert_to_tensor
            from tensorflow import complex64 as tfcomplex64
            from tensorflow.linalg import eigh as tfeigh

        for i in range(iter_num):
            # states spanning optimal subspace minimizing gauge invariant spread
            P_avg = w_b * np.sum(P_nbr_min, axis=-3)
            Z = comp_states.conj() @ P_avg @ np.swapaxes(comp_states, -1, -2)
            
            if tf_speedup:
                Z_tf = convert_to_tensor(Z, dtype=tfcomplex64)
                _, eigvecs_tf = tfeigh(Z_tf)
                eigvecs = eigvecs_tf.numpy()
            else:
                _, eigvecs = np.linalg.eigh(Z)  # [val, idx]


            evecs_keep = eigvecs[..., -(N_wfs - N_inner) :]
            states_min = np.swapaxes(evecs_keep, -1, -2) @ comp_states

            min_wfa = WFArray(self.model, self.mesh, nstates=states_min.shape[-2])
            min_wfa._set_wfs(states_min, cell_periodic=True, spin_flattened=True)
            P_new = min_wfa._P
            P_nbr_new = min_wfa._P_nbr

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

            if verbose:
                delta = omega_I_new - omega_I_prev
                print(f"iter {i:4d} | Ω_I = {omega_I_new.real:12.9e} | ΔΩ = {delta.real:10.5e}")

            if abs(omega_I_prev - omega_I_new) <= tol:
                if verbose:
                    print(f"Converged within tolerance in {i} iterations. Breaking the loop.")
                break

            if omega_I_new > omega_I_prev:
                beta = max(beta - 0.01, 0)
                if verbose:
                    print(f"Warning: Ω_I is increasing. Reducing beta to {beta}.")

            omega_I_prev = omega_I_new

        if frozen_bands is not None:
            return_states = np.concatenate((inner_states, states_min), axis=-2)
            return return_states
        else:
            return states_min

    def max_loc_unitary(
        self, eps=1e-3, iter_num=100, verbose=False, tol=1e-10, grad_min=1e-3
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
        w_b, k_shell, idx_shell = self.energy_eigstates.get_shell_weights()
        # Assumes only one shell for now
        w_b, k_shell, idx_shell = w_b[0], k_shell[0], idx_shell[0]
        nks = self.nks
        shape_mesh = self.mesh.shape_mesh
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
        eta = 1
        for i in range(iter_num):
            r_n = (
                -(1 / Nk)
                * w_b
                * np.sum(
                    log_diag_M_imag := np.log(np.diagonal(M, axis1=-1, axis2=-2)).imag,
                    axis=(0, 1),
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
            A_R = (R - np.transpose(R, axes=(0, 1, 2, 4, 3)).conj()) / 2
            S_T = (T + np.transpose(T, axes=(0, 1, 2, 4, 3)).conj()) / (2j)
            G = 4 * w_b * np.sum(A_R - S_T, axis=-3)
            U = np.einsum("...ij, ...jk -> ...ik", U, mat_exp(eta * eps * G))

            for idx, idx_vec in enumerate(idx_shell):
                M[..., idx, :, :] = (
                    np.swapaxes(U, -1, -2).conj()
                    @ M0[..., idx, :, :]
                    @ np.roll(U, shift=tuple(-idx_vec), axis=tuple(self.mesh.k_axes))
                )

            grad_mag = np.linalg.norm(np.sum(G, axis=tuple(self.mesh.k_axes)))
            omega_tilde_new = self._get_omega_til(M, w_b, k_shell)

            if verbose:
                delta = omega_tilde_new - omega_tilde_prev
                print(f"iter {i:4d} | Ω_tilde = {omega_tilde_new.real:12.9e} | ΔΩ = {delta.real:12.5e} | ‖∇‖ = {grad_mag:10.5e}")

            # Check for convergence
            if (
                abs(grad_mag) <= grad_min
                and abs(omega_tilde_prev - omega_tilde_new) <= tol
            ):
                if verbose:
                    print(f"Converged within tolerance in {i} iterations. Breaking the loop.")
                break

            if grad_mag_prev < grad_mag and i != 0:
                if verbose:
                    print("Warning: Gradient increasing.")
                # Reduce step size to try and stabilize
                eps *= 0.9

            grad_mag_prev = grad_mag
            omega_tilde_prev = omega_tilde_new

        return U

    def disentangle(
        self,
        disentang_window="occupied",
        frozen_window=None,
        dim_ss=None,
        twfs=None,
        iter_num=1000,
        tol=1e-5,
        beta=1,
        tf_speedup: bool = False,
        verbose=False,
    ):
        r"""Disentanglement of a subspace that minimizes gauge-dependent spread.

        This procedure aims to find a ``dim_ss``-dimensional subspace that
        minimizes the gauge-dependent spread. This is achieved using the iterative
        Souza-Marzari-Vanderbilt (disentanglement) method [souza-marzari-vanderbilt]_.
        Optionally, one may choose a disentanglement band window from which a linear
        combination of states is selected. One may also choose a frozen band window
        which picks a set of bands to be fully included in the disentangled subspace.

        After running the disentanglement procedure, the ``.tilde_states`` wavefunction
        array is populated with the states spanning the subspace that minimizes the
        gauge-independent spread.

        Parameters
        ----------
        disentang_window : {str, list}, optional
            Disentanglement window, by default "occupied"
        frozen_window : list, optional
            Frozen window, by default None
        twfs : list[list[tuple]], optional
            Trial wavefunctions, by default None
        n_wfs : int, optional
            Number of states in disentangled subspace, by default None
        iter_num : int, optional
            Maximum number of iterations for the optimization, by default 1000
        tol : float, optional
            Tolerance, by default 1e-5
        beta : int, optional
            Mixing parameter, by default 1
        tf_speedup : bool, optional
            Whether to use `tensorflow` for speedups, by default False
        verbose : bool, optional
            Whether to print progress messages, by default False

        Notes
        ------
        This method uses the algorithm outlined in the original paper [souza-marzari-vanderbilt]_.
        The subspace is updated via   

        References
        ----------
        .. [souza-marzari-vanderbilt] Souza, A. M., Marzari, N., & Vanderbilt, D. 
        """
        # if we haven't done single-shot projection yet (set tilde states)
        if twfs is not None:
            # initialize tilde states
            self.set_trial_wfs(twfs)

            twfs = self.trial_wfs

            n_occ = int(self.energy_eigstates.nstates / 2)  # assuming half filled
            band_idxs = list(range(0, n_occ))  # project onto occ manifold

            psi_til_init = self._single_shot_project(
                self.energy_eigstates.psi_nk, twfs, state_idx=band_idxs
            )
            self._set_bloch_like_states(psi_til_init, cell_periodic=False)
        else:
            assert hasattr(
                self.tilde_states, "_u_nk"
            ), "Need pass trial wavefunction list or initalize tilde states with single_shot_projection()."

        # Minimizing Omega_I via disentanglement
        util_min = self._optimal_subspace(
            n_wfs=dim_ss,
            disentang_bands=disentang_window,
            frozen_bands=frozen_window,
            iter_num=iter_num,
            verbose=verbose,
            beta=beta,
            tol=tol,
            tf_speedup=tf_speedup,
        )

        self._set_bloch_like_states(util_min, cell_periodic=True)

    def max_localize(self, eps=1e-3, iter_num=1000, tol=1e-5, grad_min=1e-3, verbose=False):
        r"""Unitary transformation to minimize the gauge-dependent spread.

        This method uses "maximal localization" to 
        iteratively find the unitary transformation that minimizes 
        the gauge-dependent spread of the Wannier functions. After calling
        this function, the unitary transformation is applied to the ``.tilde_states``
        and its wavefunctions are updated.

        Parameters
        ----------
        eps : float
            Step size for gradient descent.
        iter_num : int
            Maximum number of iterations for the optimization.
        grad_min : float
            Minimum gradient magnitude for convergence.
        verbose : bool
            If True, print progress messages.

        Notes
        -----
        Finds the optimal unitary rotation that minimizes the gauge
        dependent spread :math:`\widetilde{\Omega}` using the Marzari-Vanderbilt
        algorithm from [marzari-vanderbilt]_.

        References
        ----------
        .. [marzari-vanderbilt] Marzari, N., & Vanderbilt, D. (1997). 
            Maximally localized generalized Wannier functions for composite energy bands. 
            *Physical Review B*, 56(20), 12847.
        """

        U = self.max_loc_unitary(
            eps=eps, iter_num=iter_num, verbose=verbose, tol=tol, grad_min=grad_min
        )

        u_tilde_wfs = self.tilde_states.get_states(flatten_spin=True)
        util_max_loc = np.einsum("...ji, ...jm -> ...im", U, u_tilde_wfs)

        self._set_bloch_like_states(util_max_loc, cell_periodic=True)

    def min_spread(
        self,
        outer_window="occupied",
        inner_window=None,
        twfs_1=None,
        twfs_2=None,
        N_wfs=None,
        iter_num_omega_i=1000,
        iter_num_omega_til=1000,
        eps=1e-3,
        tol_omega_i=1e-5,
        tol_omega_til=1e-10,
        grad_min=1e-3,
        beta=1,
        verbose=False,
    ):
        r"""Find the maximally localized Wannier functions using the projection method.

        This method performs three steps:

        1. Disentangles the Wannier functions to find the optimal subspace that minimizes
        the gauge-dependent spread.

        2. Applies a second projection to the subspace to find the optimal gauge.

        3. Maximally localizes the Wannier functions within the optimal gauge.
        """

        ### Subspace selection ###
        self.disentangle(
            outer_window=outer_window,
            inner_window=inner_window,
            twfs=twfs_1,
            N_wfs=N_wfs,
            iter_num=iter_num_omega_i,
            tol=tol_omega_i,
            beta=beta,
            verbose=verbose,
        )

        ### Second projection ###
        # if we need a smaller number of twfs b.c. of subspace selec
        if twfs_2 is not None:
            twfs = self._get_trial_wfs(twfs_2)
            psi_til_til = self._single_shot_project(
                self.tilde_states.psi_nk,
                twfs,
                state_idx=list(range(self.tilde_states.nstates)),
            )
        # choose same twfs as in subspace selection
        else:
            psi_til_til = self._single_shot_project(
                self.tilde_states.psi_nk,
                self.trial_wfs,
                state_idx=list(range(self.tilde_states.nstates)),
            )

        self._set_bloch_like_states(psi_til_til, cell_periodic=False)

        ### Finding optimal gauge with maxloc ###
        self.max_localize(
            eps=eps,
            iter_num=iter_num_omega_til,
            tol=tol_omega_til,
            grad_min=grad_min,
            verbose=verbose,
        )


    def interp_bands(self, k_nodes, n_interp: int = 20, wan_idxs=None, ret_eigvecs=False):
        r"""Wannier interpolate the band structure along a k-path.

        This method uses the Wannier functions to interpolate the band structure
        along a specified k-path. It constructs a tight-binding Hamiltonian in the
        Wannier basis, diagonalizes it, and Fourier transforms back to k-space
        along the k-path defined by ``k_nodes``. 

        """
        u_tilde = self.tilde_states.get_states(flatten_spin=False)
        if wan_idxs is not None:
            u_tilde = np.take_along_axis(u_tilde, wan_idxs, axis=-2)

        H_k = self.energy_eigstates.hamiltonian
        if self.model.nspin == 2:
            new_shape = H_k.shape[:-4] + (2 * self.model.norb, 2 * self.model.norb)
            H_k = H_k.reshape(*new_shape)

        H_rot_k = u_tilde.conj() @ H_k @ np.swapaxes(u_tilde, -1, -2)
        eigvals, eigvecs = np.linalg.eigh(H_rot_k)
        eigvecs = np.einsum("...ij, ...ik->...kj", u_tilde, eigvecs)
        # eigvecs = np.swapaxes(eigvecs, -1, -2)

        k_mesh = self.mesh.get_k_points()
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
        k_path, _, _ = self.model.k_path(k_nodes, nk=n_interp, report=False)

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

    def _interp_op(self, O_k, k_path, plaq=False):
        return self.tilde_states.interp_op(O_k, k_path, plaq=plaq)

    def _get_supercell(self, wan_idx, special_sites=None):
        w0 = self.WFs  # .reshape((*self.WFs.shape[:self.k_mesh.dim+1], -1))
        center = self.centers[wan_idx]
        orbs = self.model.orb_vecs
        lat_vecs = self.model.lat_vecs

        # Initialize arrays to store positions and weights
        positions = {
            "all": {"xs": [], "ys": [], "r": [], "wt": [], "phase": []},
            "home even": {"xs": [], "ys": [], "r": [], "wt": [], "phase": []},
            "home odd": {"xs": [], "ys": [], "r": [], "wt": [], "phase": []},
            "omit": {"xs": [], "ys": [], "r": [], "wt": [], "phase": []},
            "even": {"xs": [], "ys": [], "r": [], "wt": [], "phase": []},
            "odd": {"xs": [], "ys": [], "r": [], "wt": [], "phase": []},
        }

        for tx, ty in self.supercell:
            for i, orb in enumerate(orbs):
                # Extract relevant parameters
                wf_value = w0[tx, ty, wan_idx, i]
                wt = np.sum(np.abs(wf_value) ** 2)
                # phase = np.arctan2(wf_value.imag, wf_value.real)
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
                # positions['all']['phase'].append(phase)

                # Handle special sites if applicable
                if special_sites is not None and i in special_sites:
                    positions["special"]["xs"].append(x)
                    positions["special"]["ys"].append(y)
                    positions["special"]["r"].append(rad)
                    positions["special"]["wt"].append(wt)
                    # positions['special']['phase'].append(phase)
                # Separate even and odd index sites
                if i % 2 == 0:
                    positions["even"]["xs"].append(x)
                    positions["even"]["ys"].append(y)
                    positions["even"]["r"].append(rad)
                    positions["even"]["wt"].append(wt)
                    # positions['even']['phase'].append(phase)
                    if tx == ty == 0:
                        positions["home even"]["xs"].append(x)
                        positions["home even"]["ys"].append(y)
                        positions["home even"]["r"].append(rad)
                        positions["home even"]["wt"].append(wt)
                        # positions['home even']['phase'].append(phase)

                else:
                    positions["odd"]["xs"].append(x)
                    positions["odd"]["ys"].append(y)
                    positions["odd"]["r"].append(rad)
                    positions["odd"]["wt"].append(wt)
                    # positions['odd']['phase'].append(phase)
                    if tx == ty == 0:
                        positions["home odd"]["xs"].append(x)
                        positions["home odd"]["ys"].append(y)
                        positions["home odd"]["r"].append(rad)
                        positions["home odd"]["wt"].append(wt)
                        # positions['home odd']['phase'].append(phase)

        # Convert lists to numpy arrays (batch processing for cleanliness)
        for key, data in positions.items():
            for sub_key in data:
                positions[key][sub_key] = np.array(data[sub_key])

        self.positions = positions

    def plot_centers(
            self, 
            omit_sites=None, 
            center_scale=200,
            section_home_cell=True, 
            color_home_cell=True,
            translate_centers=False,
            show=False, legend=False, pmx=4, pmy=4,
            kwargs_centers={'s': 80, 'marker': '*', 'c': 'g'},
            kwargs_omit={'s': 50, 'marker': 'x', 'c':'k'},
            kwargs_lat_ev={'s':10, 'marker': 'o', 'c':'k'}, 
            kwargs_lat_odd={'s':10, 'marker': 'o', 'facecolors':'none', 'edgecolors':'k'},
            fig=None, ax=None
        ):
        return plot_centers(
            self, omit_sites=omit_sites, center_scale=center_scale,
            section_home_cell=section_home_cell, color_home_cell=color_home_cell,
            translate_centers=translate_centers,
            show=show, legend=legend, pmx=pmx, pmy=pmy, kwargs_centers=kwargs_centers,
            kwargs_lat_ev=kwargs_lat_ev,
            kwargs_lat_odd=kwargs_lat_odd, fig=fig, ax=ax
            )

    def plot_decay(
        self, wan_idx, fit_deg=None, fit_rng=None, ylim=None, 
        fig=None, ax=None, show=False
        ):

        return plot_decay(self, wan_idx=wan_idx, fit_deg=fit_deg, fit_rng=fit_rng,
            ylim=ylim, fig=fig, ax=ax, show=show)

    def plot_density(
        self, wan_idx,
        mark_home_cell=False,
        mark_center=False, show_lattice=False,
        show=False,
        scatter_size=40, lat_size=2, fig=None, ax=None, cbar=True
        ):

        return plot_density(
            self, wan_idx=wan_idx,
            mark_home_cell=mark_home_cell, mark_center=mark_center,
            show_lattice=show_lattice,
            show=show, scatter_size=scatter_size, 
            lat_size=lat_size, fig=fig, ax=ax, cbar=cbar
        )
    
