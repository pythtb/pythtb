import numpy as np
from .wf_array import WFArray
from .mesh import Mesh
from itertools import product
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tb_model import TBModel

__all__ = ["Wannier"]

class Wannier:
    """
    Class for constructing Wannier functions from Bloch wavefunctions.

    Parameters
    ----------
    model : TBModel
        The tight-binding model associated with these Wannier functions.
    energy_eigstates : WFArray
        The Bloch wavefunctions corresponding to the energy eigenstates.
    nks : list
        The k-point mesh dimensions.
    """
    def __init__(self, model: "TBModel", energy_eigstates: WFArray):
        self.model: "TBModel" = model
        self.energy_eigstates: WFArray = energy_eigstates

        self.k_mesh: Mesh = energy_eigstates.mesh
        self._nks: list = self.k_mesh.shape_k

        self.tilde_states: WFArray = WFArray(model, self.k_mesh)
        self.tilde_states.solve_mesh(use_metal=True)

        halfs = [nk // 2 for nk in self._nks]
        ranges = [np.arange(-h, h) for h in halfs]
        mesh = np.meshgrid(*ranges, indexing="ij")
        # used for real space looping of WFs
        self.supercell = np.stack(mesh, axis=-1).reshape(  # (..., len(nks))
            -1, len(self._nks)
        )  # (product, dims)
        # list(product(*[range(-int((nk-nk%2)/2), int((nk-nk%2)/2)) for nk in nks]))  # used for real space looping of WFs

    def get_Bloch_Ham(self):
        return self.tilde_states.get_Bloch_Ham()

    def get_centers(self, Cartesian=False):
        if Cartesian:
            return self.centers
        else:
            return self.centers @ np.linalg.inv(self.model._lat_vecs)

    def get_trial_wfs(self, tf_list):
        """
        Args:
            tf_list: list[int | list[tuple]]
                list of tuples defining the orbital and amplitude of the trial function
                on that orbital. Of the form [ [(orb, amp), ...], ...]. If spin is included,
                then the form is [ [(orb, spin, amp), ...], ...]

        Returns:
            tfs: np.ndarray
                Array of trial functions
        """

        # number of trial functions to define
        num_tf = len(tf_list)

        if self.model.nspin == 2:
            tfs = np.zeros([num_tf, self.model.norb, 2], dtype=complex)
            for j, tf in enumerate(tf_list):
                assert isinstance(
                    tf, (list, np.ndarray)
                ), "Trial function must be a list of tuples"
                for orb, spin, amp in tf:
                    tfs[j, orb, spin] = amp
                tfs[j] /= np.linalg.norm(tfs[j])

        elif self.model.nspin == 1:
            # initialize array containing tfs = "trial functions"
            tfs = np.zeros([num_tf, self.model.norb], dtype=complex)
            for j, tf in enumerate(tf_list):
                assert isinstance(
                    tf, (list, np.ndarray)
                ), "Trial function must be a list of tuples"
                for site, amp in tf:
                    tfs[j, site] = amp
                tfs[j] /= np.linalg.norm(tfs[j])

        return tfs

    def set_trial_wfs(self, tf_list):
        tfs = self.get_trial_wfs(tf_list)
        self.trial_wfs = tfs
        self.n_twfs = tfs.shape[0]
        return

    def get_tf_ovlp_mat(self, band_idxs, psi_wfs=None):
        """
        Returns A_{k, n, j} = <psi_{n,k} | t_{j}> where psi are Bloch states and t are
        the trial wavefunctions.

        Args:
            psi_wfs (np.array): Bloch eigenstates
            tfs (np.array): trial wfs
            state_idx (list): band indices to form overlap matrix with

        Returns:
            A (np.array): overlap matrix
        """
        if psi_wfs is None:
            # get Bloch psi_nk energy eigenstates
            psi_wfs = self.energy_eigstates.get_bloch_states(flatten_spin=True)["bloch"]

        # only keep band_idxs
        psi_wfs = np.take(psi_wfs, band_idxs, axis=-2)

        assert hasattr(
            self, "trial_wfs"
        ), "Must initialize trial wfs with set_trial_wfs()"
        trial_wfs = self.trial_wfs
        # flatten along spin dimension in case spin is considered
        trial_wfs = trial_wfs.reshape((*trial_wfs.shape[:1], -1))

        A_k = np.einsum("...ij, kj -> ...ik", psi_wfs.conj(), trial_wfs)
        return A_k

    def set_tf_ovlp_mat(self, band_idxs):
        A_k = self.get_tf_ovlp_mat(band_idxs)
        self.A_k = A_k
        return

    def set_tilde_states(self, tilde_states, cell_periodic=False):
        # set tilde states
        self.tilde_states.set_wfs(
            tilde_states, cell_periodic=cell_periodic, spin_flattened=True
        )

        # Fourier transform Bloch-like states to set WFs
        psi_wfs = self.tilde_states._psi_wfs
        dim_k = len(psi_wfs.shape[:-2])
        self.WFs = np.fft.ifftn(psi_wfs, axes=[i for i in range(dim_k)], norm=None)

        # set spreads
        spread = self.spread_recip(decomp=True)
        self.spread = spread[0][0]
        self.omega_i = spread[0][1]
        self.omega_til = spread[0][2]
        self.centers = spread[1]

    def get_psi_tilde(self, psi_wfs, state_idx):
        """
        Performs optimal alignment of psi_wfs with tfs.
        """
        A_k = self.get_tf_ovlp_mat(state_idx, psi_wfs=psi_wfs)
        V_k, _, Wh_k = np.linalg.svd(A_k, full_matrices=False)

        # flatten spin dimensions
        psi_wfs = psi_wfs.reshape((*psi_wfs.shape[: self.k_mesh.dim + 1], -1))
        # take only state_idxs
        psi_wfs = np.take(psi_wfs, state_idx, axis=-2)
        # optimal alignment
        psi_tilde = np.einsum(
            "...mn, ...mj -> ...nj", V_k @ Wh_k, psi_wfs
        )  # shape: (*nks, states, orbs*n_spin])

        return psi_tilde

    def single_shot(
        self, tf_list: list | None = None, band_idxs: list | None = None, tilde=False
    ):
        """
        Sets the Wannier functions in home unit cell with associated spreads, centers, trial functions
        and Bloch-like states using the single shot projection method.

        Args:
            tf_list (list): List of tuples with sites and weights. Can be un-normalized.
            band_idxs (list | None): Band indices to Wannierize. Defaults to occupied bands (lower half).
        Returns:
            w_0n (np.array): Wannier functions in home unit cell
        """
        if tf_list is None:
            assert hasattr(
                self, "trial_wfs"
            ), "Must initialize trial wfs with set_trial_wfs()"
        else:
            self.set_trial_wfs(tf_list)

        if tilde:
            # projecting back onto tilde states
            if band_idxs is None:  # assume we are projecting onto all tilde states
                band_idxs = list(range(self.tilde_states._n_states))

            psi_til_til = self.get_psi_tilde(
                self.tilde_states._psi_wfs, state_idx=band_idxs
            )
            self.set_tilde_states(psi_til_til, cell_periodic=False)

        else:
            # projecting onto Bloch energy eigenstates
            if band_idxs is None:  # assume we are Wannierizing occupied bands
                n_occ = int(self.energy_eigstates._n_states / 2)  # assuming half filled

                # if self.model._nspin == 1:
                #     n_occ = int(self.energy_eigstates._n_states / 2)  # assuming half filled
                # elif self.model._nspin == 2:
                #     # TODO check *2
                #     n_occ = int(self.energy_eigstates._n_states / 2)#*2  # assuming half filled

                band_idxs = list(range(0, n_occ))

            # shape: (*nks, states, orbs*n_spin])
            psi_tilde = self.get_psi_tilde(
                self.energy_eigstates._psi_wfs, state_idx=band_idxs
            )
            # TODO Check if this is messing up in reshape
            if self.model._nspin == 2:
                psi_tilde = psi_tilde.reshape(
                    (*psi_tilde.shape[: self.k_mesh.dim + 1], -1, 2)
                )
            self.tilde_states.set_wfs(psi_tilde, cell_periodic=False)

        psi_wfs = self.tilde_states._psi_wfs
        dim_k = self.k_mesh.dim
        # DFT
        self.WFs = np.fft.ifftn(psi_wfs, axes=[i for i in range(dim_k)], norm=None)

        spread = self.spread_recip(decomp=True)
        self.spread = spread[0][0]
        self.omega_i = spread[0][1]
        self.omega_til = spread[0][2]
        self.centers = spread[1]

    def spread_recip(self, decomp=False):
        """
        Args:
            M (np.array):
                overlap matrix
            decomp (bool, optional):
                Whether to compute and return decomposed spread. Defaults to False.

        Returns:
            spread | [spread, Omega_i, Omega_tilde], expc_rsq, expc_r_sq :
                quadratic spread, the expectation of the position squared,
                and the expectation of the position vector squared
        """
        M = self.tilde_states._M
        w_b, k_shell, _ = self.k_mesh.get_weights()
        w_b, k_shell = w_b[0], k_shell[0]  # Assume only one shell for now

        n_states = self.tilde_states._n_states
        nks = self.tilde_states.k_mesh.nks
        k_axes = tuple([i for i in range(len(nks))])
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
            Omega_tilde = (
                (1 / Nk)
                * w_b
                * (
                    np.sum((-log_diag_M_imag - k_shell @ r_n.T) ** 2)
                    + np.sum(abs(M) ** 2)
                    - np.sum(abs_diag_M_sq)
                )
            )
            return [spread_n, Omega_i, Omega_tilde], r_n, rsq_n

        else:
            return spread_n, r_n, rsq_n

    def _get_Omega_til(self, M, w_b, k_shell):
        nks = self.tilde_states.k_mesh.nks
        Nk = self.tilde_states.k_mesh.Nk
        k_axes = tuple([i for i in range(len(nks))])

        diag_M = np.diagonal(M, axis1=-1, axis2=-2)
        log_diag_M_imag = np.log(diag_M).imag
        abs_diag_M_sq = abs(diag_M) ** 2

        r_n = -(1 / Nk) * w_b * np.sum(log_diag_M_imag, axis=k_axes).T @ k_shell

        Omega_tilde = (
            (1 / Nk)
            * w_b
            * (
                np.sum((-log_diag_M_imag - k_shell @ r_n.T) ** 2)
                + np.sum(abs(M) ** 2)
                - np.sum(abs_diag_M_sq)
            )
        )
        return Omega_tilde

    def _get_Omega_I(self, M, w_b, k_shell):
        Nk = self.tilde_states.k_mesh.Nk
        n_states = self.tilde_states._n_states
        Omega_i = w_b * n_states * k_shell.shape[0] - (1 / Nk) * w_b * np.sum(
            abs(M) ** 2
        )
        return Omega_i

    def get_Omega_I(self, tilde=True):
        if tilde:
            P, Q = self.tilde_states.get_projector(return_Q=True)
            _, Q_nbr = self.tilde_states.get_nbr_projector(return_Q=True)
        else:
            P, Q = self.energy_eigstates.get_projector(return_Q=True)
            _, Q_nbr = self.energy_eigstates.get_nbr_projector(return_Q=True)

        nks = self.k_mesh.nks
        Nk = np.prod(nks)
        num_nnbrs = self.k_mesh.num_nnbrs
        w_b, _, idx_shell = self.k_mesh.get_weights(N_sh=1)

        T_kb = np.zeros((*nks, num_nnbrs), dtype=complex)
        for idx, idx_vec in enumerate(idx_shell[0]):  # nearest neighbors
            T_kb[..., idx] = np.trace(
                P[..., :, :] @ Q_nbr[..., idx, :, :], axis1=-1, axis2=-2
            )

        return (1 / Nk) * w_b[0] * np.sum(T_kb)

    def get_Omega_I_k(self, tilde=True):
        if tilde:
            P = self.tilde_states.get_projector()
            _, Q_nbr = self.tilde_states.get_nbr_projector(return_Q=True)
        else:
            P = self.energy_eigstates.get_projector()
            _, Q_nbr = self.energy_eigstates.get_nbr_projector(return_Q=True)

        nks = self.k_mesh.nks
        Nk = np.prod(nks)
        num_nnbrs = self.k_mesh.num_nnbrs
        w_b, _, idx_shell = self.k_mesh.get_weights(N_sh=1)

        T_kb = np.zeros((*nks, num_nnbrs), dtype=complex)
        for idx, idx_vec in enumerate(idx_shell[0]):  # nearest neighbors
            T_kb[..., idx] = np.trace(
                P[..., :, :] @ Q_nbr[..., idx, :, :], axis1=-1, axis2=-2
            )

        return (1 / Nk) * w_b[0] * np.sum(T_kb, axis=-1)

    ####### Maximally Localized WF #######

    def find_optimal_subspace(
        self,
        N_wfs=None,
        inner_window=None,
        outer_window="occupied",
        iter_num=100,
        verbose=False,
        tol=1e-10,
        alpha=1,
    ):
        # useful constants
        nks = self._nks
        Nk = np.prod(nks)
        n_orb = self.model.n_orb
        n_occ = int(n_orb / 2)
        if self.model._nspin == 2:
            n_occ *= 2

        # eigenenergies and eigenstates for inner/outer window
        energies = self.energy_eigstates.get_energies()
        unk_states = self.energy_eigstates.get_states()["Cell periodic"]
        # initial subspace
        init_states = self.tilde_states

        if self.model._nspin == 2:
            unk_states = unk_states.reshape(
                (*unk_states.shape[: self.k_mesh.dim + 1], -1)
            )

        #### Setting inner/outer energy windows ####

        # number of states in target manifold
        if N_wfs is None:
            N_wfs = init_states._n_states

        # outer window
        if outer_window == "occupied":
            outer_window_type = "bands"  # optimally would like to use band indices

            # used in case inner window is defined by energy values
            outer_band_idxs = list(range(n_occ))
            outer_band_energies = energies[..., outer_band_idxs]
            outer_energies = [
                np.argmin(outer_band_energies),
                np.argmax(outer_band_energies),
            ]

            # mask out states outside outer window
            nan = np.empty(unk_states.shape)
            nan.fill(np.nan)
            states_sliced = np.where(
                np.logical_and(
                    energies[..., np.newaxis] >= outer_energies[0],
                    energies[..., np.newaxis] <= outer_energies[1],
                ),
                unk_states,
                nan,
            )
            mask_outer = np.isnan(states_sliced)
            masked_outer_states = np.ma.masked_array(states_sliced, mask=mask_outer)

        elif list(outer_window.keys())[0].lower() == "bands":
            outer_window_type = "bands"

            # used in case inner window is defined by energy values
            outer_band_idxs = list(outer_window.values())[0]
            outer_band_energies = energies[..., outer_band_idxs]
            outer_energies = [
                np.argmin(outer_band_energies),
                np.argmax(outer_band_energies),
            ]

            # mask out states outside outer window
            nan = np.empty(unk_states.shape)
            nan.fill(np.nan)
            states_sliced = np.where(
                np.logical_and(
                    energies[..., np.newaxis] >= outer_energies[0],
                    energies[..., np.newaxis] <= outer_energies[1],
                ),
                unk_states,
                nan,
            )
            mask_outer = np.isnan(states_sliced)
            masked_outer_states = np.ma.masked_array(states_sliced, mask=mask_outer)

        elif list(outer_window.keys())[0].lower() == "energy":
            outer_window_type = "energy"

            # energy window
            outer_energies = np.sort(list(outer_window.values())[0])

            # mask out states outside outer window
            nan = np.empty(unk_states.shape)
            nan.fill(np.nan)
            states_sliced = np.where(
                np.logical_and(
                    energies[..., np.newaxis] >= outer_energies[0],
                    energies[..., np.newaxis] <= outer_energies[1],
                ),
                unk_states,
                nan,
            )
            mask_outer = np.isnan(states_sliced)
            masked_outer_states = np.ma.masked_array(states_sliced, mask=mask_outer)

        # inner window
        if inner_window is None:
            N_inner = 0
            inner_window_type = outer_window_type
            inner_band_idxs = None

        elif list(inner_window.keys())[0].lower() == "bands":
            inner_window_type = "bands"

            inner_band_idxs = list(inner_window.values())[0]
            inner_band_energies = energies[..., inner_band_idxs]
            inner_energies = [
                np.argmin(inner_band_energies),
                np.argmax(inner_band_energies),
            ]

            # used in case outer window is energy dependent
            nan = np.empty(unk_states.shape)
            nan.fill(np.nan)
            states_sliced = np.where(
                np.logical_and(
                    energies[..., np.newaxis] >= inner_energies[0],
                    energies[..., np.newaxis] <= inner_energies[1],
                ),
                unk_states,
                nan,
            )
            mask_inner = np.isnan(states_sliced)
            masked_inner_states = np.ma.masked_array(states_sliced, mask=mask_inner)
            inner_states = masked_inner_states

        elif list(inner_window.keys())[0].lower() == "energy":
            inner_window_type = "energy"

            inner_energies = np.sort(list(inner_window.values())[0])

            nan = np.empty(unk_states.shape)
            nan.fill(np.nan)
            states_sliced = np.where(
                np.logical_and(
                    energies[..., np.newaxis] >= inner_energies[0],
                    energies[..., np.newaxis] <= inner_energies[1],
                ),
                unk_states,
                nan,
            )
            mask_inner = np.isnan(states_sliced)
            masked_inner_states = np.ma.masked_array(states_sliced, mask=mask_inner)
            inner_states = masked_inner_states
            N_inner = (~inner_states.mask).sum(axis=(-1, -2)) // n_orb

        if inner_window_type == "bands" and outer_window_type == "bands":
            # defer to the faster function
            return self.find_optimal_subspace_bands(
                N_wfs=N_wfs,
                inner_bands=inner_band_idxs,
                outer_bands=outer_band_idxs,
                iter_num=iter_num,
                verbose=verbose,
                tol=tol,
                alpha=alpha,
            )

        # minimization manifold
        if inner_window is not None:
            # states in outer manifold and outside inner manifold
            min_mask = ~np.logical_and(~mask_outer, mask_inner)
            min_states = np.ma.masked_array(unk_states, mask=min_mask)
            min_states = np.ma.filled(min_states, fill_value=0)
        else:
            min_states = masked_outer_states
            min_states = np.ma.filled(min_states, fill_value=0)

        # number of states to keep in minimization subspace
        if inner_window is None:
            # keep all the states from minimization manifold
            num_keep = np.full(min_states.shape, N_wfs)
            keep_mask = np.arange(min_states.shape[-2]) >= num_keep
            keep_mask = np.swapaxes(keep_mask, axis1=-1, axis2=-2)
        else:  # n_inner is k-dependent when using energy window
            N_inner = (~inner_states.mask).sum(axis=(-1, -2)) // n_orb
            num_keep = N_wfs - N_inner  # matrix of integers
            keep_mask = np.arange(min_states.shape[-2]) >= (
                num_keep[:, :, np.newaxis, np.newaxis]
            )
            keep_mask = keep_mask.repeat(min_states.shape[-2], axis=-2)
            keep_mask = np.swapaxes(keep_mask, axis1=-1, axis2=-2)

        # N_min = (~min_states.mask).sum(axis=(-1,-2))//n_orb
        # N_outer = (~masked_outer_states.mask).sum(axis=(-1,-2))//n_orb

        # Assumes only one shell for now
        w_b, _, idx_shell = self.k_mesh.get_weights(N_sh=1)
        num_nnbrs = self.k_mesh.num_nnbrs
        bc_phase = self.k_mesh.bc_phase

        # Projector of initial tilde subspace at each k-point
        P = init_states.get_projector()
        P_nbr, Q_nbr = init_states.get_nbr_projector(return_Q=True)
        T_kb = np.zeros((*nks, num_nnbrs), dtype=complex)
        for idx, idx_vec in enumerate(idx_shell[0]):  # nearest neighbors
            T_kb[..., idx] = np.trace(
                P[..., :, :] @ Q_nbr[..., idx, :, :], axis1=-1, axis2=-2
            )
        P_min = np.copy(P)  # for start of iteration
        P_nbr_min = np.copy(P_nbr)  # for start of iteration
        Q_nbr_min = np.copy(Q_nbr)  # for start of iteration

        omega_I_prev = (1 / Nk) * w_b[0] * np.sum(T_kb)

        #### Start of minimization iteration ####
        for i in range(iter_num):
            P_avg = np.sum(w_b[0] * P_nbr_min, axis=-3)
            Z = min_states.conj() @ P_avg @ np.transpose(min_states, axes=(0, 1, 3, 2))
            # masked entries correspond to subspace spanned by states outside min manifold
            Z = np.ma.filled(Z, fill_value=0)

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
            keep_states = np.ma.masked_array(states_min, mask=keep_mask)
            keep_states = np.ma.filled(keep_states, fill_value=0)
            # need to concatenate with frozen states

            P_new = np.einsum("...ni,...nj->...ij", keep_states, keep_states.conj())
            P_min = alpha * P_new + (1 - alpha) * P_min  # for next iteration
            for idx, idx_vec in enumerate(idx_shell[0]):  # nearest neighbors
                states_pbc = (
                    np.roll(keep_states, shift=tuple(-idx_vec), axis=(0, 1))
                    * bc_phase[..., idx, np.newaxis, :]
                )
                P_nbr_min[..., idx, :, :] = np.einsum(
                    "...ni, ...nj->...ij", states_pbc, states_pbc.conj()
                )
                Q_nbr_min[..., idx, :, :] = np.eye(n_orb) - P_nbr_min[..., idx, :, :]
                T_kb[..., idx] = np.trace(
                    P_min[..., :, :] @ Q_nbr_min[..., idx, :, :], axis1=-1, axis2=-2
                )

            omega_I_new = (1 / Nk) * w_b[0] * np.sum(T_kb)
            diff = omega_I_prev - omega_I_new
            omega_I_prev = omega_I_new

            if verbose and diff > 0:
                print("Warning: Omega_I is increasing.")

            if verbose:
                print(f"{i} Omega_I: {omega_I_new.real}")

            if abs(diff) * (iter_num - i) <= tol:
                # assuming the change in omega_i monatonically decreases at this rate,
                # omega_i will not change more than tolerance with remaining steps
                print("Omega_I has converged within tolerance. Breaking loop")
                if inner_window is not None:
                    min_keep = np.ma.masked_array(keep_states, mask=keep_mask)
                    subspace = np.ma.concatenate((min_keep, inner_states), axis=-2)
                    subspace_sliced = subspace[np.where(~subspace.mask)]
                    subspace_sliced = subspace_sliced.reshape((*nks, N_wfs, n_orb))
                    subspace_sliced = np.array(subspace_sliced)
                    return subspace_sliced
                else:
                    return keep_states

        # loop has ended
        if inner_window is not None:
            min_keep = np.ma.masked_array(keep_states, mask=keep_mask)
            subspace = np.ma.concatenate((min_keep, inner_states), axis=-2)
            subspace_sliced = subspace[np.where(~subspace.mask)]
            subspace_sliced = subspace_sliced.reshape((*nks, N_wfs, n_orb))
            subspace_sliced = np.array(subspace_sliced)
            return subspace_sliced
        else:
            return keep_states

    def find_optimal_subspace_bands(
        self,
        N_wfs=None,
        inner_bands=None,
        outer_bands="occupied",
        iter_num=100,
        verbose=False,
        tol=1e-10,
        alpha=1,
    ):
        """Finds the subspaces throughout the BZ that minimizes the gauge-independent spread.

        Used when the inner and outer windows correspond to bands rather than energy values. This function
        is faster when compared to energy windows. By specifying bands, the arrays have fixed sizes at each k-point
        and operations can be vectorized with numpy.
        """
        nks = self._nks
        Nk = np.prod(nks)
        n_orb = self.model._n_orb
        n_occ = int(n_orb / 2)

        # Assumes only one shell for now
        w_b, _, idx_shell = self.k_mesh.get_weights(N_sh=1)
        bc_phase = self.k_mesh.bc_phase

        # initial subspace
        init_states = self.tilde_states
        energy_eigstates = self.energy_eigstates
        u_wfs = energy_eigstates.get_states(flatten_spin=True)["Cell periodic"]
        # u_wfs_til = init_states.get_states(flatten_spin=True)["Cell periodic"]

        if N_wfs is None:
            # assume number of states in the subspace is number of tilde states
            N_wfs = init_states._n_states

        if outer_bands == "occupied":
            outer_bands = list(range(n_occ))

        outer_states = u_wfs.take(outer_bands, axis=-2)

        # Projector of initial tilde subspace at each k-point
        if inner_bands is None:
            N_inner = 0
            P = init_states.get_projector()
            P_nbr, Q_nbr = init_states.get_nbr_projector(return_Q=True)
            T_kb = np.einsum("...ij, ...kji -> ...k", P, Q_nbr)
        else:
            N_inner = len(inner_bands)
            inner_states = u_wfs.take(inner_bands, axis=-2)

            P_inner = np.swapaxes(inner_states, -1, -2) @ inner_states.conj()
            Q_inner = np.eye(P_inner.shape[-1]) - P_inner
            P_tilde = self.tilde_states.get_projector()

            # chosing initial subspace as highest eigenvalues
            MinMat = Q_inner @ P_tilde @ Q_inner
            eigvals, eigvecs = np.linalg.eigh(MinMat)
            min_states = np.einsum(
                "...ij, ...ik->...jk", eigvecs[..., -(N_wfs - N_inner) :], outer_states
            )

            P = np.swapaxes(min_states, -1, -2) @ min_states.conj()
            states_pbc_all = np.empty(
                (*min_states.shape[:-2], len(idx_shell[0]), *min_states.shape[-2:]),
                dtype=min_states.dtype,
            )
            for idx, idx_vec in enumerate(idx_shell[0]):  # nearest neighbors
                states_pbc_all[..., idx, :, :] = (
                    np.roll(min_states, shift=tuple(-idx_vec), axis=(0, 1))
                    * bc_phase[..., idx, np.newaxis, :]
                )
            P_nbr = np.swapaxes(states_pbc_all, -1, -2) @ states_pbc_all.conj()
            Q_nbr = np.eye(n_orb) - P_nbr
            T_kb = np.einsum("...ij, ...kji -> ...k", P, Q_nbr)

        P_min = np.copy(P)  # for start of iteration
        P_nbr_min = np.copy(P_nbr)  # for start of iteration

        # manifold from which we borrow states to minimize omega_i
        comp_bands = list(np.setdiff1d(outer_bands, inner_bands))
        comp_states = u_wfs.take(comp_bands, axis=-2)

        omega_I_prev = (1 / Nk) * w_b[0] * np.sum(T_kb)

        for i in range(iter_num):
            # states spanning optimal subspace minimizing gauge invariant spread
            P_avg = w_b[0] * np.sum(P_nbr_min, axis=-3)
            Z = comp_states.conj() @ P_avg @ np.swapaxes(comp_states, -1, -2)
            eigvals, eigvecs = np.linalg.eigh(Z)  # [val, idx]
            evecs_keep = eigvecs[..., -(N_wfs - N_inner) :]
            states_min = np.swapaxes(evecs_keep, -1, -2) @ comp_states

            P_new = np.swapaxes(states_min, -1, -2) @ states_min.conj()

            states_pbc_all = np.empty(
                (*states_min.shape[:-2], len(idx_shell[0]), *states_min.shape[-2:]),
                dtype=states_min.dtype,
            )
            for idx, idx_vec in enumerate(idx_shell[0]):  # nearest neighbors
                states_pbc_all[..., idx, :, :] = (
                    np.roll(states_min, shift=tuple(-idx_vec), axis=(0, 1))
                    * bc_phase[..., idx, np.newaxis, :]
                )
            P_nbr_new = np.swapaxes(states_pbc_all, -1, -2) @ states_pbc_all.conj()

            if alpha != 1:
                # for next iteration
                P_min = alpha * P_new + (1 - alpha) * P_min
                P_nbr_min = alpha * P_nbr_new + (1 - alpha) * P_nbr_min
            else:
                # for next iteration
                P_min = P_new
                P_nbr_min = P_nbr_new

            if verbose:
                Q_nbr_min = np.eye(n_orb * self.model._nspin) - P_nbr_min
                T_kb = np.einsum("...ij, ...kji -> ...k", P_min, Q_nbr_min)
                omega_I_new = (1 / Nk) * w_b[0] * np.sum(T_kb)

                if omega_I_new > omega_I_prev:
                    print("Warning: Omega_I is increasing.")
                    alpha = max(alpha - 0.1, 0)

                if abs(omega_I_prev - omega_I_new) * (iter_num - i) <= tol:
                    # omega_i will not change by more than tol with remaining steps (if monotonically decreases)
                    print("Omega_I has converged within tolerance. Breaking loop")
                    break

                print(f"{i} Omega_I: {omega_I_new.real}")
                omega_I_prev = omega_I_new

            else:
                if abs(np.amax(P_new - P_min)) <= tol:
                    print(np.amax(P_new - P_min))
                    break

        if inner_bands is not None:
            return_states = np.concatenate((inner_states, states_min), axis=-2)
            return return_states
        else:
            return states_min

    def mat_exp(self, M):
        eigvals, eigvecs = np.linalg.eig(M)
        U = eigvecs
        U_inv = np.linalg.inv(U)
        # Diagonal matrix of the exponentials of the eigenvalues
        exp_diagM = np.exp(eigvals)
        # Construct the matrix exponential
        expM = np.einsum(
            "...ij, ...jk -> ...ik",
            U,
            np.multiply(U_inv, exp_diagM[..., :, np.newaxis]),
        )
        return expM

    def find_min_unitary(
        self, eps=1e-3, iter_num=100, verbose=False, tol=1e-10, grad_min=1e-3
    ):
        """
        Finds the unitary that minimizing the gauge dependent part of the spread.

        Args:
            M: Overlap matrix
            eps: Step size for gradient descent
            iter_num: Number of iterations
            verbose: Whether to print the spread at each iteration
            tol: If difference of spread is lower that tol for consecutive iterations,
                the loop breaks

        Returns:
            U: The unitary matrix
        """
        M = self.tilde_states._M
        w_b, k_shell, idx_shell = self.k_mesh.get_weights()
        # Assumes only one shell for now
        w_b, k_shell, idx_shell = w_b[0], k_shell[0], idx_shell[0]
        nks = self._nks
        Nk = np.prod(nks)
        num_state = self.tilde_states._n_states

        U = np.zeros(
            (*nks, num_state, num_state), dtype=complex
        )  # unitary transformation
        U[...] = np.eye(num_state, dtype=complex)  # initialize as identity
        M0 = np.copy(M)  # initial overlap matrix
        M = np.copy(M)  # new overlap matrix

        # initializing
        omega_tilde_prev = self._get_Omega_til(M, w_b, k_shell)
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
            U = np.einsum("...ij, ...jk -> ...ik", U, self.mat_exp(eta * eps * G))

            for idx, idx_vec in enumerate(idx_shell):
                M[..., idx, :, :] = (
                    np.swapaxes(U, -1, -2).conj()
                    @ M0[..., idx, :, :]
                    @ np.roll(U, shift=tuple(-idx_vec), axis=(0, 1))
                )

            grad_mag = np.linalg.norm(np.sum(G, axis=(0, 1)))
            omega_tilde_new = self._get_Omega_til(M, w_b, k_shell)

            if verbose:
                print(f"{i} Omega_til = {omega_tilde_new.real}, Grad mag: {grad_mag}")

            if (
                abs(grad_mag) <= grad_min
                and abs(omega_tilde_prev - omega_tilde_new) * (iter_num - i) <= tol
            ):
                print(
                    "Omega_tilde minimization has converged within tolerance. Breaking the loop."
                )
                return U

            if grad_mag_prev < grad_mag and i != 0:
                if verbose:
                    print("Warning: Gradient increasing.")
                eps *= 0.9

            grad_mag_prev = grad_mag
            omega_tilde_prev = omega_tilde_new

        return U

    def subspace_selec(
        self,
        outer_window="occupied",
        inner_window=None,
        twfs=None,
        N_wfs=None,
        iter_num=1000,
        tol=1e-5,
        alpha=1,
        verbose=False,
    ):
        # if we haven't done single-shot projection yet (set tilde states)
        if twfs is not None:
            # initialize tilde states
            twfs = self.get_trial_wfs(twfs)

            n_occ = int(self.energy_eigstates._n_states / 2)  # assuming half filled
            band_idxs = list(range(0, n_occ))  # project onto occ manifold

            psi_til_init = self.get_psi_tilde(
                self.energy_eigstates._psi_wfs, twfs, state_idx=band_idxs
            )
            self.set_tilde_states(psi_til_init, cell_periodic=False)
        else:
            assert hasattr(
                self.tilde_states, "_u_wfs"
            ), "Need pass trial wavefunction list or initalize tilde states with single_shot()."

        # Minimizing Omega_I via disentanglement
        util_min = self.find_optimal_subspace(
            N_wfs=N_wfs,
            outer_window=outer_window,
            inner_window=inner_window,
            iter_num=iter_num,
            verbose=verbose,
            alpha=alpha,
            tol=tol,
        )

        self.set_tilde_states(util_min, cell_periodic=True)

        return

    def max_loc(self, eps=1e-3, iter_num=1000, tol=1e-5, grad_min=1e-3, verbose=False):

        U = self.find_min_unitary(
            eps=eps, iter_num=iter_num, verbose=verbose, tol=tol, grad_min=grad_min
        )

        u_tilde_wfs = self.tilde_states.get_states(flatten_spin=True)["Cell periodic"]
        util_max_loc = np.einsum("...ji, ...jm -> ...im", U, u_tilde_wfs)

        self.set_tilde_states(util_max_loc, cell_periodic=True)

        return

    def ss_maxloc(
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
        alpha=1,
        verbose=False,
    ):
        """Find the maximally localized Wannier functions using the projection method."""

        ### Subspace selection ###
        self.subspace_selec(
            outer_window=outer_window,
            inner_window=inner_window,
            twfs=twfs_1,
            N_wfs=N_wfs,
            iter_num=iter_num_omega_i,
            tol=tol_omega_i,
            alpha=alpha,
            verbose=verbose,
        )

        ### Second projection ###
        # if we need a smaller number of twfs b.c. of subspace selec
        if twfs_2 is not None:
            twfs = self.get_trial_wfs(twfs_2)
            psi_til_til = self.get_psi_tilde(
                self.tilde_states._psi_wfs,
                twfs,
                state_idx=list(range(self.tilde_states._psi_wfs.shape[2])),
            )
        # chose same twfs as in subspace selec
        else:
            psi_til_til = self.get_psi_tilde(
                self.tilde_states._psi_wfs,
                self.trial_wfs,
                state_idx=list(range(self.tilde_states._psi_wfs.shape[2])),
            )

        self.set_tilde_states(psi_til_til, cell_periodic=False)

        ### Finding optimal gauge with maxloc ###
        self.max_loc(
            eps=eps,
            iter_num=iter_num_omega_til,
            tol=tol_omega_til,
            grad_min=grad_min,
            verbose=verbose,
        )

        return

    def interp_energies(self, k_path, wan_idxs=None, ret_eigvecs=False, u_tilde=None):
        if u_tilde is None:
            # if self.model._nspin == 2:
            #     u_tilde = self.tilde_states.get_states(flatten_spin=True)["Cell periodic"]
            # else:
            u_tilde = self.tilde_states.get_states(flatten_spin=False)["Cell periodic"]
        if wan_idxs is not None:
            u_tilde = np.take_along_axis(u_tilde, wan_idxs, axis=-2)

        H_k = self.get_Bloch_Ham()
        if self.model._nspin == 2:
            new_shape = H_k.shape[:-4] + (2 * self.model._norb, 2 * self.model._norb)
            H_k = H_k.reshape(*new_shape)

        H_rot_k = u_tilde.conj() @ H_k @ np.swapaxes(u_tilde, -1, -2)
        eigvals, eigvecs = np.linalg.eigh(H_rot_k)
        eigvecs = np.einsum("...ij, ...ik->...kj", u_tilde, eigvecs)
        # eigvecs = np.swapaxes(eigvecs, -1, -2)

        k_mesh = self.k_mesh.square_mesh
        k_idx_arr = self.k_mesh.idx_arr
        nks = self.k_mesh.nks
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
                phase = np.exp(-1j * 2 * np.pi * np.vdot(k_mesh[k_idx], R_vec))
                # H_R[idx, :, :] += H_rot_k[k_idx] * phase / Nk
                # u_R[idx] += u_tilde[k_idx] * phase / Nk
                eval_R[idx] += eigvals[k_idx] * phase / Nk
                evecs_R[idx] += eigvecs[k_idx] * phase / Nk

        # interpolate to arbitrary k
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
            return eigvals

    def interp_op(self, O_k, k_path, plaq=False):
        return self.tilde_states.interp_op(O_k, k_path, plaq=plaq)

    def report(self):
        assert hasattr(
            self.tilde_states, "_u_wfs"
        ), "First need to set Wannier functions"
        print("Wannier function report")
        print(" --------------------- ")

        print("Quadratic spreads:")
        for i, spread in enumerate(self.spread):
            print(f"w_{i} --> {spread.round(5)}")
        print("Centers:")
        centers = self.get_centers()
        for i, center in enumerate(centers):
            print(f"w_{i} --> {center.round(5)}")
        print(rf"Omega_i = {self.omega_i}")
        print(rf"Omega_tilde = {self.omega_til}")

    def get_supercell(self, Wan_idx, omit_sites=None):
        w0 = self.WFs  # .reshape((*self.WFs.shape[:self.k_mesh.dim+1], -1))
        center = self.centers[Wan_idx]
        orbs = self.model._orb_vecs
        lat_vecs = self.model._lat_vecs

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
                wf_value = w0[tx, ty, Wan_idx, i]
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

                # Handle omit site if applicable
                if omit_sites is not None and i in omit_sites:
                    positions["omit"]["xs"].append(x)
                    positions["omit"]["ys"].append(y)
                    positions["omit"]["r"].append(rad)
                    positions["omit"]["wt"].append(wt)
                    # positions['omit']['phase'].append(phase)
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
