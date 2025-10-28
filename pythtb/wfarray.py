from .utils import _is_int, _offdiag_approximation_warning_and_stop
from .tbmodel import TBModel
from .mesh import Mesh
from .lattice import Lattice
import numpy as np
from functools import partial
import inspect
import warnings
import functools
import logging

logger = logging.getLogger(__name__)

__all__ = ["WFArray"]

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


class WFArray:
    r"""Wavefunction container defined on a sampling mesh.

    A :class:`WFArray` stores states on a discrete mesh of k-points
    and/or adiabatic parameters :math:`\lambda`. Once populated, 
    it can be queried for Berry connections, Berry curvature, 
    Chern numbers, and other derived quantities, or passed to :class:`Wannier` 
    when constructing Wannier functions or obtaining smooth gauges from
    the projection method.

    The underlying :class:`Mesh` may represent a full Monkhorst–Pack grid,
    a reduced-dimensional path, or even a mesh that contains only
    parameter axes. In every case, :class:`WFArray` tracks the mesh layout,
    the stored states, and the necessary phase conventions so downstream
    utilities can consume the data consistently.

    The stored wavefunctions may be the Hamiltonian eigenstates returned
    by :meth:`solve_model`. Alternatively, you can inject states in an arbitrary
    gauge by calling :meth:`set_states` (or by assigning through the
    ``[...]`` operator) before applying the higher-level Berry/Wannier
    routines.

    Parameters
    ----------
    lattice : :class:`Lattice`
        Lattice structure. This provides information about the lattice vectors, the
        orbital positions, and the periodic directions. Used to determine the phase
        factors for Bloch <-> cell-periodic transformations. If the model is finite,
        this can be a lattice with zero k-dimensions, and the lattice vectors are 
        only used to define orbital positions.
    mesh : :class:`Mesh`
        Sampling grid. The number of k-type axes can be anything from zero (purely
        parametric sweep) to the full ``model.dim_k``; when it is smaller it is
        interpreted as a path or lower-dimensional slice.
    spinful : bool, optional
        Whether the model includes spin degrees of freedom (defaults to ``False``).
        This determines the shape of the stored wavefunction array. If ``True``,
        each orbital is assumed to have two spin components (`WFArray.nspin == 2`).
    nstates : int, optional
        Number of bands per mesh point to store (defaults to ``WFArray.norb * WFArray.nspin``).

    See Also
    --------
    :class:`pythtb.TBModel`
    :class:`pythtb.Mesh`
    :class:`pythtb.Wannier`
    :ref:`formalism`
    :ref:`haldane-bp-nb` : For an example of using :class:`WFArray` on a regular grid of points in k-space.
    :ref:`cone-nb` : For an example of using :class:`WFArray` on a non-regular grid of points in k-space.
    :ref:`three-site-thouless-nb` : For an example of using :class:`Mesh` with an adiabatic dimension.
        This example shows how one of the directions of :class:`WFArray` object need not be a k-vector direction, 
        but can instead be a Hamiltonian parameter :math:`\lambda`. See also discussion after equation 4.1 in
        :ref:`formalism`.
    :ref:`cubic-slab-hwf-nb` : For an example of using :class:`WFArray` to store hybrid Wannier functions.
    
    Notes
    -----
    .. important::
        - Some features are only defined for regular grids and/or in the energy eigenstate gauge. 
        - If populating
          manually, the user is responsible for ensuring that the proper periodic boundary conditions
          are met, and that the states correspond to the points on the ``mesh``. Otherwise, the results 
          may be inconsistent.
    .. tip::
      :class:`WFArray` cooperates with :class:`Wannier` to construct smooth Wannier gauges:
      pass the diagonalized array to ``Wannier(tb, wfarray)`` and use 
      :meth:`Wannier.single_shot_projection`.

    - Wavefunctions are always stored with mesh axes leading, followed by bands, orbital,
      and (if present) spin indices. Utility methods accept the same ordering.
    - :meth:`solve` automatically diagonalises the model on the mesh, applies
      periodic gauge fixes on closed k-loops, and seeds cached overlap matrices.
      When no k-axes are present the same container can still hold parameter-only
      eigenstates (useful for adiabatic/finite systems).

    Examples
    --------
    Populate a uniform Monkhorst-Pack grid and compute the Berry curvature

    >>> mesh = Mesh(dim_k=2, axis_types=['k', 'k'])
    >>> mesh.build_grid(shape=(20, 20), gamma_centered=True)
    >>> wfa = WFArray(tb_model, mesh)
    >>> wfa.solve()
    >>> curv = wfa.berry_curvature(non_abelian=False)

    Store a 1D parameter sweep (no k-axes)::

    >>> mesh = Mesh(dim_k=0, dim_lambda=1, axis_types=['l'])
    >>> mesh.build_grid(shape=(101,), lambda_start=0.0, lambda_stop=2*np.pi)
    >>> wfa = WFArray(tb_model, mesh, nstates=tb_model.nstate)
    >>> wfa.set_states(eigenvectors_lambda, is_cell_periodic=False)

    Access/replace a single mesh point::

    >>> wfa[i_kx, j_ky, ell] = eigenvectors  # shape (nstates, norb[, nspin])
    """

    def __init__(self, lattice: Lattice, mesh: Mesh, nstates: int = None, spinful: bool = False):
        if not isinstance(lattice, Lattice):
            raise TypeError("lattice must be of type pythtb.Lattice")
        
        self._lattice = lattice

        if not isinstance(mesh, Mesh):
            raise TypeError("mesh must be of type pythtb.Mesh")
        if self.dim_k != mesh.dim_k:
            raise ValueError(f"Model dim_k ({self.dim_k}) does not match mesh dim_k ({mesh.dim_k})")
        
        self._mesh = mesh

        if True in (np.array(self.shape_mesh, dtype=int) <= 1).tolist():
            raise ValueError(
                "Dimension of WFArray object in each direction must be 2 or larger.\n"
                "This is required for periodic boundary conditions (PBC) to be applied.\n"
                "Maybe you need to build the mesh first?"
            )
        
        if not isinstance(spinful, bool):
            raise TypeError("Argument spinful must be a boolean.")
        self._spinful = spinful

        if nstates is not None:
            if not _is_int(nstates):
                raise TypeError("Argument nstates must be an integer.")
            self._nstates = nstates
        else:
            self._nstates = self.norb * self.nspin  # Default to total number of bands
       
        # wfs indexed by [k1, k2,..., state, orb, spin]
        self._wfs = np.empty(self.shape, dtype=complex)
        # energies indexed by [k1, k2,..., state]
        self._energies = None

    def __getitem__(self, index):
        self._check_index(index)
        return self._wfs[index]

    def __setitem__(self, index, value):
        self._check_index(index)
        if not isinstance(value, (list, np.ndarray)):
            raise TypeError("Value must be a list or numpy array!")
        
        value = np.array(value, dtype=complex)

        if self.nspin == 2:
            if value.ndim == self.naxes + 2:
                if value.shape[-1] != self.norb*2:
                    raise ValueError("Value shape does not match expected shape for spinful model!")
                value = value.reshape(*value.shape[:-1], self.norb, 2)

        else:
            if value.shape != self.shape[len(self.shape_mesh):]:
                raise ValueError("Incompatible shape for wavefunction!")

        self._wfs[index] = value
        self._invalidate_caches()

    def _check_index(self, index):
        # Normalize to a tuple of ints
        if isinstance(index, (tuple, list, np.ndarray)):
            if len(index) != self.naxes:
                raise TypeError(f"Index should be an integer or a tuple of length {self.naxes}.")

        if self.naxes == 1:
            if isinstance(index, (tuple, list, np.ndarray)):
                index = index[0]
            elif not _is_int(index):
                raise TypeError("Indices should be integers.")
            
            idxs = (int(index),)
        else:
            if not isinstance(index, (tuple, list, np.ndarray)):
               raise TypeError("Index should be a tuple, list, or ndarray.")
            if not all(_is_int(k) for k in index):
                raise TypeError("Index should be array-like of integers.")
            
            idxs = tuple(int(k) for k in index)

        for i, k in enumerate(idxs):
            lo, hi = -self.shape_mesh[i], self.shape_mesh[i]
            if k < lo or k >= hi:
                raise IndexError("Index outside the range of the WFArray.")

    def _invalidate_caches(self):
        for attr in ("_P", "_Q", "_P_nbr", "_Q_nbr", "_Mmn"):
            if hasattr(self, attr):
                delattr(self, attr)
    @property
    def model(self):
        """The :class:`TBModel` associated with the :class:`WFArray`."""
        return self._model

    @property
    def lattice(self) -> Lattice:
        """The :class:`Lattice` associated with the :class:`WFArray`."""
        return self._lattice

    @property
    def mesh(self):
        """The :class:`Mesh` associated with the :class:`WFArray`."""
        return self._mesh
    
    @property
    def filled(self) -> bool:
        """Whether the wavefunctions are filled (i.e., not empty)."""
        # if uninitialzed, wfs will be np.empty
        return self._wfs.size > 0

    @property
    def wfs(self) -> np.ndarray:
        """The (cell-periodic) wavefunctions."""
        return self._wfs

    @property
    def u_nk(self) -> np.ndarray:
        """The cell-periodic wavefunctions."""
        if not self.filled:
            raise ValueError("Wavefunctions are not initialized.")
        if self.dim_k == 0:
            raise ValueError("Cell-periodic wavefunctions are not defined for 0D k-space.")

        return getattr(self, "_u_nk", None)
    
    @property
    def cell_periodic(self) -> np.ndarray:
        """The cell-periodic wavefunctions."""
        return self.u_nk

    @property
    def psi_nk(self) -> np.ndarray:
        """The Bloch wavefunctions."""
        if not self.filled:
            raise ValueError("Wavefunctions are not initialized.")
        if self.dim_k == 0:
            raise ValueError("Bloch wavefunctions are not defined for 0D k-space.")
        
        return getattr(self, "_psi_nk", None)
    
    @property 
    def bloch_states(self) -> np.ndarray:
        """The Bloch wavefunctions."""
        return self.psi_nk
    
    @property
    def Mmn(self) -> np.ndarray:
        r"""The overlap matrix of the wavefunctions.

        The overlap matrix is defined as

        .. math::
            M_{mn}^{(\mathbf{b})}(\mathbf{k}) = \langle u_{m,\mathbf{k}} | u_{n,\mathbf{k}+\mathbf{b}} \rangle

        where :math:`\mathbf{b}` is a vector connecting nearest neighbor k-points in the mesh. Here, the 
        neighboring k-points are computed in Cartesian space. 

        Returns
        -------
        np.ndarray
            The overlap matrix of the wavefunctions. Shape is ``(shape_mesh..., nnbrs, nstates, nstates)``.

        Notes
        -----
        - The overlap matrix is only defined for regular grids.
        - The overlap matrix is not defined for 0D k-space.
        - To compute the overlap matrix using reduced neighbors, use :meth:`overlap_matrix`.
        """
        if not self.filled:
            raise ValueError("Wavefunctions are not initialized.")
        if not self.mesh.is_grid:
            raise ValueError("Overlap matrix is only defined for regular grids.")
        if self.dim_k == 0:
            raise ValueError("Overlap matrix is not defined for 0D k-space.")

        if not hasattr(self, "_Mmn"):
            self._Mmn = self.overlap_matrix(use_k_metric=True)

        return self._Mmn

    @property
    def energies(self) -> np.ndarray:
        """The band energies of the energy eigenstates of the :class:`TBModel`.
        
        Notes
        -----
        - The energies are only defined when the states stored in the :class:`WFArray` 
          are eigenstates of the Hamiltonian.
        """
        if not self.filled:
            raise ValueError("Wavefunctions are not initialized.")
        if self._energies is None:
            raise ValueError("Energies are not initialized. Use `solve_model` to compute them.")
        
        return self._energies

    @property
    def hamiltonian(self) -> np.ndarray:
        r"""The Hamiltonian defined on the :class:`Mesh`."""
        return getattr(self, "_H", None)

    @property
    def shape(self) -> tuple:
        """The shape of the state array."""
        wfs_dim = np.array(self.shape_mesh, dtype=int)
        wfs_dim = np.append(wfs_dim, self.nstates)
        wfs_dim = np.append(wfs_dim, self.norb)
        if self.nspin == 2:
            wfs_dim = np.append(wfs_dim, self.nspin)
        return tuple(wfs_dim)
    
    @property
    def nstates(self) -> int:
        """The number of states (or bands) in the state array."""
        return self._nstates

    @property
    def nspin(self) -> int:
        """The number of spin components."""
        return 2 if self.spinful else 1

    @property
    def spinful(self) -> bool:
        """Whether the :class:`WFArray` includes spin degrees of freedom."""
        return self._spinful

    @property
    def norb(self) -> int:
        """The number of orbitals defined in the :class:`Lattice`."""
        return self.lattice.norb
    
    @property
    def shape_mesh(self) -> tuple:
        """The shape of the :class:`Mesh`."""
        return self.mesh.shape_mesh
    
    @property
    def dim_k(self) -> int:
        """The dimension of k-space in the :class:`Mesh`."""
        return self.lattice.dim_k

    @property
    def dim_lambda(self) -> int:
        """The dimension of lambda space in the :class:`Mesh`."""
        return self.mesh.dim_lambda

    @property
    def naxes(self) -> int:
        """The number of axes in the :class:`Mesh`."""
        return self.mesh.num_axes
    
    @property
    def nks(self) -> tuple:
        """The number of points along each k-axis in the :class:`Mesh`."""
        return self.mesh.shape_k
    
    @property
    def shape_k(self) -> tuple:
        """The number of points along each k-axis in the :class:`Mesh`."""
        return self.mesh.shape_k
    
    @property
    def nlams(self) -> tuple:
        """The number of points along each lambda-axis in the :class:`Mesh`."""
        return self.mesh.shape_lambda

    @property
    def shape_lambda(self) -> tuple:
        """The number of points along each lambda-axis in the :class:`Mesh`."""
        return self.mesh.shape_lambda

    @property
    def k_points(self) -> np.ndarray:
        """The k-points in the :class:`Mesh`."""
        return self.mesh.get_k_points()
    
    @property
    def param_points(self) -> np.ndarray:
        """The parameter points in the :class:`Mesh`."""
        return self.mesh.get_param_points()
    
            
    def set_states(
        self, 
        wfs, 
        is_cell_periodic: bool = True, 
        is_spin_axis_flat: bool = False
        ):
        """Sets the wavefunctions in the *WFArray* object.

        This function is used to update the wavefunctions stored in the object.
        It is typically called internally after diagonalization. However, 
        it can also be called externally to manually set the wavefunctions.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        wfs : np.ndarray
            Wavefunctions to populate the mesh with. The shape must match the expected
            shape for the given mesh and spin configuration.
        is_cell_periodic : bool, optional
            If True, the wavefunctions are treated as cell-periodic (Bloch states).
            Default is True.
        is_spin_axis_flat : bool, optional
            If True, the spin and orbital indices are flattened into a single index.
            Default is False. This must match the shape of the input ``wfs``.

        Notes
        ------
        This function sets the Bloch and cell-periodic eigenstates as class attributes
        when `wfs` is defined on the a k-mesh. When the model is finite, only the
        ``.wfs`` attribute is set and ``is_cell_periodic`` argument is ignored.

        .. warning::
            This function should be used carefully to ensure that the wavefunctions 
            are consistent with the mesh and model.
            It is up to the user to ensure periodic boundary conditions and other mesh properties 
            are properly accounted for.
        """
        if not isinstance(wfs, np.ndarray):
            raise TypeError("wfs must be a numpy ndarray.")
        
        # Check the shape of wfs
        if is_spin_axis_flat and self.nspin == 2:
            if wfs.shape != self.shape_mesh + (self.nstates, self.norb * self.nspin):
                raise ValueError(
                    f"wfs shape {wfs.shape} does not match expected shape for flattened spin: "
                    f"{self.shape_mesh + (self.nstates, self.norb * self.nspin)}"
                )
            self._nstates = wfs.shape[-2]
        if not is_spin_axis_flat and self.nspin == 2:
            if wfs.shape != self.shape_mesh + (self.nstates, self.norb, self.nspin):
                raise ValueError(
                    f"wfs shape {wfs.shape} does not match expected shape for non-flattened spin: "
                    f"{self.shape_mesh + (self.nstates, self.norb, self.nspin)}"
                )
            self._nstates = wfs.shape[-3]
        elif self.nspin == 1:
            if wfs.shape != self.shape_mesh + (self.nstates, self.norb):
                raise ValueError(
                    f"wfs shape {wfs.shape} does not match expected shape for spinless model: "
                    f"{self.shape_mesh + (self.nstates, self.norb)}"
                )
            self._nstates = wfs.shape[-2]
            
        wfs = wfs.reshape(self.shape)

        # Compute phase factors for Bloch <-> cell-periodic transformation
        if self.dim_k > 0:
            if is_cell_periodic:
                phases = self._get_phases(inverse=False)
                psi_nk = wfs * phases
                self._u_nk = self._wfs = wfs
                self._psi_nk = psi_nk
            else:
                phases = self._get_phases(inverse=True)
                u_nk = wfs * phases
                self._u_nk = self._wfs = u_nk
                self._psi_nk = wfs
        
        else:
            if not is_cell_periodic:
                logger.warning("Setting non-cell-periodic wavefunctions for 0D k-space.")
            self._wfs = wfs

        self._invalidate_caches()
    

    def remove_states(self, state_idxs):
        r"""Remove states from the *WFArray* object.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        state_idxs : array-like of int
            Indices of the states to remove.
        """
        indices = np.array(state_idxs, dtype=int)
        if indices.ndim != 1:
            raise ValueError("Parameter indices must be a one-dimensional array.")
        if np.any(indices < 0) or np.any(indices >= self.nstates):
            raise ValueError("Indices out of bounds for the number of states.")
        
        n_states = indices.shape[0]

        if self.nspin == 2:
            state_ax = -2
        elif self.nspin == 1:
            state_ax = -1
        else:
            raise ValueError(
                "WFArray object can only handle spinless or spin-1/2 models."
            )

        self._wfs = np.delete(self._wfs, indices, axis=state_ax)
        self._nstates -= n_states

        self._energies = np.delete(self._energies, indices, axis=-1)
        if getattr(self, "_u_nk", None) is not None:
            self._u_nk = np.delete(self._u_nk, indices, axis=state_ax)
        if getattr(self, "_psi_nk", None) is not None:
            self._psi_nk = np.delete(self._psi_nk, indices, axis=state_ax)


    def choose_states(self, state_idxs):
        r"""Pick a subset of states to keep in the :class:`WFArray`.

        This method modifies the existing states in place to keep only 
        those specified by ``state_idxs``.

        Parameters
        ----------
        state_idxs : array-like of int
            Indices of states to keep.

            .. versionchanged:: 2.0.0
                Renamed from ``subset`` for consistency.

        Notes
        ------
        This modifies the shape of the ``.wfs``, ``.energies``, 
        ``.u_nk`` and ``.psi_nk`` arrays.

        Examples
        --------
        Make new *WFArray* object containing only two states

        >>> wf_new = wf.choose_states([3, 5])

        """
        remove_indices = np.setdiff1d(np.arange(self.nstates), state_idxs)
        self.remove_states(remove_indices)
    

    def empty_like(self, nstates=None) -> "WFArray":
        r"""Create a new :class:`WFArray` object with the same :class:`Lattice` and :class:`Mesh`.

        Parameters
        ----------
        nstates : int, optional
            Number of states for the new :class:`WFArray`. If None, uses the current number of states.

            .. versionchanged:: 2.0.0
                Renamed from ``nsta_arr`` for consistency with initialization.

        Returns
        -------
        WFArray
            A new :class:`WFArray` object with the same :class:`Lattice` and :class:`Mesh`.
        """
        # make a full copy of the WFArray
        wf_new = WFArray(self.lattice, self.mesh, nstates=nstates, nspin=self.nspin)
        return wf_new
    

    def get_k_shell(self, n_shell: int, report: bool = False):
        r"""Generates shells of k-points around the Gamma point.

        Returns array of vectors connecting the origin to nearest 
        neighboring k-points in the mesh. The vectors are expressed
        in inverse units of lattice vectors.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        n_shell : int
            Number of nearest neighbor shells to include.
        report : bool
            If True, prints a summary of the k-shell.

        Returns
        -------
        k_shell : list[np.ndarray[float]]
            List of :math:`\mathbf{b}` vectors in inverse units of lattice vectors
            connecting nearest neighbor k-mesh points. Shape is
            ``(n_shell, M_s, dim_k)`` where ``M_s`` is the number of k-points 
            in the s-th shell.
        idx_shell : list[np.ndarray[int]]
            Each entry is an array of integer shifts that takes a k-point 
            index in the mesh to its nearest neighbors. Shape is
            ``(n_shell, M_s, dim_k)`` where ``M_s`` is the number of k-points 
            in the s-th shell.
        """

        return self.lattice.nn_k_shell(self.nks, n_shell, report=report)

    def get_shell_weights(self, n_shell : int = 1, report: bool = False):
        r"""Generates the finite difference weights on a k-shell.

        This function uses the k-shells generated by :func:`get_k_shell` 
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

        .. versionadded:: 2.0.0

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
            connecting nearest neighbor k-mesh points. Length is ``n_shell``.
        idx_shell : list[np.ndarray[int]]
            Each entry is an array of integer shifts that takes a k-point 
            index in the mesh to its nearest neighbors.
            Length is ``n_shell``.
        """
        return self.lattice.k_shell_weights(self.nks, n_shell, report=report)
    
    def states(
            self, 
            state_idx=None,
            flatten_spin_axis: bool=False, 
            return_psi: bool=False
            ) -> np.ndarray:
        r"""Return cell-periodic (and optionally Bloch) states.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        state_idx : int, array-like, optional
            Index or indices of the states to return. If None, all states are returned.
        flatten_spin_axis : bool, optional
            If True, the spin and orbital indices are flattened into a single index.
            Default is False.
        return_psi : bool, optional
            If True, the function also returns the Bloch wavefunctions.

        Returns
        -------
        u_nk : np.ndarray
            Cell-periodic states (periodic in real space). Has shape ``(nk1, nk2,..., nl1, nl2,..., nstate, norb(,nspin))``.
            If ``flatten_spin_axis=True``, the last two axes are flattened into a single axis of size ``norb*nspin``.
        psi_nk : np.ndarray, optional
            Bloch states (periodic in k-space). Has shape ``(nk1, nk2,..., nl1, nl2,..., nstate, norb(,nspin))``.
            If ``flatten_spin_axis=True``, the last two axes are flattened into a single axis of size ``norb*nspin``.

        See Also
        --------
        :ref:`formalism`
        """
        if return_psi and self.dim_k == 0:
            raise ValueError("Bloch states are not defined for 0D k-space.")
        
        u = np.copy(self.wfs)
        psi = None if not return_psi else np.copy(self.psi_nk)

        if state_idx is not None:
            if _is_int(state_idx):
                state_idx = [state_idx]
            state_idx = np.array(state_idx, dtype=int)
            if state_idx.ndim != 1:
                raise ValueError("state_idx must be an integer or a one-dimensional array of integers.")
            if np.any(state_idx < 0) or np.any(state_idx >= self.nstates):
                raise IndexError("state_idx contains out-of-bounds indices.")
            
            if self.nspin == 2:
                u = u[..., state_idx, :, :]
                if psi is not None:
                    psi = psi[..., state_idx, :, :]
            else:
                u = u[..., state_idx, :]
                if psi is not None:
                    psi = psi[..., state_idx, :]

        if flatten_spin_axis and self.nspin == 2:
            u = u.reshape((*u.shape[:-2], -1))
            if psi is not None:
                psi = psi.reshape((*psi.shape[:-2], -1))

        return (u, psi) if return_psi else u

    def _nbr_projectors(self, return_Q=False):
        if self.dim_k == 0:
            raise NotImplementedError("Nearest neighbor projectors are not defined for 0D k-space.")
        if not self.mesh.is_grid:
            raise NotImplementedError("Mesh must be a grid to compute nearest neighbor projectors.")

        # Retrieve cached projectors if available
        P = getattr(self, "_P", None)
        if P is None:
            P = self.projectors(return_Q=False)

        P_nbr = getattr(self, "_P_nbr", None)
        Q_nbr = getattr(self, "_Q_nbr", None)
        if P_nbr is not None:
            if return_Q and Q_nbr is not None:
                return P_nbr, Q_nbr
            return P_nbr

        _, nnbr_idx_shell = self.get_k_shell(n_shell=1, report=False)
        num_nnbrs = nnbr_idx_shell[0].shape[0]

        P_nbr = np.zeros(
            (P.shape[:-2] + (num_nnbrs,) + P.shape[-2:]), dtype=complex
        )
        if return_Q:
            Q_nbr = np.zeros_like(P_nbr)

        for idx, idx_vec in enumerate(nnbr_idx_shell[0]):  # nearest neighbors
            u_shifted = self.roll_states_with_pbc(idx_vec, flatten_spin_axis=True)
            P = np.matmul(u_shifted.swapaxes(-2, -1), u_shifted.conj())
            P_nbr[..., idx, :, :] = P

            if return_Q:
                Q = np.eye(u_shifted.shape[-1]) - P
                Q_nbr[..., idx, :, :] = Q

        self._P_nbr = P_nbr

        if return_Q:
            self._Q_nbr = Q_nbr
            return P_nbr, Q_nbr
        return P_nbr

    def projectors(self, state_idx=None, return_Q=False):
        r"""Returns the band projectors associated with the states in the WFArray.

        The band projectors are defined as the outer product of the wavefunctions:

        .. math::

            P_{n\mathbf{k}} = \lvert u_{n\mathbf{k}}(\mathbf{r})\rangle \langle u_{n\mathbf{k}}(\mathbf{r}) \rvert, 
            \quad Q_{n\mathbf{k}} = \mathbb{I} - P_{n\mathbf{k}}
            
        .. versionadded:: 2.0.0
        
        Parameters
        ----------
        return_Q : bool, optional
            If True, the function also returns the orthogonal projector Q.

        Returns
        -------
        P : np.ndarray
            The band projectors.
        Q : np.ndarray, optional
            The orthogonal projectors.
        """
        # Return cached projectors if available and no specific state indices are requested
        if state_idx is None:
            if getattr(self, "_P", None) is not None:
                return self._P if not return_Q else (self._P, self._Q)

        u_nk = self.states(flatten_spin_axis=True)

        if state_idx is not None:
            u_nk = u_nk[..., state_idx, :]

        # band projectors
        P = np.matmul(u_nk.swapaxes(-2, -1), u_nk.conj())
        if state_idx is None:
            self._P = P
            self._Q = np.eye(u_nk.shape[-1]) - P

        if return_Q:
            Q = np.eye(u_nk.shape[-1]) - P
            return P, Q
        return P

    def solve_model(
            self, 
            model: TBModel = None,
            use_tensorflow: bool = False
            ):
        r"""Diagonalizes the :class:`TBModel` over the :class:`Mesh` points.

        Populates the state array with the eigenstates and eigenenergies of the Hamiltonian generated
        by the :class:`TBModel` on the points set in `Mesh`. 

        If the :class:`Mesh` has parametric dimensions, a ``model_func`` must be provided that returns
        the modified :class:`TBModel`. Some of the arguments in the ``model_func`` may be kept fixed by specifying 
        their names as keys and values as the values in the ``fixed_params`` dictionary.
        The keys of ``fixed_params`` must match parameter names of the ``model_func``.
        Additionally, if some of the parameters are left variable, these names of the parameters in the ``model_func``
        must also match the names of the :math:`\lambda` axes defined in the :class:`Mesh`. This will generate the 
        Hamiltonian at the values of the parameters specified by the mesh. See examples for more details.

        .. versionadded:: 2.0.0
            Replaces :meth:`solve_on_one_point` and :meth:`solve_on_grid`.

        Parameters
        ----------
        model : TBModel, optional
            A tight-binding model instance. If not provided, the model function
            must be used to generate the Hamiltonian.
        model_func : callable, optional
            A function that returns a :class:`TBModel` given a set of parameters.
            Only should be used if the mesh has parametric dimensions.
        fixed_params :  dict, optional
            A dictionary of fixed parameters to be passed to the model function.
            It has the structure ``{<param_name>: <param_value>, ...}``. 
            ``param_name`` must match the names of the parameters expected by 
            the model function.
        use_tensorflow : bool, optional
            If True, uses the `tf.linalg.eigh` function to diagonalize the Hamiltonian.
            This can be beneficial for large systems where GPU acceleration is available.
            This requires TensorFlow to be installed. Default is False.

        Notes
        ------
        - The eigenfunctions :math:`\psi_{n {\bf k}}` are by convention
          chosen to obey a periodic gauge, i.e.,
          :math:`\psi_{n,{\bf k+G}}=\psi_{n {\bf k}}` not only up to a
          phase, but they are also equal in phase. It follows that
          the cell-periodic Bloch functions are related by
          :math:`u_{n,{\bf k_0+G}}=e^{-i{\bf G}\cdot{\bf r}} u_{n {\bf k_0}}`.
          See :download:`notes on tight-binding formalism </misc/pythtb-formalism.pdf>` 
          section 4.4 and equation 4.18 for more detail.

        - This routine automatically finds the directions in the `Mesh` that include endpoints of
          the Brillouin zone, meaning the value of one of the components of the k-vector (`k_dir`) 
          differ at the beginning and end by a reciprocal lattice vector along that axis 
          (1 in reduced units). Periodic boundary conditions are then automatically 
          imposed. This sets the cell-periodic Bloch function at the end of the mesh in this direction
          equal to the first, multiplied by a phase factor. Explicitly, this means we set 
          :math:`u_{n,{\bf k_0+G}}=e^{-i{\bf G}\cdot{\bf r}} u_{n {\bf k_0}}` 
          for the corresponding reciprocal lattice vector :math:`\mathbf{G} = \mathbf{b}_{\texttt{k_dir}}`,
          where :math:`\mathbf{b}_{\texttt{k_dir}}` is the reciprocal lattice basis vector corresponding to the
          direction `k_dir`. The state :math:`u_{n{\bf k_0}}` is the state populated in the first element
          of the mesh along the `mesh_dir` axis.

        - When the `Mesh` grid includes endpoints in k-space, functions that compute derivatives
          with respect to k (e.g., Berry connection, Berry curvature, etc.) will automatically
          use finite difference formulas that account for periodic boundary conditions. Explicitly, 
          this means that the finite difference formula will include the overlap matrix element
          :math:`M_{mn}^{(\mathbf{b})}(\mathbf{k}) = \langle u_{m,\mathbf{k}} | u_{n,\mathbf{k}+\mathbf{b}} \rangle`
          that connects the states at the beginning and end of the mesh along the `mesh_dir` axis.

        Examples
        --------
        Say we have a parametric model function defined as follows:

        >>> def model_func(param1, param2):
        ...     lat_vecs = [[1, 0], [0, 1]]
        ...     orb_vecs = [[0,0], [0.5, 0.5]]
        ...     lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1])
        ...     model = TBModel(lattice=lat, nspin=1)
        ...     # Set model hoppings and onsite energies with parameters
        ...     return model

        The returned model will be 2D in k-space for this example.
        We want to vary ``param2``, and store the Hamiltonian for each value of ``param2``.
        First, we construct the ``Mesh`` by specifying the dimensions and axis types/names:

        >>> mesh = Mesh(dim_k=2, dim_lambda=1, axis_types=['k','k','l'], axis_names=['k1', 'k2', 'param2'])

        Note that we must name the last axis as ``param2`` to follow the name we set in the model function.
        We build the mesh values by using ``build_grid``. We will construct a uniform 2D grid in k-space of shape
        ``(20, 20)`` and a uniform 1D grid for ``param2`` with 5 points going from 0 to :math:`2\pi`.

        >>> mesh.build_grid(shape=(20, 20, 5), lambda_start=0, lambda_end=2*np.pi)

        To initialize the ``WFArray``, we generate a reference model with some set of parameters fixed. This
        is so that we can infer the lattice structure and orbital information from the model. We pass the reference 
        model and mesh to the ``WFArray`` constructor.

        >>> ref_model = model_func(0, 0)
        >>> wfa = WFArray(ref_model, mesh)

        Now say we want to keep ``param_1`` fixed to be 1. We can do this by setting the `fixed_params` argument 
        when calling `solve`.

        >>> wfa.solve(model_func=model_func, fixed_params={'param1': 1})

        The Hamiltonian now has the correct form with respect to the fixed parameters. The model is 
        spinless and has 2 orbitals, so the shape of the Hamiltonian is:

        >>> wfa.hamiltonian.shape
        (20, 20, 5, 2, 2)

        The eigenvalues and eigenstates are stored in the ``.energies`` and ``.wfs`` attributes, respectively.

        >>> wfa.energies.shape
        (20, 20, 5, 2)
        >>> wfa.wfs.shape
        (20, 20, 5, 2, 2)
        """
        if self.spinful != model.spinful:
            raise ValueError("Spinful setting of WFArray does not match the model.")
        
        # Gather parameter names and values from the mesh
        mesh_param_names = [ax.name for ax in self.mesh.lambda_axes]
        mesh_param_vals = [
            self.mesh.get_axis_range(i, j)
            for i, j in zip(self.mesh.lambda_axis_indices, self.mesh.lambda_component_indices)
        ]
        params = {name: val for name, val in zip(mesh_param_names, mesh_param_vals)}

        # Prepare k-points (flatten only the k-grid; λ stays leading)
        if self.dim_k > 0:
            n_kpts = int(np.prod(self.nks))
            k_flat = self.k_points.reshape(n_kpts, -1) # (Nk, dim_k)
        else:
            k_flat = None

        eigvals, eigvecs = model.solve_ham(
            k_pts=k_flat, params=params, return_eigvecs=True, flatten_spin_axis=True,
            tf_speedup=use_tensorflow
            )

        # Reshape eigenvectors to the full mesh shape
        eigvecs = eigvecs.reshape(*self.shape)
        eigvals = eigvals.reshape(*self.mesh.shape_mesh, self.nstates)
        
        # Set attributes
        self.set_states(eigvecs, is_cell_periodic=True, is_spin_axis_flat=False)
        self._energies = eigvals
        self._model = model
        # Smallest direct gap across all bands (if ≥ 2 bands)
        if self.nstates > 1:
            gaps = eigvals[..., 1:] - eigvals[..., :-1]
            self.gaps = gaps.min(axis=tuple(range(self.naxes)))
        else:
            self.gaps = None
            
        # Enforce PBCs along winding k-axes
        axes = self.mesh.axes
        for idx, ax in enumerate(axes):
            if ax.has_endpoint and ax.is_loop:
                endpt_comps = ax.endpoint_components
                if ax.winds_bz:
                    # These contain endpoints (k_i = 1 in reduced units)
                    bz_wind_comps = ax.winds_bz_components
                    overlap = endpt_comps and bz_wind_comps
                    for comp in overlap:
                        logger.debug(f"Imposing PBC in mesh direction {ax} for k-component {comp}")
                        self._impose_pbc(idx, comp)
                else:
                    logger.debug(f"Imposing loop in mesh direction {ax}")
                    self._impose_loop(idx)


    @deprecated("Use `solve` instead.")
    def solve_on_grid(self, start_k=None):
        r"""
        .. deprecated:: 2.0.0
            :meth:`solve_on_grid` has been deprecated. Use :meth:`solve_model` instead.
        """
        return
    
    @deprecated("Use `solve` instead.")
    def solve_on_one_point(self, kpt, mesh_indices):
        r"""
        .. deprecated:: 2.0.0
            :meth:`solve_on_one_point` has been deprecated. Use :meth:`solve_model` instead.
        """
        return
    
    def _get_phases(self, inverse=False):
        r"""Compute phase factors for converting between cell-periodic and Bloch wavefunctions.

        Parameters
        ----------
        inverse : bool, optional
            If True, compute phase factors for converting from Bloch to cell-periodic wavefunctions.
            If False, compute phase factors for converting from cell-periodic to Bloch wavefunctions.
            Defaults to False.

        Returns
        -------
        phases : np.ndarray
            Array of phase factors with shape [nk1, ..., nkd, nl1, ..., nlm, norb, (nspin)].
            The last dimension is present only if the model has spin.
        """
        lam = -1 if inverse else 1
        nks = self.nks
        flat_k_mesh = self.k_points.reshape(-1, self.dim_k)

        per_dir = list(range(flat_k_mesh.shape[-1]))
        # slice second dimension to only keep only periodic dimensions in orb
        per_orb = self.lattice.orb_vecs[:, per_dir]

        # compute a list of phase factors: exp(pm i k . tau) of shape [k_val, orbital]
        phases = np.exp(lam * 1j * 2 * np.pi * per_orb @ flat_k_mesh.T, dtype=complex).T

        # reshape phases to have shape: [nk1, nk2, ..., nkd, norb]
        phases = phases.reshape(*nks, self.norb)

        # broadcast along lambda
        n_lambda_axes = self.mesh.num_lambda_axes
        
        # Shape of wfs is (nk1, ..., nkN, nl1, ..., nlM, nstate, norb[, nspin])
        # Need to broadcast along all M lambda axes to direct multiply phases w/ wfs
        for i in range(n_lambda_axes):
            phases = phases[..., np.newaxis, :]

        # broadcast along state dimension (next to last)
        phases = phases[..., np.newaxis, :]
        
        # broadcast spin dimension (last)
        if self.nspin == 2:
            phases = phases[..., np.newaxis]

        return phases
    
    @staticmethod
    def _edge_slices(ax):
        """Helper function to get slices for the left and right edges of an axis."""
        # add one for Python counting and one for ellipses
        # Example ax = 2 (2 defines the axis in Python counting)
        slc_l = [slice(None)] * (ax + 2)  # e.g., [:, :, :, :]
        slc_r = [slice(None)] * (ax + 2)  # e.g., [:, :, :, :]
        # last element along mesh_dir axis
        slc_l[ax] = -1 # e.g., [:, :, -1, :]
        # first element along mesh_dir axis
        slc_r[ax] = 0 # e.g., [:, :, 0, :]
        # take all components of remaining axes with ellipses
        slc_l[ax + 1] = Ellipsis # e.g., [:, :, -1, ...]
        slc_r[ax + 1] = Ellipsis # e.g., [:, :, 0, ...]
        return tuple(slc_l), tuple(slc_r)
    
    
    def _get_pbc_phases(self, mesh_dir, k_dir):
        r"""Compute phase factors for periodic boundary conditions in forward direction.

        Parameters
        ----------
        mesh_dir : int
            Direction of the `WFArray` along which periodic boundary conditions are imposed.
        k_dir : int
            Direction of the k-vector in the Brillouin zone corresponding to `mesh_dir`.

        Returns
        -------
        phases : np.ndarray
            Array of phase factors with shape [nk1, ..., nkd, norb, (nspin)].
            The last dimension is present only if the model has spin.
        """

        if k_dir not in self.lattice.periodic_dirs:
            raise Exception(
                "Periodic boundary condition can be specified only along periodic directions!"
            )

        if not _is_int(mesh_dir):
            raise TypeError("mesh_dir should be an integer!")
        if mesh_dir < 0 or mesh_dir >= self.naxes:
            raise IndexError("mesh_dir outside the range!")
        
        orb_vecs = self.lattice.orb_vecs
        # Compute phase factors from orbital vectors dotted with G parallel to k_dir
        phase = np.exp(-2j * np.pi * orb_vecs[:, k_dir])
        phase = phase if self.nspin == 1 else phase[:, np.newaxis]
    
        # mesh_dir is the direction of the array along which we impose pbc
        # and it is also the direction of the k-vector along which we
        # impose pbc e.g.
        # mesh_dir=0 corresponds to kx, mesh_dir=1 to ky, etc.
        # mesh_dir=2 corresponds to lambda, etc.

        slc_lft, slc_rt = self._edge_slices(mesh_dir)
        return phase, slc_lft, slc_rt
    
    @deprecated(
            "Periodic boundary conditions are "
            "now imposed automatically when calling `solve` if the mesh includes endpoints in k-space.\n"
    )
    def impose_pbc(self, mesh_dir: int, k_dir: int):
        r"""
        .. deprecated:: 2.0.0

            Periodic boundary conditions are now imposed automatically when calling `solve` if the mesh includes endpoints in k-space.
            Previously, this was done manually by calling `impose_pbc`, meaning the wavefunction at the
            last point along the mesh direction was set equal to the first point, multiplied by a phase factor.
            Including 1 in the reduced coordinates of the k-vector along a given axis automatically triggers the imposition of periodic boundary 
            conditions along that axis. 
        """
        return

    def _impose_pbc(self, mesh_dir: int, k_dir: int):
        r"""Impose periodic boundary conditions on the WFArray.

        This routine sets the cell-periodic Bloch function
        at the end of the mesh in direction `k_dir` equal to the first,
        multiplied by a phase factor, overwriting the previous value.
        Explicitly, this means we set
        :math:`u_{n,{\bf k_0+G}}=e^{-i{\bf G}\cdot{\bf r}} u_{n {\bf k_0}}` for the
        corresponding reciprocal lattice vector :math:`\mathbf{G} = \mathbf{b}_{\texttt{k_dir}}`,
        where :math:`\mathbf{b}_{\texttt{k_dir}}` is the reciprocal lattice basis vector corresponding to the
        direction `k_dir`. The state :math:`u_{n{\bf k_0}}` is the state populated in the first element
        of the mesh along the `mesh_dir` axis.

        Parameters
        ----------
        mesh_dir : int
            Direction of `WFArray` along which you wish to impose periodic boundary conditions.

        k_dir : int
            Corresponding to the periodic k-vector direction
            in the Brillouin zone of the underlying *TBModel*. Since
            version 1.7.0 this parameter is defined so that it is
            specified between 0 and *dim_r-1*.


        Notes
        -----
        This function will impose these periodic boundary conditions along
        one direction of the array. We are assuming that the k-point
        mesh increases by exactly one reciprocal lattice vector along
        this direction. 

        Examples
        --------
        Imposes periodic boundary conditions along the mesh_dir=0
        direction of the `WFArray` object, assuming that along that
        direction the `k_dir=1` component of the k-vector is increased
        by one reciprocal lattice vector.  This could happen, for
        example, if the underlying TBModel is two dimensional but
        `WFArray` is a one-dimensional path along :math:`k_y` direction.

        >>> wf.impose_pbc(mesh_dir=0, k_dir=1)

        """
        if self.dim_k == 0:
            raise ValueError(
                "Cannot impose periodic boundary conditions in 0D k-space.\n"
                "Use `_impose_loop` instead."
                )
        if k_dir not in self.lattice.periodic_dirs:
            raise ValueError(
                "Periodic boundary condition can be specified only along periodic directions!"
            )

        phase, slc_lft, slc_rt = self._get_pbc_phases(mesh_dir, k_dir)

        # Set the last point along mesh_dir axis equal to first
        # multiplied by the phase factor
        self._wfs[slc_lft] = self._wfs[slc_rt] * phase

        if self.u_nk is not None:
            # Set the last point along mesh_dir axis equal to first
            # multiplied by the phase factor
            self._u_nk[slc_lft] = self._u_nk[slc_rt] * phase
            self._psi_nk[slc_lft] = self._psi_nk[slc_rt]


    @deprecated("Using `solve` is sufficient to set the wavefunction at the end equal to the beginning with equal phase.")    
    def impose_loop(self, mesh_dir):
        r"""
        .. deprecated:: 2.0.0

            This function has been deprecated.
            This routine was used to set the eigenvectors equal (with equal phase) at the beginning
            and end of the mesh along the `mesh_dir` direction
            by replacing the last eigenvector with the first one along the `mesh_dir` direction
            (for each band). By using a `Mesh` that includes loops, if the Hamiltonian
            is the same at the beginning and end of the loop, the states will be equal
            (with equal phase) automatically when calling `solve`.
        """
        return

    def _impose_loop(self, mesh_dir):
        r"""Impose a loop condition along a given mesh direction.

        This routine can be used to set the
        eigenvectors equal (with equal phase), by replacing the last
        eigenvector with the first one along the `mesh_dir` direction
        (for each band).

        Parameters
        ----------
        mesh_dir: int
            Direction of `WFArray` along which you wish to
            impose periodic boundary conditions.

        See Also
        --------
        :func:`pythtb.WFArray.impose_pbc`

        Notes
        -----
        This routine should not be used if the first and last points
        are related by a reciprocal lattice vector; in that case,
        :func:`pythtb.WFArray.impose_pbc` should be used instead.

        It is assumed that the first and last points along the
        `mesh_dir` direction correspond to the same Hamiltonian (this
        is **not** checked).

        Examples
        --------
        Suppose the WFArray object is three-dimensional
        corresponding to `(kx, ky, lambda)` where `(kx, ky)` are
        wavevectors of a 2D insulator and lambda is an
        adiabatic parameter that goes around a closed loop.
        Then to insure that the states at the ends of the lambda
        path are equal (with equal phase) in preparation for
        computing Berry phases in lambda for given `(kx, ky)`,
        do 

        >>> wf._impose_loop(mesh_dir = 2)
        """
        if not _is_int(mesh_dir):
            raise TypeError("mesh_dir must be an integer.")
        if mesh_dir < 0 or mesh_dir >= self.naxes:
            raise ValueError(
                f"mesh_dir must be between 0 and {self.naxes-1}, got {mesh_dir}."
            )
        if mesh_dir in self.mesh.k_axes and self.mesh.is_k_torus:
            raise ValueError("Cannot impose loop condition on periodic k-space axis.")

        slc_lft, slc_rt = self._edge_slices(mesh_dir)
        self._wfs[slc_lft] = self._wfs[slc_rt]

        if self.dim_k > 0:
            if self.u_nk is not None:
                self._u_nk[slc_lft] = self._u_nk[slc_rt]
            if self.psi_nk is not None:
                self._psi_nk[slc_lft] = self._psi_nk[slc_rt]

    def _unit_shift(self, axis: int):
        """Return an integer shift vector with +1 along *axis* over sampling axes."""
        v = [0] * self.naxes
        v[axis] = 1
        return v

    @staticmethod
    def _bounded_shift(A: np.ndarray, axis: int, sh: int) -> np.ndarray:
        """Shift array A by *sh* along *axis* without wrapping; fill vacated slab with zeros."""
        if sh == 0:
            return A
        sl_all = [slice(None)] * A.ndim
        B = np.zeros_like(A)
        if sh > 0:
            sl_src = sl_all.copy(); sl_dst = sl_all.copy()
            sl_src[axis] = slice(0, -sh)
            sl_dst[axis] = slice(sh, None)
        else:  # sh < 0
            shn = -sh
            sl_src = sl_all.copy(); sl_dst = sl_all.copy()
            sl_src[axis] = slice(shn, None)
            sl_dst[axis] = slice(0, -shn)
        B[tuple(sl_dst)] = A[tuple(sl_src)]
        return B

    def _boundary_phase_for_shift(self, shift_vec):
        """Compute exp(-i G dot r) mask for a multi-axis integer shift.

        The returned array is broadcast to match the stored state tensor shape
        (including lambda axes and the state axis). For spinful models, it
        is also broadcast over the spin axis.
        """
        nks = np.array(self.nks)
        dim_k = nks.size
        if dim_k == 0:
            return np.array(1.0, dtype=complex)

        mesh = self.mesh

        # Use shift only on sampling axes that are *periodic*
        k_shift = []
        k_ax_idx = mesh.k_axis_indices
        for local_k_axis, mesh_axis in enumerate(k_ax_idx):
            sh = int(shift_vec[mesh_axis])
            if mesh.is_axis_bz_winding(mesh_axis):
                k_shift.append(sh)
            elif mesh.is_axis_closed(mesh_axis):
                logger.info(f"Axis {mesh_axis} is closed; removing shift.")
                k_shift.append(0)
            else:
                logger.info(f"Axis {mesh_axis} is not winding bz or is closed; removing shift.")
                k_shift.append(0)
        k_shift = np.array(k_shift, dtype=int)

        # Integer index grid over k-axes: shape (*nks, dim_k)
        idx_grid = np.stack(np.meshgrid(*[np.arange(n) for n in nks], indexing="ij"), axis=-1)
        shifted = idx_grid + k_shift  # (*nks, dim_k)

        # Detect wraps per k-axis
        mask_pos = shifted >= nks  # crossed +BZ face on that k-axis
        mask_neg = shifted < 0     # crossed -BZ face on that k-axis
        G_axes = mask_pos.astype(np.int8) - mask_neg.astype(np.int8)  # (*nks, dim_k)

        # Map sampling-axis windings to k-components via mesh topology mask.
        # Build mapping M (dim_k_axes × dim_k_components)
        M = np.zeros((dim_k, dim_k), dtype=int)
        for local_k_axis, mesh_axis in enumerate(k_ax_idx):
            for c in range(dim_k):
                if mesh.is_axis_bz_winding(mesh_axis, c):
                    M[local_k_axis, c] = 1

        # Project signed wraps from k-axes to k-components
        G_comp = np.einsum('...i, ic -> ...c', G_axes, M, dtype=int)

        # Dot with orbital positions τ (reduced coords), shape (norb, dim_k)
        # Use only the real-space components that actually correspond to k-space
        # directions, in the *model.per* order. This matters when dim_k < dim_r or
        # when the periodic axes are not the first dim_k Cartesian components.
        per = getattr(self.lattice, "periodic_dirs", None)
        if per is None:
            # Fallback to the first dim_k components, but warn because this may be wrong
            if self.dim_k != self.lattice.dim_r:
                logger.warning(
                    "WFArray._boundary_phase_for_shift: lattice.periodic_dirs is missing; "
                    "falling back to first dim_k real-space components."
                )
            orb = self.lattice.orb_vecs[:, :dim_k]
        else:
            per = np.asarray(per, dtype=int)
            if per.size < dim_k:
                raise ValueError(
                    f"model.per lists only {per.size} periodic directions but dim_k={dim_k}."
                )
            # Select exactly dim_k periodic components in the order used for k-space
            orb = self.lattice.orb_vecs[:, per[:dim_k]]

        dot = np.tensordot(G_comp, orb, axes=([G_comp.ndim-1],[1]))  # (*nks, norb)

        phase = np.exp(-2j * np.pi * dot)

        # Shape of wfs is (nk1, ..., nkN, nl1, ..., nlM, nstate, norb[, nspin])
        # Need to broadcast along all M lambda axes to direct multiply phases w/ wfs
        for _ in range(self.mesh.num_lambda_axes):
            phase = phase[..., np.newaxis, :]

        # broadcast along state dimension (next to last)
        phase = phase[..., np.newaxis, :]
        
        # broadcast spin dimension (last)
        if self.nspin == 2:
            phase = phase[..., np.newaxis]

        return phase.astype(complex)
    
    def _invalidate_boundary_links(self, array: np.ndarray, shift_vec) -> np.ndarray:
        """Stamp NaNs on slabs where a neighbour does not exist for the given shift."""
        mesh = self.mesh
        ndims = self.naxes

        if not isinstance(shift_vec, (tuple, list, np.ndarray)):
            shift_vec = (shift_vec,)

        for axis, shift in enumerate(shift_vec):
            if axis >= ndims or shift == 0:
                continue

            wraps = mesh.is_axis_looped(axis) or mesh.is_axis_bz_winding(axis)
            closed = mesh.is_axis_closed(axis)
            if wraps and not closed:
                continue

            boundary_index = -1 if shift > 0 else 0
            slicer = [slice(None)] * array.ndim
            slicer[axis] = boundary_index
            array[tuple(slicer)] = np.nan + 0j

        return array

    #TODO: Fix examples
    def roll_states_with_pbc(
            self, 
            shift_vec: list[int], 
            flatten_spin_axis: bool = True, 
            strip_boundary: bool = False
            ):
        """Roll states with periodic boundary conditions.

        This method rolls the wavefunction states according to the given shift vector,
        applying the appropriate boundary phases to axes that have periodic boundary 
        conditions.

        Parameters
        ----------
        shift_vec : list[int]
            List of integer shifts for each axis.
        flatten_spin_axis : bool, optional
            Whether to flatten the spin axis into the orbital axis, by default True.
        strip_boundary : bool, optional
            Whether to strip the boundary after rolling, by default False. This
            will remove the boundary states along axes with non-periodic boundary 
            conditions.

        Returns
        -------
        np.ndarray
            The rolled wavefunction states with applied boundary conditions.

        Examples
        --------
        >>> rolled_wfa = wfa.roll_states_with_pbc([1, 0])
        >>> np.allclose(rolled_wfa[4, 3], rolled_wfa[3, 3])
        ... True
        """
        states = self.wfs 
        mesh = self.mesh

        if np.any(abs(np.array(shift_vec, dtype=int)) > 1):
            raise ValueError("Only unit shifts (+1, 0, -1) are supported in shift_vec.")

        if len(shift_vec) < mesh.num_k_axes:
            raise ValueError("shift_vec must have at least as many elements as k-axes in the mesh.")
        elif len(shift_vec) > mesh.num_axes:
            raise ValueError("shift_vec must have at most as many elements as total axes in the mesh.")

        rolled = states
        for ax, sh in enumerate(shift_vec):
            if not sh:
                continue
            wraps = mesh.is_axis_looped(ax) or mesh.is_axis_bz_winding(ax)
            closed = mesh.is_axis_closed(ax)
            if not closed and wraps:
                rolled = np.roll(rolled, shift=-int(sh), axis=ax)
            else:
                logger.info(f"Applying bounded shift {sh} to axis {ax} without wrapping.")
                rolled = self._bounded_shift(rolled, axis=ax, sh=-int(sh))

        if strip_boundary:
            sl = [slice(None)] * rolled.ndim
            for ax, sh in enumerate(shift_vec):
                loops = mesh.is_axis_looped(ax) or mesh.is_axis_bz_winding(ax)
                closed = mesh.is_axis_closed(ax)
                if sh and (closed or not loops):
                    # drop the last index in that direction
                    sl[ax] = slice(None, -1)
            rolled = rolled[tuple(sl)]

        phase = self._boundary_phase_for_shift(tuple(shift_vec))
        rolled = rolled * phase

        if flatten_spin_axis and self.nspin == 2:
            rolled = rolled.reshape(*rolled.shape[:-2], self.norb * self.nspin)
        return rolled 
    

    def overlap_matrix(self, use_k_metric: bool = False) -> np.ndarray:
        r"""Compute the overlap matrix of the cell periodic eigenstates on nearest neighbor k-shell.

        Overlap matrix is of the form

        .. math::

            M_{m,n}^{\mathbf{b}}(\mathbf{k}, \lambda) 
            = \langle u_{m, \mathbf{k}, \lambda} | u_{n, \mathbf{k+b}, \lambda} \rangle

        where :math:`\mathbf{b}` is a displacement vector connecting nearest neighbor k-points.

        When ``use_k_metric=True``, the function computes nearest neighbor k-points in the mesh considering the metric in
        Cartesian space. This means that :math:`\mathbf{b}` is not necessarily a unit vector in 
        reduced coordinates, but rather the vector that connects the closest k-points in Cartesian 
        space.

        When ``use_k_metric=False``, the function computes nearest neighbor k-points by shifting
        the k-points by one step along each k-axis in the mesh. This means that :math:`\mathbf{b}` 
        is a unit vector in reduced coordinates along each k-axis.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        use_k_metric : bool, optional
            Whether to use the k-metric for neighbor lookup. If True, the function computes nearest
            neighbor k-points in the mesh considering the metric in Cartesian space. If False, the
            function computes nearest neighbor k-points by shifting the k-points by one step along
            each k-axis in the mesh. Default is False.

        Returns
        -------
        M : np.ndarray
            Overlap matrix with shape ``(*shape_k, *shape_lambda, num_nnbrs, n_states, n_states)``

        Notes
        -----
        - :func:`overlap_matrix` delegates neighbor lookup to :meth:`roll_states_with_pbc`, so the
          behaviour at mesh boundaries depends entirely on how each sampling axis is
          labelled in the :class:`Mesh`:

            - **Periodic, no endpoints**

              Axes marked as looped/winding the BZ but without
              duplicated endpoints. We wrap with :func:`numpy.roll`, so the last k-point is 
              paired with the first and a Bloch phase is applied automatically. Every element 
              of the returned link tensor is meaningful.

            - **Periodic, endpoints included** 

              Axes that include the terminal point
              explicitly (``endpoint=True`` or user-provided meshes that repeat the BZ
              boundary). These are considered “closed” and we *do not* wrap. Instead the
              array is shifted without wraparound and the vacated slab is zero-filled. The
              final slice in the returned links therefore vanishes; drop it when forming
              Wilson loops or Berry phases. Calling the functions :meth:`wilson_loop` or
              :meth:`berry_phase` will automatically handle this for you.

            - **Non-periodic axes**

              Lambda axes or open directions. They take the same
              zero-filled code path as the previous case because no physical neighbour
              exists beyond the edge. Those terminal slices should likewise be ignored by
              downstream consumers.

        - In practice, after calling `overlap_matrix`, discard rows where the entries are ``np.nan``
          before accumulating Wilson loops or Berry phases.
        """

        if use_k_metric:
            logger.info("Computing overlap matrix using k-metric for neighbor lookup.")
            # Assumes only one shell for now
            _, idx_shell = self.get_k_shell(n_shell=1, report=False)
            idx_shell = idx_shell[0]

            # overlap matrix
            M = np.zeros(
                (*self.shape_mesh, len(idx_shell), self.nstates, self.nstates),
                dtype=complex,
            )

            u_nk = self.states(flatten_spin_axis=True)
            for idx, idx_vec in enumerate(idx_shell):  # nearest neighbors
                # introduce phases to states when k+b is across the BZ boundary
                states_pbc = self.roll_states_with_pbc(idx_vec, flatten_spin_axis=True)
                overlaps = np.einsum("...mj, ...nj -> ...mn", u_nk.conj(), states_pbc)
                overlaps = self._invalidate_boundary_links(overlaps, idx_vec)
                M[..., idx, :, :] = overlaps

        else:
            logger.info("Computing overlap matrix without k-metric for neighbor lookup.")
            # get number of k-axes
            n_k_axes = self.mesh.num_k_axes

            # overlap matrix
            M = np.zeros(
                (*self.shape_mesh, n_k_axes, self.nstates, self.nstates),
                dtype=complex,
            )

            u_nk = self.states(flatten_spin_axis=True)
            for axis in range(n_k_axes):  # nearest neighbors
                shift_vec = self._unit_shift(axis)
                # introduce phases to states when k+b is across the BZ boundary
                states_pbc = self.roll_states_with_pbc(shift_vec, flatten_spin_axis=True)
                overlaps = np.einsum("...mj, ...nj -> ...mn", u_nk.conj(), states_pbc)
                overlaps = self._invalidate_boundary_links(overlaps, shift_vec)
                M[..., axis, :, :] = overlaps

        return M
    

    def links(self, state_idx=None, axis_idxs=None):
        r"""Compute the overlap links (unitary matrices) for the wavefunctions.

        The overlap links along a given direction are defined as the unitary part of the overlap
        between the wavefunctions and their neighbors in the forward direction along each
        mesh directions. Specifically, the overlap matrices are computed as

        .. math::

            M_{nm}^{\mu}(\mathbf{k}) = \langle u_{nk} | u_{m, k + \delta k_{\mu}} \rangle

        where :math:`\mu` is the direction along which the link is computed, and
        :math:`\delta k_{\mu}` is the shift in the wavevector along that direction. The
        :math:`k` here could be a point in an arbitrary parameter mesh. The unitary link that
        is returned by the function is obtained through the singular value decomposition
        (SVD) of the overlap matrix :math:`M^{\mu}(\mathbf{k}) = V^{\mu} \Sigma^{\mu} (W^{\mu})^\dagger`
        as,

        .. math::

            U^{\mu}(\mathbf{k}) = V^{\mu} (W^{\mu})^\dagger

        .. versionadded:: 2.0.0

        .. warning:: 
            The neighbor at the boundary is defined with periodic boundary conditions by default.
            In most cases, this means that the last point in the mesh of :math:`U^{\mu}(\mathbf{k})`
            along each direction should be disregarded (see Notes for further details).

        Parameters
        ----------
        state_idx : int or list of int
            Index or indices of the states for which to compute the links.
            If an integer is provided, only that state will be considered.
            If a list is provided, links for all specified states will be computed.
        axis_idxs : list of int, optional
            List of `Mesh` axes along which to compute the links.
            If not provided, links will be computed for all directions in the mesh.

        Returns
        -------
        U_forward (np.ndarray):
            Array of shape ``(dim, *shape_k, *shape_l, n_states, n_states)``
            where 
            
            - ``dim`` is the number of dimensions of the mesh corresponding to :math:`\mu`
              in the equations above. If ``axis_idxs`` is provided, ``dim=len(axis_idxs)``; and
              the indexing of the first axis corresponds to the order of directions
              in ``axis_idxs``.
            - ``shape_k`` is the tuple of sizes of the mesh along each k-dimension, similarly
            - ``shape_l`` is the tuple of sizes of the mesh along each lambda-dimension,
            - The last two axes are the matrix elements of the unitary link matrices,
              where ``n_states`` is the number of states in the `WFArray` object.

        Notes
        -----
        - :meth:`links` delegates neighbor lookup to :meth:`roll_states_with_pbc`, so the
          behaviour at mesh boundaries depends entirely on how each sampling axis is
          labelled in the :class:`Mesh`:

            - **Periodic, no endpoints**

              Axes marked as looped/winding the BZ but without
              duplicated endpoints. We wrap with :meth:`numpy.roll`, so the last k-point is 
              paired with the first and a Bloch phase is applied automatically. Every element 
              of the returned link tensor is meaningful.

            - **Periodic, endpoints included** 

              Axes that include the terminal point
              explicitly (``endpoint=True`` or user-provided meshes that repeat the BZ
              boundary). These are considered “closed” and we *do not* wrap. Instead the
              array is shifted without wraparound and the vacated slab is zero-filled. The
              final slice in the returned links therefore vanishes; drop it when forming
              Wilson loops or Berry phases. Calling the functions :meth:`wilson_loop` or
              :meth:`berry_phase` will automatically handle this for you.

            - **Non-periodic axes**

              Lambda axes or open directions. They take the same
              zero-filled code path as the previous case because no physical neighbour
              exists beyond the edge. Those terminal slices should likewise be ignored by
              downstream consumers.

        - In practice, after calling :meth:`links`, discard rows where the entries
          are ``np.nan`` (typically the last index along any closed or nonperiodic axis).
        """
        wfs = self.states(flatten_spin_axis=True)

        # State selection
        if state_idx is not None:
            if isinstance(state_idx, (list, np.ndarray)):
                # If state_idx is a list or array, select those states
                state_idx = np.array(state_idx, dtype=int)
            elif isinstance(state_idx, int):
                # If state_idx is a single integer, convert to array
                state_idx = np.array([state_idx], dtype=int)
            else:
                raise TypeError("state_idx must be an integer, list, or numpy array.")

            wfs = np.take(wfs, state_idx, axis=-2)

        if axis_idxs is None:
            # If no specific directions are provided, compute links for all directions
            axis_idxs = list(range(self.naxes))

        U_forward = []
        for mu in axis_idxs:
            shift_vec = self._unit_shift(mu)
            wfs_shifted = self.roll_states_with_pbc(
                shift_vec=shift_vec, flatten_spin_axis=True)
            wfs_shifted = np.take(wfs_shifted, state_idx, axis=-2)

            # < u_{n, k} | u_{m, k + delta k_mu} >
            ovr_mu = wfs.conj() @ wfs_shifted.swapaxes(-2, -1)

            U_forward_mu = np.zeros_like(ovr_mu, dtype=complex)
            V, _, Wh = np.linalg.svd(ovr_mu, full_matrices=False)
            U_forward_mu = V @ Wh
            U_forward_mu = self._invalidate_boundary_links(U_forward_mu, shift_vec)
            U_forward.append(U_forward_mu)

        U_forward = np.array(U_forward)

        return U_forward

    @staticmethod
    def _wilson_loop(wfs_loop, wilson_evals: bool=False):
        r"""Wilson loop unitary matrix
        
        Computes the Wilson loop unitary matrix and its eigenvalues for multiband Berry phases.
        The Wilson loop is a geometric quantity that characterizes the topology of the
        band structure. It is defined as the product of the overlap matrices between
        neighboring wavefunctions in the loop. Specifically, it is given by

        .. math::

            U_{\text{Wilson}} = \prod_{n} U_{n}

        where :math:`U_{n}` is the unitary part of the overlap matrix between neighboring 
        wavefunctions in the loop, and the index :math:`n` labels the position in the loop 
        (see :meth:`links` for more details).

        When ``wilson_evals=True``, the function computes the eigenvalues of the Wilson loop
        unitary matrix. The eigenvalues are complex numbers of the form

        .. math::
            \lambda_n = e^{i \phi_n}
        
        where :math:`\phi_n` are the multiband Berry phases associated with each band.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        wfs_loop : np.ndarray
            Has format ``[loop_idx, band, orbital(, spin)]`` and loop has to be one dimensional.
            Assumes that first and last loop-point are the same. Therefore if
            there are n wavefunctions in total, will calculate phase along n-1
            links only!
        wilson_evals : bool, optional
            If True, then will compute eigenvalues of the Wilson loop unitary and
            return the negative phases.
            Otherwise just return the Wilson loop unitary matrix. Default is False.

        Returns
        -------
        U_wilson : np.ndarray
            Wilson loop unitary matrix of shape ``(band, band)``.
        eval_pha : np.ndarray, optional
            Multiband Berry phases associated with each band.
            Returned only if ``wilson_evals=True``, otherwise not returned.

        See Also
        --------
        :meth:`_berry_loop`
        :meth:`links`

        Notes
        ------
        ``wilson_evals`` are to be distinguished from multiband Berry phases, in :meth:`berry_phase`.
        The ``berry_evals`` are the phase arguments of ``wilson_evals`` and are always returned between 
        :math:`-\pi` and :math:`\pi`.
        """
        # check that wfs_loop has appropriate shape
        if wfs_loop.ndim < 3 or wfs_loop.ndim > 4:
            raise ValueError(
                "wfs_loop must be a 3D or 4D array with shape [loop_idx, band, orbital(, spin)]"
            )

        # check if there is a spin axis, then flatten
        is_spin = wfs_loop.ndim == 4 and wfs_loop.shape[-1] == 2
        if is_spin:
            # flatten spin axis
            wfs_loop = wfs_loop.reshape(wfs_loop.shape[0], wfs_loop.shape[1], -1, 2)

        ovr_mats = wfs_loop[:-1].conj() @ wfs_loop[1:].swapaxes(-2, -1)
        V, _, Wh = np.linalg.svd(ovr_mats, full_matrices=False)
        U_link = V @ Wh
        U_wilson = U_link[0]
        for i in range(1, len(U_link)):
            U_wilson = U_wilson @ U_link[i]

        # calculate phases of all eigenvalues
        if wilson_evals:
            eigvals = np.linalg.eigvals(U_wilson)  # Wilson loop eigenvalues
            return U_wilson, eigvals
        else:
            return U_wilson

    @staticmethod
    def _berry_loop(wfs_path, berry_evals: bool = False):
        r"""Berry phase along a one-dimensional path of wavefunctions.

        The Berry phase along a one-dimensional path of wavefunctions
        is computed using the Wilson loop unitary matrix.

        When ``berry_evals=False``, the Berry phase is computed as the logarithm
        of the determinant of the product of the overlap matrices between
        neighboring wavefunctions in the path. In otherwords, the Berry phase is
        given by the formula:

        .. math::

            \phi = -\text{Im} \ln \det U_{\rm Wilson}

        where :math:`U` is the Wilson loop unitary matrix obtained from
        :meth:`wilson_loop`. 

        When ``berry_evals=True``, the function returns an array of
        the individual phases (multiband Berry phases) for each band. 
        They are computed as

        .. math::

            \phi_n = -\text{Im} \ln \lambda_n

        where :math:`\lambda_n` are the eigenvalues of the Wilson loop 
        unitary matrix. These multiband Berry phases correspond to the
        "maximally localized Wannier centers" or "Wilson loop eigenvalues".

        .. versionadded:: 2.0.0

        Parameters
        ----------
        wfs_loop : np.ndarray
            Wavefunctions in the path, with shape ``(path_idx, band, orbital, spin)``. 
        berry_evals : bool, optional
            Default is `False`. If `True`, will return the argument of the eigenvalues
            of the Wilson loop unitary matrix instead of the total Berry phase.
            If False, will return the total Berry phase for the loop.

        Returns
        -------
        berry_phase : float
            The total Berry phase for the loop.
        berry_evals : np.ndarray, optional
            If berry_evals is True, returns an array of multiband Berry phases
            associated with each band.

        See Also
        --------
        :meth:`links`
        :meth:`berry_phase`
        :meth:`wilson_loop`
        :ref:`formalism` : Section 4.5

        Notes
        -----
        The loop is assumed to be one-dimensional. 
        The wavefunctions in the loop should be ordered such that the first point 
        corresponds to the first wavefunction,
        the second point to the second wavefunction, and so on, up to the last point,
        which corresponds to the last wavefunction.
         
        When the path of wavefunctions is closed, the Berry 
        phase corresponds to the geometric phase acquired by the wavefunctions
        as they are transported around the loop. If the path is not closed, the
        Berry phase will depend on the specific path taken.
        """
        if wfs_path.ndim < 3 or wfs_path.ndim > 4:
            raise ValueError(
                "wfs_path must be a 3D or 4D array with shape (path_idx, band, orbital(, spin))"
            )
        
        if berry_evals:
            U_wilson, eigvals = WFArray._wilson_loop(wfs_path, wilson_evals=berry_evals)
            eigvals_phase = -np.angle(eigvals)  # Multiband Berry phases
            # sort the eigenvalues
            eigvals_phase = np.sort(eigvals_phase)
            berry_phase = -np.angle(np.linalg.det(U_wilson))
            return berry_phase, eigvals_phase
        else:
            U_wilson = WFArray._wilson_loop(wfs_path, wilson_evals=berry_evals)
            berry_phase = -np.angle(np.linalg.det(U_wilson))
            return berry_phase

    def wilson_loop(
            self,
            axis_idx: int, 
            state_idx=None, 
            wilson_evals: bool = False
            ):
        r"""Wilson loop unitary matrix along a given mesh axis.
        
        Computes the Wilson loop unitary matrix and its eigenvalues for multiband Berry phases.
        The Wilson loop is a geometric quantity that characterizes the topology of the
        band structure. It is defined as the product of the overlap matrices between
        neighboring wavefunctions in the loop. Specifically, it is given by

        .. math::

            U_{\text{Wilson}} = \prod_{n} U_{n}

        where :math:`U_{n}` is the unitary part of the overlap matrix between neighboring 
        wavefunctions in the loop, and the index :math:`n` labels the position in the loop 
        (see :meth:`links` for more details).

        When ``wilson_evals=True``, the function computes and returns the eigenvalues of the 
        Wilson loop unitary matrix. The eigenvalues are complex numbers of the form

        .. math::
            \lambda_n = e^{i \phi_n}
        
        where :math:`\phi_n` are the multiband Berry phases associated with each band.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        axis_idx : int
            Index of ``Mesh`` axis along which Wilson loop is computed. 
        state_idx : int, array-like, optional
            Optional band index or array of band indices to be included
            in the subsequent calculations. If unspecified, all bands are 
            included.
        wilson_evals : bool, optional
            If True, then will compute eigenvalues of the Wilson loop unitary and
            return the negative phases.
            Otherwise just return the Wilson loop unitary matrix. Default is False.

        Returns
        -------
        U_wilson : np.ndarray
            Wilson loop unitary matrix of shape ``(band, band)``.
        eval_pha : np.ndarray, optional
            Multiband Berry phases associated with each band.
            Returned only if ``wilson_evals=True``, otherwise not returned.

        See Also
        --------
        :meth:`berry_phase`
        :meth:`links`

        Notes
        ------
        - ``wilson_evals`` are to be distinguished from multiband Berry phases, in :meth:`berry_phase`.
          The ``berry_evals`` are the phase arguments of ``wilson_evals`` and are always returned between 
          :math:`-\pi` and :math:`\pi`.
        """
        # print(axis_idx)
        
        if not isinstance(axis_idx, int) or axis_idx < 0 or axis_idx >= self.naxes:
            raise ValueError(f"axis_idx must be an integer in [0, {self.naxes-1}]")

        # States (optionally restricted to a subspace)
        u = self.states(state_idx=state_idx, flatten_spin_axis=True)
        u_expanded = self.states(state_idx=state_idx, flatten_spin_axis=False)

        u_loop = u  # init wf loop

        for ax, comp in self.mesh._get_loop_ax_comp():
            # If axis is periodic and open, we need to append
            # the first state to the end
            if ax == axis_idx and not self.mesh.is_axis_closed(ax, comp):
                # If component is along k and wraps bz, apply phase
                if self.mesh.is_axis_bz_winding(ax, comp):
                    logger.debug("Applying phase to state at beginning to end of open periodic axis.")
                    phase, _, _ = self._get_pbc_phases(ax, comp)
                    u_first = np.take(u_expanded, 0, axis=axis_idx)
                    u_last = u_first * phase
                # No phase is applied
                else:
                    u_last = np.take(u_expanded, 0, axis=axis_idx)

                # flatten spin
                if self.nspin == 2:
                    u_last = u_last.reshape(*u_last.shape[:-2], -1)

                logger.debug("Appending state at beginning to end of open periodic axis.")
                u_loop = np.concatenate([u_loop, np.expand_dims(u_last, axis=axis_idx)], axis=axis_idx)

        # Bring loop axis first for easy slicing over transverse axes
        u_loop = np.moveaxis(u_loop, axis_idx, 0)
        tail_shape = u_loop.shape[1:-2] # Shape of the tail (transverse) axes
        n_sub = u_loop.shape[-2] # Number of subbands

        if wilson_evals:
            evals = np.empty((*tail_shape, n_sub), dtype=float)

        unitar = np.empty((*tail_shape, n_sub, n_sub), dtype=complex)

        # Iterate over transverse indices without flattening
        it = np.ndindex(*tail_shape) if tail_shape else [()]
        for idx in it:
            # Take all points along loop axis, and the given transverse indices
            # plus all states and orbitals (and spin)
            slicer = (slice(None),) + idx + (slice(None), slice(None))
            wf_line = u_loop[slicer]  # shape: (n_mu or n_mu+1, n_sub, norb*spin)

            if wilson_evals:
                # val are the individual phases of Wilson loop eigenvalues
                U, eval = self._wilson_loop(wf_line, wilson_evals=wilson_evals)
                evals[idx] = eval
            else:
                U = self._wilson_loop(wf_line, wilson_evals=wilson_evals)

            # val is the total Berry phase for the loop
            unitar[idx] = U

        unitar = np.array(unitar)

        if wilson_evals:
            evals = np.array(evals)
            return unitar, evals

        return unitar

    def berry_phase(
            self,
            axis_idx: int, 
            state_idx=None, 
            berry_evals: bool = False,
            contin: bool = True
            ):
        r"""Berry phase along a given mesh axis.

        By default, the function returns the Berry phase traced
        over the specified set of bands. Optionally, with ``berry_evals=True``,
        the function will also return the multiband berry phases associated with
        the phase of the eigenvalues of the :meth:`wilson_loop` unitary 
        matrix (corresponding to "hybrid Wannier centers" or "Wilson loop eigenvalues").
        Explicitly, these take the form

        .. math::
            \phi_n = -\text{Im} \ln \lambda_n

        where :math:`\lambda_n` are the eigenvalues of the Wilson loop unitary matrix
        from :meth:`wilson_loop`.

        Parameters
        ----------
        axis_idx : int
            Index of ``Mesh`` axis along which Berry phase is
            computed. This parameters needs not be specified for
            a one-dimensional ``WFArray``.

            .. versionchanged:: 2.0.0
                Changed parameter name from `dir` to `axis_idx` to avoid conflict
                with built-in Python function `dir()`.

        state_idx : int, array-like, optional
            Optional band index or array of band indices to be included
            in the subsequent calculations. If unspecified, all bands are 
            included.

            .. versionchanged:: 2.0.0
                Renamed from ``occ``. The band indices are not required to be
                occupied bands only. The default behavior is to include all bands,
                and the ``"all"`` option has been removed.

        contin : bool, optional
            If True (default) then the branch choice of the Berry phase (which is 
            indeterminate modulo :math:`2\pi`) is made so that neighboring strings 
            (in the direction of increasing index value) have as close as
            possible phases. The phase of the first string (with lowest
            index) is always constrained to be between :math:`-\pi` and :math:`\pi`.
            If False, the Berry phase for every string is constrained to be
            between :math:`-\pi` and :math:`\pi`. 

        berry_evals : bool, optional
            If True then will compute and return the phases of the eigenvalues of the
            product of overlap matrices. (These numbers correspond also
            to hybrid Wannier function centers.) These phases are either
            forced to be between :math:`-\pi` and :math:`\pi` (if ``contin=False``) or
            they are made to be continuous (if ``contin=True``).

        Returns
        -------
        phase : np.ndarray
            Total accumulated Berry phase along the specified axis. 
            When only a single axis is present in the ``Mesh``, this is
            a scalar. When multiple axes are present, this is an array
            with one less dimension than the original mesh, corresponding 
            to the total Berry phase for the remaining ``Mesh`` points.
            For example, if ``Mesh`` has three axes, indexed by ``[i,j,k]``,
            and we specify ``axis_idx=1``, then ``phase`` will be two 
            dimensional array with indices ``[i,k]``.

        evals : np.ndarray, optional
            Phases of each eigenvalue of the Wilson loop unitary (product of
            unitary part of overlap matrices along the specified axis).
            In the convention used for the previous example, 
            ``evals`` in this case would have indices ``[i,k,n]``,
            where ``n`` refers to the index of the individual phase of
            the product matrix eigenvalue.

        See Also
        ---------
        :ref:`haldane-bp-nb` : For an example
        :ref:`cone-nb` : For an example
        :ref:`three-site-thouless-nb` : For an example
        :meth:`wilson_loop` : For a function that computes Wilson loops.
        :ref:`formalism` : Sec. 4.5 for the discretized formula used to compute Berry phase.

        Notes
        -----
        - For a single ``Mesh`` axis in ``WFArray`` (i.e., a single string), the
          computed Berry phases are always chosen to be between :math:`-\pi` 
          and :math:`\pi`. For a higher dimensional ``WFArray``, the Berry phase 
          is computed for each one-dimensional string of points, and an array of
          Berry phases is returned. The Berry phase for the first string
          (with lowest index) is always constrained to be between :math:`-\pi` and
          :math:`\pi`. The range of the remaining phases depends on the value of
          the input parameter ``contin``.

        - For an array of size ``N`` in direction ``axis_idx``, the Berry phase
          is computed from the ``N-1`` inner products of neighboring
          eigenfunctions. This corresponds to an "open-path Berry
          phase" if the first and last points have no special
          relation. If they correspond to the same physical
          Hamiltonian, then a closed-path Berry phase will be computed.

        - The bands in ``state_idx`` should be non-degenerate with states
          outside the manifold. This means they should be well separated in energy.
          It is the responsibility of the user to check that this is satisfied.

        Examples
        ---------
        Computes Berry phases along second direction for three lowest
        occupied states. For example, if wf is threedimensional, then
        ``pha[2, 3]`` would correspond to Berry phase of string of states
        along ``wf[2, :, 3]``

        >>> pha = wf.berry_phase([0, 1, 2], 1)
        """
        if not isinstance(axis_idx, int) or axis_idx < 0 or axis_idx >= self.naxes:
            raise ValueError(f"axis_idx must be an integer in [0, {self.naxes-1}]")

        # States (optionally restricted to a subspace)
        u = self.states(state_idx=state_idx, flatten_spin_axis=True)
        u_expanded = self.states(state_idx=state_idx, flatten_spin_axis=False)

        u_loop = u  # init wf loop

        for ax, comp in self.mesh._get_loop_ax_comp():
            # If axis is periodic and open, we need to append
            # the first state to the end
            if ax == axis_idx and not self.mesh.is_axis_closed(ax, comp):
                # If component is along k and wraps bz, apply phase
                if self.mesh.is_axis_bz_winding(ax, comp):
                    logger.debug("Applying phase to state at beginning to end of open periodic axis.")
                    phase, _, _ = self._get_pbc_phases(ax, comp)
                    u_first = np.take(u_expanded, 0, axis=axis_idx)
                    u_last = u_first * phase
                # No phase is applied
                else:
                    u_last = np.take(u_expanded, 0, axis=axis_idx)

                # flatten spin
                if self.nspin == 2:
                    u_last = u_last.reshape(*u_last.shape[:-2], -1)

                logger.debug("Appending state at beginning to end of open periodic axis.")
                u_loop = np.concatenate([u_loop, np.expand_dims(u_last, axis=axis_idx)], axis=axis_idx)

        # Bring loop axis first for easy slicing over transverse axes
        u_loop = np.moveaxis(u_loop, axis_idx, 0)
        tail_shape = u_loop.shape[1:-2] # Shape of the tail (transverse) axes
        n_sub = u_loop.shape[-2] # Number of subbands

        if berry_evals:
            out = np.empty((*tail_shape, n_sub), dtype=float)
        else:
            out = np.empty(tail_shape, dtype=float)

        # Iterate over transverse indices without flattening
        it = np.ndindex(*tail_shape) if tail_shape else [()]
        for idx in it:
            # Take all points along loop axis, and the given transverse indices
            # plus all states and orbitals (and spin)
            slicer = (slice(None),) + idx + (slice(None), slice(None))
            wf_line = u_loop[slicer]  # shape: (n_mu or n_mu+1, n_sub, norb*spin)

            if berry_evals:
                # val are the individual phases of Berry loop eigenvalues
                _, val = self._berry_loop(wf_line, berry_evals=berry_evals)
            else:
                # val is the total Berry phase for the loop
                val = self._berry_loop(wf_line, berry_evals=berry_evals)

            out[idx] = val

        out = np.array(out)

        # Make continuous
        if contin:
            if len(tail_shape) == 0:
                # Make phases continuous for each band
                # ret = np.unwrap(ret, axis=0)
                pass

            elif berry_evals:
                # 2D case
                if out.ndim == 2:
                    out = _array_phases_cont(out, out[0])
                # 3D case
                elif out.ndim == 3:
                    for i in range(out.shape[1]):
                        if i == 0:
                            clos = out[0, 0]
                        else:
                            clos = out[0, i-1]
                        out[:, i] = _array_phases_cont(out[:, i], clos)
                elif self._dim_arr != 1:
                    raise ValueError("Wrong dimensionality!")

            else:
                # 2D case
                if out.ndim == 1:
                    out = _one_phase_cont(out, out[0])
                # 3D case
                elif out.ndim == 2:
                    for i in range(out.shape[1]):
                        if i == 0:
                            clos = out[0, 0]
                        else:
                            clos = out[0, i-1]
                        out[:, i] = _one_phase_cont(out[:, i], clos)
                elif self._dim_arr != 1:
                    raise ValueError("Wrong dimensionality!")

        return out
        

    def berry_flux(
            self, 
            plane = None, 
            state_idx = None, 
            non_abelian: bool = False
            ):
        r"""Berry flux tensor.

        The Berry flux tensor quantifies the geometric phase acquired by
        Bloch states as they are adiabatically transported around a closed
        loop in parameter space (e.g., in momentum space or along adiabatic
        dimensions). In the discretized Fukui–Hatsugai–Suzuki (FHS) formalism, 
        the closed loop is taken around each **4-point plaquette** of the 
        parameter mesh.
        
        
        The Abelian Berry flux is defined as the trace over the band indices of the non-Abelian
        Berry flux tensor.

        .. math::

            \mathcal{F}_{\mu\nu}(\mathbf{k}) = \sum_{n} (\mathcal{F}_{\mu\nu}(\mathbf{k}))_{n, n}.

        In the case of a 2-dimensional *WFArray* array calculates the
        Berry curvature over the entire plane.  In higher dimensional case
        it will compute flux over all 2-dimensional slices of a 
        higher-dimensional *WFArray*.

        .. versionremoved:: 2.0.0
            The `individual_phases` parameter has been removed.

        Parameters
        ----------
        plane : array_like of shape (2,), optional
            Array or tuple of two indices defining the axes in the
            WFArray mesh which the Berry flux is computed over. By default,
            all directions are considered, and the full Berry flux tensor is
            returned.

            .. versionchanged:: 2.0.0
                Renamed from ``dirs``.

        state_idx : array_like, optional
            Optional array of indices of states to be included
            in the subsequent calculations, typically the indices of
            bands considered occupied. If not specified, or None, all bands are
            included.

            .. versionchanged:: 2.0.0
                Renamed from ``occ``. The band indices are not required to be
                occupied bands only. The default behavior is to include all bands,
                and the ``"all"`` option has been removed.

        non_abelian : bool, optional
            If *True* then the non-Abelian Berry flux tensor is computed.
            If *False* then the Berry flux is computed using the abelian formula,
            which corresponds to the band-traced non-Abelian Berry curvature.
            Default value is *False*.

            .. versionadded:: 2.0.0

        Returns
        -------
        flux : ndarray
            The Berry flux tensor, which is an array of general shape
            `[ndims, ndims, *flux_shape, n_states, n_states]`. The
            shape will depend on the parameters passed to the function.

            If plane is `None` (default), then the first two axes
            `(ndims, ndims)` correspond to the plane directions, otherwise,
            these axes are absent.

            If `abelian` is `False` then the last two axes are the band indices
            running over the selected `state_idx` indices.
            If `abelian` is `True` (default) then the last two axes are absent, and
            the returned flux is a scalar value, not a matrix.

        Notes
        ------
        For a given pair of mesh directions :math:`(\mu, \nu)`, the plaquette
        is formed by the points:

        .. math::

            \begin{pmatrix}
            \mathbf{k} + \hat{\mu} + \hat{\nu} \\
            \mathbf{k} + \hat{\mu} - \hat{\nu} \\
            \mathbf{k} - \hat{\mu} - \hat{\nu} \\
            \mathbf{k} - \hat{\mu} + \hat{\nu}
            \end{pmatrix}

        Let :math:`U_{\mu}(\mathbf{k})` denote the unitary **link matrix**
        (unitary part of overlap matrix between states) from
        :math:`\mathbf{k}` to :math:`\mathbf{k} + \hat{\mu}`:

        .. math::

            \big[ U_{\mu}(\mathbf{k}) \big]_{mn} =
                \langle u_{m}(\mathbf{k}) \,|\, u_{n}(\mathbf{k} + \hat{\mu}) \rangle

        where :math:`m,n` run over specified band indices.

        The (Abelian) Berry flux tensor is computed by taking the imaginary part of the logarithm 
        of the determinant of the product of the link matrices around the plaquettes.
        It is defined as,

        .. math::

            \mathcal{F}_{\mu\nu}(\mathbf{k}) = 
            -\mathrm{Im}\ln\det[U_{\mu}(\mathbf{k}) U_{\nu}(\mathbf{k} + \hat{\mu}) 
            U_{\mu}^{-1}(\mathbf{k} + \hat{\nu}) U_{\nu}^{-1}(\mathbf{k})].

        The (non-Abelian) Berry flux tensor is computed by taking the 
        imaginary part of the matrix logarithm of the product of the link matrices
        around the plaquettes. It is defined as

        .. math::

            \mathcal{F}_{\mu\nu}(\mathbf{k}) =
            -\mathrm{Im} \,\ln \Big[
                U_{\mu}(\mathbf{k}) \;
                U_{\nu}(\mathbf{k} + \hat{\mu}) \;
                U_{\mu}^\dagger(\mathbf{k} + \hat{\nu}) \;
                U_{\nu}^\dagger(\mathbf{k})
            \Big]

        This definition holds for multi-band subspaces, where the link
        matrices are square and unitary in the occupied-band space.

        Examples
        --------
        Computes Berry curvature of first three bands in 2D model

        >>> flux = wf.berry_flux([0, 1, 2]) # shape: (dim1, dim2, nk1, nk2)
        >>> flux = wf.berry_flux([0, 1, 2], plane=(0, 1)) # shape: (nk1, nk2)
        >>> flux = wf.berry_flux([0, 1, 2], plane=(0, 1), abelian=False) # shape: (nk1, nk2, n_states, n_states)

        3D model example

        >>> flux = wf.berry_flux([0, 1, 2], plane=(0, 1)) # shape: (nk1, nk2, nk3)
        """
        if (self.naxes) < 2:
            raise ValueError("Berry curvature only defined if number of mesh axes >= 2.")
        
        # Validate state_idx
        if state_idx is None:
            state_idx = np.arange(self.nstates)
        elif isinstance(state_idx, (list, np.ndarray, tuple)):
            state_idx = np.array(state_idx, dtype=int)
            if state_idx.ndim != 1:
                raise ValueError("state_idx must be a one-dimensional array.")
            if np.any(state_idx < 0) or np.any(state_idx >= self.nstates):
                raise ValueError(f"state_idx must be between 0 and {self.nstates-1}.")
        else:
            raise TypeError("state_idx must be None, a list, tuple, or numpy array.")
        if len(state_idx) == 0:
            raise ValueError("state_idx cannot be empty.")
        if np.any(np.diff(state_idx) < 0):
            raise ValueError("state_idx must be sorted in ascending order.")

        ndims = self.naxes  # Total dimensionality of adiabatic space: d
        # Validate plane
        if plane is not None:
            if not isinstance(plane, (list, tuple, np.ndarray)):
                raise TypeError("plane must be None, a list, tuple, or numpy array.")
            if len(plane) != 2:
                raise ValueError("plane must contain exactly two directions.")
            if any(p < 0 or p >= ndims for p in plane):
                raise ValueError(f"Plane indices must be between 0 and {ndims-1}.")
            if plane[0] == plane[1]:
                raise ValueError("Plane indices must be different.")
            
        n_states = len(state_idx)  # Number of states considered
        flux_shape = list(
            self.shape_mesh
        )  # Number of points in adiabatic mesh: (nk1, nk2, ..., nkd)

        # Initialize the Berry flux array
        if plane is None:
            shape = (
                (ndims, ndims, *flux_shape, n_states, n_states)
                if non_abelian
                else (ndims, ndims, *flux_shape)
            )
            berry_flux = np.zeros(shape, dtype=complex)
            dirs = list(range(ndims))
            plane_idxs = ndims
        else:
            p, q = plane  # Unpack plane directions
            dirs = [p, q]
            plane_idxs = 2

            shape = (*flux_shape, n_states, n_states) if non_abelian else (*flux_shape,)
            berry_flux = np.zeros(shape, dtype=float)

        # Trim the last point along closed/non-periodic axes to avoid overcounting
        for ax in dirs:
            if (self.mesh.is_axis_closed(ax)
                or (
                    not self.mesh.is_axis_looped(ax) 
                    and not self.mesh.is_axis_bz_winding(ax)
                    )
                ):
                logger.debug(
                    f"Axis {ax} is closed or non-periodic. "
                    "Trimming the last point in the flux array to avoid overcounting."
                    )
                if plane is None:
                    berry_flux = np.delete(berry_flux, -1, axis=ax+2)
                else:
                    berry_flux = np.delete(berry_flux, -1, axis=ax)

        # U_forward: Unitary part of overlaps <u_{nk} | u_{n, k+delta k_mu}>
        U_forward = self.links(state_idx=state_idx, axis_idxs=dirs)

        # Compute Berry flux for each pair of states
        for mu in range(plane_idxs):
            for nu in range(mu + 1, plane_idxs):
                # NOTE: The order of U_forward follows the order in dirs, so we index accordingly
                # e.g., if dirs = [p, q], then mu=0 -> p, mu=1 -> q
                U_mu = U_forward[mu]
                U_nu = U_forward[nu]

                # Shift the links along the mu and nu directions
                # NOTE: We index dirs to get the correct ordering
                axis_mu = dirs[mu]
                axis_nu = dirs[nu]
                U_nu_shift_mu = np.roll(U_nu, -1, axis=axis_mu)
                U_mu_shift_nu = np.roll(U_mu, -1, axis=axis_nu)

                # Wilson loops: W = U_{mu}(k_0) U_{nu}(k_0+delta_mu) U^{-1}_{mu}(k_0+delta_mu+delta_nu) U^{-1}_{nu}(k_0)
                U_wilson = (
                    U_mu
                    @ U_nu_shift_mu
                    @ U_mu_shift_nu.conj().swapaxes(-1, -2)
                    @ U_nu.conj().swapaxes(-1, -2)
                )

                # Trim the last point along closed/non-periodic axes to avoid overcounting
                for ax in dirs:
                    if (self.mesh.is_axis_closed(ax)
                        or (
                            not self.mesh.is_axis_looped(ax) 
                            and not self.mesh.is_axis_bz_winding(ax)
                            )
                        ):
                        logger.debug(
                            f"Axis {ax} is closed or non-periodic. "
                            "Trimming the last point in the Wilson loop to avoid overcounting."
                            )
                        # slice_per_axis[ax] = slice(None, -1)
                        U_wilson = np.delete(U_wilson, -1, axis=ax)

                if non_abelian:
                    # Non-Abelian lattice field strength: F = -i Log(U_wilson)
                    # Matrix log using eigen-decompositon
                    # Eigen-decompose U_wilson = V diag(-phi_j) V^{-1}, phi_j in (-pi, pi]
                    eigvals, eigvecs = np.linalg.eig(U_wilson)
                    phi = -np.angle(eigvals)
                    F_diag = np.einsum(
                        "...i, ij -> ...ij", phi, np.eye(phi.shape[-1])
                    )
                    eigvecs_inv = np.linalg.inv(eigvecs)
                    phases_plane = eigvecs @ F_diag @ eigvecs_inv
                else:
                    det_U = np.linalg.det(U_wilson)
                    phases_plane = -np.angle(det_U)

                if plane is None:
                    # Store the Berry flux in a 2D array for each pair of directions
                    berry_flux[mu, nu] = phases_plane
                    berry_flux[nu, mu] = -phases_plane
                else:
                    berry_flux = phases_plane.real

        return berry_flux
    
    def berry_curvature(
        self,
        plane = None,
        state_idx = None,
        non_abelian: bool = False,
        return_flux: bool = False,
        ):
        r"""Berry curvature tensor.

        The difference between this function and :meth:`berry_flux` is that this function computes a dimensionful
        Berry curvature tensor, while :meth:`berry_flux` is dimensionless. Effectively, this function divides by
        the area of the plaquette. The area is set by the mesh spacing along each direction.

        The Berry curvature can be approximated by the flux by simply dividing by the
        area of the plaquette, approximating the flux as a constant over the small loop.

        .. math::

            \Omega_{\mu\nu}(\mathbf{k}) \approx \frac{\mathcal{F}_{\mu\nu}(\mathbf{k})}{A_{\mu\nu}},

        where :math:`A_{\mu\nu}` is the area (in Cartesian units) of the plaquette in parameter space. 

        .. versionadded:: 2.0.0

        Parameters
        ----------
        state_idx : int or list of int, optional
            Index or indices of the states to compute the Berry curvature for. 
            By default None, which computes for all states.
        plane : array_like, optional
            Array or tuple of two indices defining the axes in the
            WFArray mesh which the Berry flux is computed over. By default,
            all directions are considered, and the full Berry flux tensor is
            returned.
        non_abelian : bool, optional
            Whether to compute the non-Abelian Berry curvature. By default False.
        return_flux : bool, optional
            Whether to return the Berry flux alongside the curvature. By default False.

        Returns
        -------
        berry_curv : np.ndarray
            Berry curvature tensor with shape depending on input parameters.
        berry_flux : np.ndarray, optional
            Berry flux tensor with shape depending on input parameters.
        """
        n_lambda = list(self.nlams)  # Number of adiabatic parameters
        nks = list(self.nks)
        dim_k = self.dim_k      # Number of k-space dimensions
        dim_lam = self.dim_lambda   # Number of adiabatic dimensions
        dim_total = dim_k + dim_lam  # Total number of dimensions

        ndims = self.naxes  # Total dimensionality of adiabatic space: d
        if plane is None:
            dirs = list(range(ndims))
        else:
            p, q = plane  # Unpack plane directions
            dirs = [p, q]

        for ax in dirs:
            if (self.mesh.is_axis_closed(ax)
                or (
                    not self.mesh.is_axis_looped(ax) 
                    and not self.mesh.is_axis_bz_winding(ax)
                    )
                ):
                if ax < len(nks):
                    nks[ax] -= 1
                else:
                    n_lambda[ax - len(nks)] -= 1
                   

        Berry_flux = self.berry_flux(state_idx=state_idx, non_abelian=non_abelian)
        Berry_curv = np.zeros_like(Berry_flux, dtype=complex)

        # Get delta vectors for each dimension in parameter space
        recip_lat_vecs = self.lattice.recip_lat_vecs  # Expressed in inverse cartesian (x,y,z) coordinates
        dks = np.zeros((dim_total, dim_total))
        dks[:dim_k, :dim_k] = recip_lat_vecs / np.array([nk for nk in self.nks])[:, None]

        # set delta lambda to be the difference between the first and last points along
        # each adiabatic axis (param_points has shape (*nl, dim_total))
        if dim_lam != 0:
            for i, param_ax in enumerate(self.mesh.lambda_axis_indices):
                component = self.mesh.lambda_component_indices[i]
                param_points = self.mesh.get_axis_range(param_ax, component_index=component)
                diff = param_points[-1] - param_points[0]
                dlam = diff / n_lambda[i]
                dks[dim_k + i, dim_k + i] = dlam

        # Divide by area elements for the (mu, nu)-plane
        for mu in range(len(dirs)):
            for nu in range(mu+1, len(dirs)):
                A = np.vstack([dks[dirs[mu]], dks[dirs[nu]]])
                area_element = np.sqrt(np.linalg.det(A @ A.T))

                # Divide flux by the area element to get approx curvature
                Berry_curv[mu, nu] = Berry_flux[mu, nu] / area_element
                Berry_curv[nu, mu] = Berry_flux[nu, mu] / area_element

        if plane is not None:
            Berry_curv, Berry_flux = Berry_curv[plane], Berry_flux[plane]
        if return_flux:
            return Berry_curv, Berry_flux
        else:
            return Berry_curv
        

    def chern_number(self, plane=(0, 1), state_idx=None):
        r"""Computes the Chern number in the specified plane.

        The Chern number is computed as the integral of the Berry flux
        over the specified plane, divided by :math:`2 \pi`.

        .. math::
            C = \frac{1}{2\pi} \sum_{\mathbf{k}_{\mu}, \mathbf{k}_{\nu}} F_{\mu\nu}(\mathbf{k}).

        The plane :math:`(\mu, \nu)` is specified by `plane`, a tuple of two indices.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        plane : tuple
            A tuple of two indices specifying the plane in which the Chern number is computed.
            The indices should be between 0 and the number of mesh dimensions minus 1. 
            If None, the Chern number is computed for the first two dimensions of the mesh.

        state_idx : array-like, optional array
            Indices of states to be included in the Chern number calculation.
            If None, all states are included. None by default.

        Returns
        -------
        chern : np.ndarray, float
            In the two-dimensional case, the result
            will be a floating point approximation of the integer Chern number
            for that plane. In a higher-dimensional space, the Chern number
            is computed for each 2D slice of the higher-dimensional space.
            E.g., the shape of the returned array is `(nk3, ..., nkd)` if the plane is 
            `(0, 1)`, where `(nk3, ..., nkd)` are the sizes of the mesh in the remaining
            dimensions.

        Examples
        --------
        Suppose we have a `WFArray` mesh in three-dimensional space
        of shape `(nk1, nk2, nk3)`. We can compute the Chern number for the
        `(0, 1)` plane as follows:

        >>> wfs = WFArray(model, [10, 11, 12])
        >>> wfs.solve_on_grid()
        >>> chern = wfs.chern_number(plane=(0, 1), state_idx=np.arange(n_occ))
        >>> print(chern.shape)
        (12,)  # shape of the Chern number array
        """
        if state_idx is None:
            state_idx = np.arange(self.nstates)  # assume half-filled occupied

        # shape of the Berry flux array: (nk1, nk2, ..., nkd)
        berry_flux = self.berry_flux(state_idx=state_idx, plane=plane, non_abelian=False)
        # shape of chern (if plane is (0,1)): (nk3, ..., nkd)
        chern = np.sum(berry_flux, axis=plane) / (2 * np.pi)

        return chern

    def position_matrix(
            self, 
            pos_dir: int,
            mesh_idx, 
            state_idx=None
            ):
        r"""Position matrix for a given k-point and set of states.

        Position operator is defined in reduced coordinates.
        The returned object :math:`X` is

        .. math::

          X_{m n {\bf k}}^{\alpha} = \langle u_{m {\bf k}} \vert
          r^{\alpha} \vert u_{n {\bf k}} \rangle

        Here :math:`r^{\alpha}` is the position operator along direction
        :math:`\alpha` that is selected by `pos_dir`.

        This routine can be used to compute the position matrix for a
        given k-point and set of states (which can be all states, or
        a specific subset).

        Parameters
        ----------
        pos_dir: int
            Direction of the position operator. ``0`` corresponds to the first
            non-periodic direction, ``1`` to the second, and so on.

            .. versionchanged:: 2.0.0
                Renamed from ``dir`` to ``pos_dir`` to avoid conflict with built-in Python function `dir()`.

        mesh_idx: array-like of int 
            Set of integers specifying the :math:`(k, \lambda)`-point of interest in the mesh.
        state_idx: array-like, optional
            List of states to be included. If not specified, all states are included.

            .. versionchanged:: 2.0.0
                Renamed from ``occ``. The band indices are not required to be
                occupied bands only. The default behavior is to include all bands,
                and the ``"all"`` option has been removed.

        
        Returns
        -------
        pos_mat : np.ndarray
            Position operator matrix :math:`X_{m n}` as defined above. 
            This is a square matrix with size determined by number of bands
            given in `evec` input array.  First index of `pos_mat` corresponds to
            bra vector (:math:`m`) and second index to ket (:math:`n`).

        
        See Also
        --------
        :func:`pythtb.TBModel.position_matrix`
        
        Notes
        -----
        The only difference in :func:`pythtb.TBModel.position_matrix` is that, 
        in addition to specifying ``pos_dir``, one also has to specify ``mesh_idx`` 
        (mesh-point of interest) and ``state_idx`` 
        (list of states to be included, which can optionally be 'all').
        """

        # Check for special case of parameter state_idx
        if state_idx is None:
            state_idx = np.arange(self.nstates, dtype=int)
        elif isinstance(state_idx, (list, np.ndarray, tuple, range)):
            state_idx = list(state_idx)
            state_idx = np.array(state_idx, dtype=int)
        else:
            raise TypeError(
                "state_idx must be a list, numpy array, tuple, or None (all states) defining "
                "band indices of interest."
            )

        if state_idx.ndim != 1:
            raise ValueError(
                """Parameter state_idx must be a one-dimensional array or None (all states)."""
            )
        
        if isinstance(mesh_idx, (list, np.ndarray, tuple)):
            mesh_idx = tuple(mesh_idx)
        else:
            raise TypeError(
                "mesh_idx must be a list, numpy array, or tuple defining "
                "k-point indices of interest."
            )

        # # check if model came from w90
        # if not self._assume_position_operator_diagonal:
        #     _offdiag_approximation_warning_and_stop()
        evec = self.wfs[tuple(mesh_idx)][state_idx]

        # make sure specified direction is not periodic!
        if pos_dir in self.lattice.periodic_dirs:
            raise Exception(
                "Can not compute position matrix elements along periodic direction!"
            )
        # make sure direction is not out of range
        if pos_dir < 0 or pos_dir >= self.lattice.dim_r:
            raise Exception("Direction out of range!")

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
        pos_tmp = self.lattice.orb_vecs[:, pos_dir]
        # reshape arrays in the case of spinfull calculation
        if self.nspin == 2:
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

    def position_expectation(
            self, 
            pos_dir: int, 
            mesh_idx=None, 
            state_idx=None
            ):
        r"""Position expectation value for a given k-point and set of states.

        These elements :math:`X_{n n}` can be interpreted as an
        average position of n-th Bloch state ``evec[n]`` along
        direction ``pos_dir``. 

        This routine can be used to compute the position expectation value for a
        given k-point and set of states (which can be all states, or
        a specific subset). 

        Parameters
        ----------
        pos_dir: int
            Direction of the position operator. ``0`` corresponds to the first
            non-periodic direction, ``1`` to the second, and so on.

            .. versionchanged:: 2.0.0
                Renamed from ``dir`` to ``pos_dir`` to avoid conflict with built-in Python function `dir()`.

        mesh_idx: array-like of int, optional
            Set of integers specifying the :math:`(k, \lambda)`-point of interest in the mesh.
            If not specified, position expectation values are computed for all mesh points.
        state_idx: array-like, optional
            List of states to be included. If not specified, all states are included.

            .. versionchanged:: 2.0.0
                Renamed from ``occ``. The band indices are not required to be
                occupied bands only. The default behavior is to include all bands,
                and the ``"all"`` option has been removed.

        Returns
        -------
        pos_exp : np.ndarray
            Diagonal elements of the position operator matrix :math:`X`.
            Length of this vector is determined by number of bands given in *evec* input
            array. 

        See Also
        --------
        :func:`pythtb.TBModel.position_expectation`
        :ref:`haldane-hwf-nb` : For an example.
        position_matrix : For definition of matrix :math:`X`.

        Notes
        -----
        The only difference in :func:`pythtb.TBModel.position_expectation` is that,
        in addition to specifying ``pos_dir``, one also has to specify ``mesh_idx``
        (mesh-point of interest) and ``state_idx`` (list of states to be included).

        Generally speaking these centers are _not_ hybrid Wannier function 
        centers (which are instead returned by :func:`position_hwf`).
        """
        
        if mesh_idx is None:
            pos_exp = np.zeros((*self.shape_mesh, self.nstates), dtype=float)
            # loop over all mesh points
            for idx in np.ndindex(*self.shape_mesh):
                pos_exp_mat = self.position_matrix(mesh_idx=idx, state_idx=state_idx, pos_dir=pos_dir).diagonal()
                pos_exp[idx] = np.array(np.real(pos_exp_mat), dtype=float)

            return pos_exp
        else:
            pos_exp_mat = self.position_matrix(mesh_idx=mesh_idx, state_idx=state_idx, pos_dir=pos_dir).diagonal()
            return np.array(np.real(pos_exp_mat), dtype=float)

    def position_hwf(
            self,
            pos_dir,  
            mesh_idx, 
            state_idx = None,
            hwf_evec: bool = False, 
            basis: str = "wavefunction"
            ):
        r"""Eigenvalues and eigenvectors of the position operator in a given basis.

        Parameters
        ----------
        mesh_idx: array-like of int
            Set of integers specifying the index of interest in the mesh.

        pos_dir: int
            Direction along which to compute the position operator.

            .. versionchanged:: 2.0.0
                Renamed from ``dir`` to ``pos_dir`` to avoid conflict with built-in Python function `dir()`.

        state_idx: array-like, optional
            List of states to be included. If not specified, all states are included.

            .. versionchanged:: 2.0.0
                Renamed from ``occ``. The band indices are not required to be
                occupied bands only. The default behavior is to include all bands,
                and the ``"all"`` option has been removed.

        hwf_evec: bool, optional
            Default is `False`. If `True`, return the eigenvectors along with eigenvalues
            of the position operator.
        basis: {"orbital", "wavefunction", "bloch"}, optional
            The basis in which to compute the position operator.

        Returns
        -------
        hwfc : np.ndarray
            Eigenvalues of the position operator matrix :math:`X`
            (also called hybrid Wannier function centers). 
            Length of this vector equals number of bands given in *evec* input
            array.  Hybrid Wannier function centers are ordered in ascending order.
            Note that in general `n`-th hwfc does not correspond to `n`-th electronic
            state `evec`.

        hwf : np.ndarray, optional
            Eigenvectors of the position operator matrix :math:`X`.
            (also called hybrid Wannier functions).  These are returned only if
            parameter ``hwf_evec=True``.

            The shape of this array is ``[h,x]`` or ``[h,x,s]`` depending on value of
            ``basis`` and ``spinful``.  

            - If ``basis = "bloch"`` then ``x`` refers to indices of
              Bloch states `evec`.  
            - If ``basis = "orbital"`` then ``x`` (or ``x`` and ``s``)
              correspond to orbital index (or orbital and spin index 
              if ``spinful=True``).

        See Also
        --------
        :ref:`haldane-hwf-nb` : For an example.
        position_matrix : For the definition of the matrix :math:`X`.
        position_expectation : For the position expectation value.
        :func:`pythtb.TBModel.position_hwf`

        Notes
        -----
        Similar to :func:`pythtb.TBModel.position_hwf`, except that
        
        in addition to specifying ``pos_dir``, one also has to specify ``mesh_idx``
        (mesh-point of interest) and ``state_idx`` (list of states to be included).

        For backwards compatibility the default value of *basis* here is different
        from that in :func:`pythtb.TBModel.position_hwf`.
        """
        # get position matrix
        pos_mat = self.position_matrix(mesh_idx=mesh_idx, state_idx=state_idx, pos_dir=pos_dir)
        evec = self.wfs[tuple(mesh_idx)][state_idx]

        # diagonalize position matrix
        if not hwf_evec:
            hwfc = np.linalg.eigvalsh(pos_mat)
            return hwfc
        else: 
            hwfc, hwf = np.linalg.eigh(pos_mat)
            # transpose so eig[i, :] is eigenvector for eval[i]-th eigenvalue
            hwf = hwf.T
            # convert to right basis
            if basis.lower().strip() in ["wavefunction", "bloch"]:
                return hwfc, hwf
            elif basis.lower().strip() == "orbital":
                if self.nspin == 1:
                    ret_hwf = np.zeros((hwf.shape[0], self.norb), dtype=complex)
                    for i in range(ret_hwf.shape[0]):
                        ret_hwf[i] = np.dot(hwf[i], evec) # project onto orbital basis
                    hwf = ret_hwf
                else:
                    ret_hwf = np.zeros((hwf.shape[0], self.norb * 2), dtype=complex)
                    # flatten spin indices
                    evec_use = evec.reshape([hwf.shape[0], self.norb * 2])
                    for i in range(ret_hwf.shape[0]):
                        ret_hwf[i] = np.dot(hwf[i], evec_use) # project onto orbital basis
                    # restore spin indices
                    hwf = ret_hwf.reshape([hwf.shape[0], self.norb, 2])
                return hwfc, hwf
            else:
                raise ValueError(
                    "Basis must be either 'wavefunction', 'bloch', or 'orbital'"
                )    

    # TODO allow for subbands
    def _trace_metric(self):
        P = self.tilde_states.projectors()
        _, Q_nbr = self.tilde_states._nbr_projectors(return_Q=True)

        nks = Q_nbr.shape[:-3]
        num_nnbrs = Q_nbr.shape[-3]
        w_b, _, _ = self.get_shell_weights(n_shell=1)

        T_kb = np.zeros((*nks, num_nnbrs), dtype=complex)
        for nbr_idx in range(num_nnbrs):  # nearest neighbors
            T_kb[..., nbr_idx] = np.trace(
                P[..., :, :] @ Q_nbr[..., nbr_idx, :, :], axis1=-1, axis2=-2
            )

        return w_b[0] * np.sum(T_kb, axis=-1)

    # TODO allow for subbands
    def _omega_til(self):
        Mmn = self._Mmn
        w_b, k_shell, idx_shell = self.get_shell_weights(n_shell=1)
        w_b = w_b[0]
        k_shell = k_shell[0]

        nks = Mmn.shape[:-3]
        Nk = np.prod(nks)
        k_axes = tuple([i for i in range(len(nks))])

        diag_M = np.diagonal(Mmn, axis1=-1, axis2=-2)
        log_diag_M_imag = np.log(diag_M).imag
        abs_diag_M_sq = abs(diag_M) ** 2

        r_n = -(1 / Nk) * w_b * np.sum(log_diag_M_imag, axis=k_axes).T @ k_shell

        Omega_tilde = (
            (1 / Nk)
            * w_b
            * (
                np.sum((-log_diag_M_imag - k_shell @ r_n.T) ** 2)
                + np.sum(abs(Mmn) ** 2)
                - np.sum(abs_diag_M_sq)
            )
        )
        return Omega_tilde

    # def _interp_op(self, O_k, k_path, plaq=False):
    #     k_mesh = np.copy(self.mesh.grid)
    #     k_idx_arr = self.mesh.idx_arr
    #     nks = self.mesh.shape_k
    #     dim_k = len(nks)
    #     Nk = np.prod([nks])

    #     supercell = list(
    #         product(
    #             *[range(-int((nk - nk % 2) / 2), int((nk - nk % 2) / 2)) for nk in nks]
    #         )
    #     )

    #     if plaq:
    #         # shift by half a mesh point to get the center of the plaquette
    #         k_mesh += np.array([(1 / nk) / 2 for nk in nks])

    #     # Fourier transform to real space
    #     O_R = np.zeros((len(supercell), *O_k.shape[dim_k:]), dtype=complex)
    #     for idx, pos in enumerate(supercell):
    #         for k_idx in k_idx_arr:
    #             R_vec = np.array(pos)
    #             phase = np.exp(-1j * 2 * np.pi * np.vdot(k_mesh[k_idx], R_vec))
    #             O_R[idx] += O_k[k_idx] * phase / Nk

    #     # interpolate to arbitrary k
    #     O_k_interp = np.zeros((k_path.shape[0], *O_k.shape[dim_k:]), dtype=complex)
    #     for k_idx, k in enumerate(k_path):
    #         for idx, pos in enumerate(supercell):
    #             R_vec = np.array(pos)
    #             phase = np.exp(1j * 2 * np.pi * np.vdot(k, R_vec))
    #             O_k_interp[k_idx] += O_R[idx] * phase

    #     return O_k_interp

    def _interp_energy(self, k_path, return_eigvecs=False):
        H_k_proj = self.get_proj_ham()
        H_k_interp = self.interp_op(H_k_proj, k_path)
        if return_eigvecs:
            u_k_interp = self.interp_op(self._u_nk, k_path)
            eigvals_interp, eigvecs_interp = np.linalg.eigh(H_k_interp)
            eigvecs_interp = np.einsum(
                "...ij, ...ik -> ...jk", u_k_interp, eigvecs_interp
            )
            eigvecs_interp = np.transpose(eigvecs_interp, axes=[0, 2, 1])
            return eigvals_interp, eigvecs_interp
        else:
            eigvals_interp = np.linalg.eigvalsh(H_k_interp)
            return eigvals_interp

    # TODO allow for subbands
    def _get_proj_ham(self):
        if not hasattr(self, "H_k_proj"):
            self.set_Bloch_ham()
        H_k_proj = self.u_nk.conj() @ self.H_k @ np.swapaxes(self.u_nk, -1, -2)
        return H_k_proj


def _no_2pi(phi, ref):
    """Shift phase phi by integer multiples of 2π so it is closest to ref."""
    while abs(ref-phi)>np.pi:
        if ref-phi>np.pi:
            phi+=2.0*np.pi
        elif ref-phi<-1.0*np.pi:
            phi-=2.0*np.pi
    return phi


def _array_phases_cont(arr_pha, clos):
    """Reads in 2d array of phases arr_pha and enforces continuity along the first index,
    i.e., no 2π jumps. First row is made as close to clos as possible."""
    ret = np.zeros_like(arr_pha)
    for i in range(arr_pha.shape[0]):
        cmpr = clos if i == 0 else ret[i-1, :]
        avail = list(range(arr_pha.shape[1]))
        for j in range(cmpr.shape[0]):
            best_k, min_dist = None, 1e10
            for k in avail:
                cur_dist = np.abs(np.exp(1j * cmpr[j]) - np.exp(1j * arr_pha[i, k]))
                if cur_dist <= min_dist:
                    min_dist = cur_dist
                    best_k = k
            avail.remove(best_k)
            ret[i, j] = _no_2pi(arr_pha[i, best_k], cmpr[j])
    return ret

def _one_phase_cont(pha, clos):
    """Reads in 1d array of numbers *pha* and makes sure that they are
    continuous, i.e., that there are no jumps of 2pi. First number is
    made as close to *clos* as possible."""
    ret = np.copy(pha)
    # go through entire list and "iron out" 2pi jumps
    for i in range(len(ret)):
        # which number to compare to
        if i == 0: 
            cmpr = clos
        else: 
            cmpr = ret[i-1]
        # make sure there are no 2pi jumps
        ret[i] = _no_2pi(ret[i], cmpr)
    return ret
