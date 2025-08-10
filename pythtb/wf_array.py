from .utils import _is_int, _offdiag_approximation_warning_and_stop
from .tb_model import TBModel
from .mesh import Mesh
import numpy as np
import copy  # for deepcopying
from itertools import product
from functools import partial
import logging

from tensorflow import convert_to_tensor
from tensorflow import complex64 as tfcomplex64
from tensorflow.linalg import eigh as tfeigh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ["WFArray"]

class WFArray:
    r"""

    This class is used to store and manipulate an array of
    wavefunctions of a tight-binding model
    :class:`pythtb.TBModel` on a regular or non-regular grid.
    These could be the Bloch energy eigenstates of the
    model, but could also be a subset of Bloch bands, 
    a set of hybrid Wannier functions for a ribbon or slab, 
    or any other set of wavefunctions that are expressed in terms 
    of the underlying basis orbitals. It provides methods that can
    be used to calculate Berry phases, Berry curvatures,
    first Chern numbers, etc.

    The wavevectors stored in *WFArray* are typically Hamiltonian
    eigenstates (e.g., Bloch functions for k-space arrays),
    with the *state* index running over all bands. However, a
    *WFArray* object can also be used for other purposes, such
    as to store only a restricted set of Bloch states (e.g.,
    just the occupied ones); a set of modified Bloch states
    (e.g., premultiplied by a position, velocity, or Hamiltonian
    operator); or for hybrid Wannier functions (i.e., eigenstates
    of a position operator in a nonperiodic direction).

    *Regular k-space grid*:
    If the grid is a regular k-mesh (no parametric dimensions),
    a single call to the function :func:`pythtb.WFArray.solve_on_grid` 
    will both construct a k-mesh that uniformly covers the Brillouin zone, 
    and populate it with the energy eigenvectors computed on this grid.
    This function will ensure that the last point along each k-dimension is 
    the same Bloch function as the first one multiplied by a phase factor to
    ensure the periodic boundary conditions are satisfied (see notes below).

    *Parametric or irregular k-space grid grid*:
    An irregular grid of points, or a grid that includes also
    one or more parametric dimensions, can be populated manually
    using the ``[]`` operator (see example below). The wavefunctions
    above are expected to be in the format `evec[state, orbital]`
    (or `evec[state, orbital, spin]` for the spinfull calculation).

    Parameters
    ----------

    model : :class:`pythtb.TBModel`
        A :class:`pythtb.TBModel` representing
        the tight-binding model associated with this array of eigenvectors.

    mesh_size: list, tuple
        A list or tuple specifying the size of the mesh of points
        in the order of reciprocal-space and/or parametric directions.

    nstates : int, optional
        Optional parameter specifying the number of states
        packed into the *WFArray* at each point on the mesh. Defaults
        to all states (i.e., `norb*nspin`).

    See Also
    --------
    :ref:`haldane-bp-nb` : For an example of using WFArray on a regular grid of points in k-space.

    :ref:`cone-nb` : For an example of using WFArray on a non-regular grid of points in k-space.

    :ref:`3site-cycle-nb` : For an example of using `WFArray` on a non-regular grid of points in parameter space.
        This example shows how one of the directions of *WFArray* object need not be a k-vector direction, 
        but can instead be a Hamiltonian parameter :math:`\lambda`. See also discussion after equation 4.1 in
        :ref:`formalism`.

    :ref:`cubic-slab-hwf-nb` : For an example of using `WFArray` to store hybrid Wannier functions.

    :func:`pythtb.TBModel.solve_ham`

    :ref:`formalism`

    Notes
    -----
    When using :func:`pythtb.WFArray.solve_on_grid` the last wavefunction along each mesh dimension
    is stored according the the boundary conditions 

    .. math::

        u_{n, \mathbf{k} + \mathbf{G}}(\mathbf{r}) = e^{-i \mathbf{G} \cdot \mathbf{r}} u_{n, \mathbf{k}}(\mathbf{r})

    where :math:`\mathbf{G}` is a reciprocal lattice vector and :math:`\mathbf{r}` is the position vector.
    See section 4.4 in :download:`notes on tight-binding formalism </misc/pythtb-formalism.pdf>` for more details.


    If WFArray is used for closed paths, either in a
    reciprocal-space or parametric direction, then one needs to
    include both the starting and ending eigenfunctions even though
    they are physically equivalent. If the array dimension in
    question is a k-vector direction and the path traverses the
    Brillouin zone in a primitive reciprocal-lattice direction,
    :func:`pythtb.WFArray.impose_pbc` can be used to associate
    the starting and ending points with each other. If it is a
    non-winding loop in k-space or a loop in parameter space,
    then :func:`pythtb.WFArray.impose_loop` can be used instead.

    Examples
    --------
    Construct `WFArray` capable of storing an 11x21 array of
    wavefunctions

    >>> wf = WFArray(tb, [11, 21])

    Populate this `WFArray` with regular grid of points in
    Brillouin zone
    
    >>> wf.solve_on_grid([0.0, 0.0])

    Compute set of eigenvectors at one k-point

    >>> eval, evec = tb.solve_one([kx, ky], eig_vectors = True)
    
    Store it manually into a specified location in the array

    >>> wf[3,4] = evec
    
    To access the eigenvectors from the same position

    >>> print(wf[3,4])

    """

    def __init__(self, model: TBModel, mesh: Mesh, nstates=None):
        self._model = model

        if model.dim_k != mesh.dim_k:
            raise ValueError(f"Model dim_k ({model.dim_k}) does not match mesh dim_k ({mesh.dim_k})")

        # check that model is of type TBModel
        if not isinstance(model, TBModel):
            raise TypeError("model must be of type TBModel")

        # check that mesh is of type Mesh
        if not isinstance(mesh, Mesh):
            raise TypeError("mesh must be of type Mesh")
        
        # store mesh
        self._mesh = mesh
        
        # derive mesh dimensions from the Mesh object
        # mesh.grid has shape (*dims, coord_dim)
        self._mesh_shape = np.array(self._mesh.grid.shape[:-1], dtype=int)
        self._ndims = self._mesh_shape.size

        # ensure each mesh dimension is at least 2
        # all dimensions should be 2 or larger, because pbc can be used
        if True in (self._mesh_shape <= 1).tolist():
            raise ValueError(
                "Dimension of WFArray object in each direction must be 2 or larger."
            )

        # number of electronic states for each k-point
        if nstates is None:
            self._nstates = self.model.nstate  # this = norb*nspin = no. of bands
            # note: 'None' means to use the default, which is all bands!
        else:
            if not _is_int(nstates):
                raise TypeError("Argument nstates is not an integer.")
            self._nstates = nstates  # set by optional argument

        # number of spin components
        self._nspin = self.model.nspin
        # number of orbitals
        self._norb = self.model.norb
        # store orbitals from the model
        self._orb = self.model.orb_vecs

        self._pbc_axes = []  # axes along which periodic boundary conditions are imposed
        self._loop_axes = []  # axes along which loops are imposed
       
        # store wavefunctions in the form [kx_index, ky_index,..., state, orb, spin]
        self._wfs = np.empty(self.shape, dtype=complex)
        self._energies = np.empty(self.mesh_shape + (self.nstates,), dtype=float)

    def __getitem__(self, key):
        self._check_key(key)
        return self._wfs[key]

    def __setitem__(self, key, value):
        self._check_key(key)
        self._wfs[key] = np.array(value, dtype=complex)

    def _check_key(self, key):
        # key is an index list specifying the grid point of interest
        if self._ndims == 1:
            if isinstance(key, (tuple, list, np.ndarray)):
                assert len(key) == 1, "Key should be an integer or a tuple of length 1!"
                key = key[0]  # convert to integer
            elif not isinstance(key, (int, np.integer)):
                raise TypeError("Key should be an integer!")
            if key < (-1) * self._mesh_shape[0] or key >= self._mesh_shape[0]:
                raise IndexError("Key outside the range!")
        else:
            if len(key) != self._ndims:
                raise TypeError("Wrong dimensionality of key!")
            for i, k in enumerate(key):
                if not _is_int(k):
                    raise TypeError("Key should be set of integers!")
                if k < (-1) * self._mesh_shape[i] or k >= self._mesh_shape[i]:
                    raise IndexError("Key outside the range!")

    @property
    def wfs(self):
        """The wavefunctions stored in the *WFArray* object."""
        return self._wfs
    
    @property
    def energies(self):
        """Returns the energies of the states stored in the *WFArray*."""
        return getattr(self, "_energies", None)

    @property
    def hamiltonian(self):
        r"""Returns the Hamiltonian matrix in (k,\lambda)-space."""
        return getattr(self, "_H_k", None)
    
    @property
    def mesh(self):
        """The mesh object associated with the *WFArray*."""
        return self._mesh

    @property
    def shape(self):
        """The shape of the wavefunction array."""
        wfs_dim = np.copy(self.mesh_shape)
        wfs_dim = np.append(wfs_dim, self.nstates)
        wfs_dim = np.append(wfs_dim, self.norb)
        if self.nspin == 2:
            wfs_dim = np.append(wfs_dim, self.nspin)
        return tuple(wfs_dim)
    
    @property
    def dim_k(self):
        """The number of k-space dimensions of the *WFArray* object."""
        return self.model.dim_k

    @property
    def dim_lambda(self):
        """The number of lambda dimensions of the *WFArray* object."""
        return self.mesh.dim_param

    @property
    def mesh_shape(self):
        """The mesh dimensions of the *WFArray* object."""
        return self._mesh_shape

    @property
    def ndims(self):
        """The number of dimensions of the *WFArray* object."""
        return self._ndims

    @property
    def pbc_axes(self):
        """The axes along which periodic boundary conditions are imposed."""
        return self._pbc_axes

    @property
    def loop_axes(self):
        """The axes along which loops are imposed."""
        return self._loop_axes

    @property
    def nstates(self):
        """The number of states (or bands) stored in the *WFArray* object."""
        return self._nstates
    
    @nstates.setter
    def nstates(self, value):
        self._nstates = value

    @property
    def nspin(self):
        """The number of spin components stored in the *WFArray* object."""
        return self._nspin

    @property
    def norb(self):
        """The number of orbitals stored in the *WFArray* object."""
        return self._norb
    
    @property
    def nks(self):
        """The number of k-points stored in the *WFArray* mesh."""
        return self.mesh.shape_k
    
    @property
    def nlams(self):
        """The number of lambda points stored in the *WFArray* mesh."""
        return self.mesh.shape_param

    @property
    def model(self):
        """The underlying TBModel object associated with the *WFArray*."""
        return self._model

    @property
    def param_path(self):
        """The parameter path (e.g., k-points) along which the model was solved.
        This is only set if the model was solved along a path using :meth:`solve_on_path`."""
        return getattr(self, "_param_path", None)

    @property
    def k_flat(self):
        r"""Returns a flattened version of the k-mesh used in the *WFArray*."""
        k_grid = self.mesh.get_k_points()
        k_flat = k_grid.reshape(-1, k_grid.shape[-1])
        return k_flat

    @property
    def k_grid(self):
        r"""Returns the KMesh object associated with the *WFArray*."""
        k_grid = self.mesh.get_k_points()
        return k_grid
    
    def set_wfs(
        self, wfs, cell_periodic: bool = True, spin_flattened=False, set_projectors=True
    ):
        """
        Sets the Bloch and cell-periodic eigenstates as class attributes.

        Args:
            wfs (np.ndarray):
                Bloch (or cell-periodic) eigenstates defined on a semi-full k-mesh corresponding
                to nks passed during class instantiation. The mesh is assumed to exlude the
                endpoints, e.g. in reduced coordinates {k = [kx, ky, kz] | k_i in [0, 1)}.
        """
    
        if self.nspin == 2:
            if spin_flattened:
                self._nstates = wfs.shape[-2]
            else:
                self._nstates = wfs.shape[-3]
        else:
            self._nstates = wfs.shape[-2]

        wfs = wfs.reshape(self.shape)

        if cell_periodic:
            phases = self._get_phases(inverse=False)
            psi_wfs = wfs * phases
            self._u_wfs = wfs
            self._psi_wfs = psi_wfs
        else:
            phases = self._get_phases(inverse=True)
            u_wfs = wfs * phases
            self._u_wfs = u_wfs
            self._psi_wfs = wfs

        # if self.dim_lam == 0 and set_projectors:
        #     # overlap matrix
        #     self._M = self._get_self_overlap_mat()
        #     # band projectors
        #     self._set_projectors()

    # TODO: possibly get rid of nbr by storing boundary states
    def _set_projectors(self, state_idx=None):
        num_nnbrs = self.mesh.num_nnbrs
        nnbr_idx_shell = self.mesh.nnbr_idx_shell

        P, Q = self.get_projectors(state_idx=state_idx, return_Q=True)
        self._P = P
        self._Q = Q

        # NOTE: lambda friendly
        self._P_nbr = np.zeros(
            (P.shape[:-2] + (num_nnbrs,) + P.shape[-2:]), dtype=complex
        )
        self._Q_nbr = np.zeros_like(self._P_nbr)
        
        u_wfs = self.get_states(flatten_spin=True)
        u_wfs = u_wfs[..., state_idx, :] if state_idx is not None else u_wfs

        # TODO need shell to iterate over extra lambda dims also, shift accordingly
        for idx, idx_vec in enumerate(nnbr_idx_shell[0]):  # nearest neighbors
            # accounting for phase across the BZ boundary
            states_pbc = (
                np.roll(u_wfs, shift=tuple(-idx_vec), axis=self.k_axes)
                * self.k_mesh.bc_phase[..., idx, np.newaxis, :]
            )
            self._P_nbr[..., idx, :, :] = np.einsum(
                "...ni, ...nj -> ...ij", states_pbc, states_pbc.conj()
            )
            self._Q_nbr[..., idx, :, :] = (
                np.eye(self._n_orb * self._nspin) - self._P_nbr[..., idx, :, :]
            )


    # TODO: allow for subbands and possible lamda dim
    def _get_self_overlap_mat(self):
        """Compute the overlap matrix of the cell periodic eigenstates.

        Overlap matrix of the form

        M_{m,n}^{k, k+b} = < u_{m, k} | u_{n, k+b} >

        Assumes that the last u_wf along each periodic direction corresponds to the
        next to last k-point in the mesh (excludes endpoints).

        Returns:
            M (np.array):
                Overlap matrix with shape [*nks, num_nnbrs, n_states, n_states]
        """

        # Assumes only one shell for now
        _, idx_shell = self.k_mesh.get_k_shell(N_sh=1, report=False)
        idx_shell = idx_shell[0]
        bc_phase = self.k_mesh.bc_phase

        # TODO: Not lambda friendly
        # overlap matrix
        M = np.zeros(
            (*self.k_mesh.nks, len(idx_shell), self._n_states, self._n_states),
            dtype=complex,
        )

        if self._nspin == 2:
            u_wfs = self.get_states(flatten_spin=True)["Cell periodic"]
        else:
            u_wfs = self.get_states()["Cell periodic"]

        for idx, idx_vec in enumerate(idx_shell):  # nearest neighbors
            # introduce phases to states when k+b is across the BZ boundary
            states_pbc = (
                np.roll(
                    u_wfs,
                    shift=tuple(-idx_vec),
                    axis=[i for i in range(self.k_mesh.dim)],
                )
                * bc_phase[..., idx, np.newaxis, :]
            )
            M[..., idx, :, :] = np.einsum(
                "...mj, ...nj -> ...mn", u_wfs.conj(), states_pbc
            )

        return M

    def get_states(self, flatten_spin=False):
        """Returns states stored in the WFArray.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        flatten_spin : bool, optional
            If True, the spin and orbital indices are flattened into a single index and
            the shape of the returned states will be [nk1, ..., nkd, [n_lambda,] n_state, n_orb * n_spin].
            If False, the original shape is preserved, [nk1, ..., nkd, [n_lambda,] n_state, n_orb, n_spin].

        Returns
        -------
        states : np.ndarray
            The wavefunctions stored in the WFArray.
        """
        # shape is [nk1, ..., nkd, [n_lambda,] n_state, n_orb[, n_spin]
        wfs = self.wfs

        # flatten last two axes together to condense spin and orbital indices
        if flatten_spin and self.nspin == 2:
            wfs = wfs.reshape((*wfs.shape[:-2], -1))

        return wfs

    def get_bloch_states(self, flatten_spin=False):
        r"""Returns Bloch and cell-periodic states from the WFArray.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        flatten_spin : bool, optional
            If True, the spin and orbital indices are flattened into a single index and
            the shape of the returned states will be [nk1, ..., nkd, [n_lambda,] n_state, n_orb * n_spin].
            If False, the original shape is preserved, [nk1, ..., nkd, [n_lambda,] n_state, n_orb, n_spin].

        Returns
        -------
        states : dict
            A dictionary containing the "bloch" and "cell" states.  The returned dictionary 
            has the following keys:

            - "bloch": Bloch states (periodic in k-space) :math:`\psi_{n\mathbf{k}}(\mathbf{r})`

            - "cell": Cell-periodic states (periodic in real space) :math:`u_{n\mathbf{k}}(\mathbf{r})`

        See Also
        --------
        get_states : For obtaining the states stored on the mesh only.

        :ref:`formalism`

        Notes
        -----
        This function assumes that the WFArray is defined on a regular k-mesh.
        """
        # shape is [nk1, ..., nkd, [n_lambda,] n_state, n_orb[, n_spin]
        u_wfs = self.wfs

        # make sure that u_wfs is consistent with being defined on a regular k-mesh
        if self.ndims != self.model.dim_k:
            raise ValueError(
                f"WFArray is initialized for a mesh of dimensions {self.ndims}, "
                f"but the model has dim_k = {self.model.dim_k}. Bloch states assumes the"
                f"WFArray is defined on a regular k-mesh."
            )
        
        phases = self._get_phases(inverse=False)
        psi_wfs = self.wfs * phases

        # flatten last two axes together to condense spin and orbital indices
        if flatten_spin:
            u_wfs = u_wfs.reshape((*u_wfs.shape[:-2], -1))
            psi_wfs = psi_wfs.reshape((*psi_wfs.shape[:-2], -1))

        return_states = {
            "cell": u_wfs,
            "bloch": psi_wfs,
        }
        return return_states

    def get_projectors(self, state_idx=None, return_Q=False):
        r"""Returns the band projectors associated with the states in the WFArray.

        .. versionadded:: 2.0.0

        The band projectors are defined as the outer product of the wavefunctions:

        .. math::

            P_{n\mathbf{k}} = |u_{n\mathbf{k}}(\mathbf{r})\rangle \langle u_{n\mathbf{k}}(\mathbf{r})| \\
            Q_{n\mathbf{k}} = \mathbb{I} - P_{n\mathbf{k}}

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

        u_wfs = self.get_states(flatten_spin=True)

        if state_idx is not None:
            u_wfs = u_wfs[..., state_idx, :]

        # band projectors
        P = np.einsum("...ni, ...nj -> ...ij", u_wfs, u_wfs.conj())
        Q = np.eye(u_wfs.shape[-1]) - P

        if return_Q:
            return P, Q
        return P
    

    def set_ham(self, model_func=None, fixed_params: dict={}):
        r"""Sets the Hamiltonian for the wavefunction array.

        .. versionadded:: 2.0.0

        If a model function is provided, it is used to generate the Hamiltonian
        with the current mesh parameters. If no parameters are defined in the
        mesh, a warning is issued.

        Parameters
        ----------
        model_fxn : callable, optional
            A function that takes mesh parameters as input and returns a
            :class:`TBModel`.

        Notes
        ------
        If there are parameter dimensions in the Mesh, then the ``model_fxn`` must be used.
        The `axis_names` are used to name the adiabatic axes corresponding to the
        variables to be passed to the model function.

        If `axis_names` were not specified in the Mesh initialization, they will be
        set to the default values of 'k_i' and 'l_i', where i is the
        index of the axis.
        """

        if model_func is not None and self.mesh.num_param_axes == self.dim_lambda == 0:
            logger.warning("Model function is provided but no parameters are defined in Mesh.")
        if model_func is None and self.mesh.num_param_axes > 0:
            raise ValueError("Model function is required when mesh has parameters.")

        flat = self.mesh.flat
        n_orb = self.norb
        n_spin = self.nspin
        n_states = self.nstates

        if model_func is None:
            Hk = self.model.hamiltonian(k_pts=flat)

            if self._mesh.is_grid:
                shape_k = self._mesh.shape_k
                Hk = Hk.reshape(*shape_k, *Hk.shape[1:])
            
            self._H_k = Hk
            return

        model_gen = partial(model_func, **fixed_params)
        self.fixed_params = fixed_params
        self.model_func = model_func
        
        dim_lambda = self.mesh.dim_param
        lambda_shape = self.nlams
        lambda_keys = self.mesh.axis_names[self.mesh.num_k_axes:]
        lambda_ranges = list(self.mesh.get_param_ranges().values())

        if self.dim_k > 0:
            n_kpts = np.prod(self.nks)
            shape_k = self.mesh.shape_k
            k_flat = self.k_flat

            # Initialize storage for wavefunctions and energies
            if n_spin == 1:
                H_kl = np.zeros(( *lambda_shape, n_kpts, n_states, n_states), dtype=complex)
            elif n_spin == 2:
                H_kl = np.zeros(( *lambda_shape, n_kpts, n_orb, n_spin, n_orb, n_spin), dtype=complex)
        else:
            k_flat = None

            if n_spin == 1:
                H_kl = np.zeros(( *lambda_shape, n_orb, n_orb), dtype=complex)
            elif n_spin == 2:
                H_kl = np.zeros(( *lambda_shape, n_orb, n_spin, n_orb, n_spin), dtype=complex)

        for idx, param_set in enumerate(np.ndindex(*lambda_shape)):
            # kwargs for model_fxn with specified parameter values
            param_dict = {lambda_keys[i]: lambda_ranges[i][param_set] for i in range(dim_lambda)}

            # Generate the model with modified parameters
            modified_model = model_gen(**param_dict)

            ham = modified_model.hamiltonian(k_pts=k_flat)

            H_kl[param_set] = ham

        if self.dim_k > 0:
            # Reshape n_kpts into the k-grid shape, then move lambda axes behind k-axes
            if n_spin == 1:
                H_kl = H_kl.reshape(*lambda_shape, *shape_k, n_states, n_states)
            elif n_spin == 2:
                H_kl = H_kl.reshape(*lambda_shape, *shape_k, n_states, n_orb, n_spin)
        
            # Current axes: [lambda_axes..., k_axes..., matrix_axes...]
            # Desired: [k_axes..., lambda_axes..., matrix_axes...]
            n_k_axes = len(shape_k)
            total_axes = H_kl.ndim
            perm = list(range(dim_lambda, dim_lambda + n_k_axes)) + list(range(0, dim_lambda)) + list(range(dim_lambda + n_k_axes, total_axes))
            H_kl = np.transpose(H_kl, axes=perm)

        else:
            if n_spin == 1:
                H_kl = H_kl.reshape(*lambda_shape, n_orb, n_orb)
            elif n_spin == 2:
                H_kl = H_kl.reshape(*lambda_shape, n_orb, n_spin, n_orb, n_spin)

        self._H_k = H_kl


    def solve_mesh(
            self, 
            model_func=None, 
            fixed_params: dict=None, 
            auto_detect_pbc=True,
            use_metal=True):
        """
        Solves for the eigenstates of the Bloch Hamiltonian defined by the model over a semi-full
        k-mesh, e.g. in 3D reduced coordinates {k = [kx, ky, kz] | k_i in [0, 1)}.

        Args:
            model_fxn (function, optional):
                A function that returns a model given a set of parameters.
            param_vals (dict, optional):
                Dictionary of parameter values for adiabatic evoltuion. Each key corresponds to
                a varying parameter and the values are arrays
        """
        if not hasattr(self, "_H_k"):
            self.set_ham(model_func=model_func, fixed_params=fixed_params)

        ham = self.hamiltonian

        # flatten spin axes if present
        if self.nspin == 2:
            ham = ham.reshape(*ham.shape[:-4], self.model.nstate, self.model.nstate)

        if not np.allclose(ham, ham.swapaxes(-1, -2).conj()):
            raise ValueError("Hamiltonian matrix is not Hermitian.")

        if use_metal:
            H_tf = convert_to_tensor(ham, dtype=tfcomplex64)
            eval, evec = tfeigh(H_tf)
            eval = eval.numpy()
            evec = evec.numpy()
        else:
            # diagonalize hamiltonian
            eval, evec = np.linalg.eigh(ham)
        # transpose matrix eig since otherwise it is confusing
        # now eig[i,:] is eigenvector for eval[i]-th eigenvalue
        evec = evec.swapaxes(-1, -2)
        evec = evec.reshape(*self.shape)

        self._wfs = evec
        self._energies = eval

        if self.nstates > 1:
            gaps = eval[..., 1:] - eval[..., :-1]
            self.gaps = gaps.min(axis=tuple(range(self.ndims)))
        else:
            self.gaps = None

        # auto-detect and impose PBC for each k-component axis
        if auto_detect_pbc:
            grid = self.mesh.grid
            shape = grid.shape[:-1]  # mesh shape
            for mesh_dir, k_comp in enumerate(self.mesh.k_axes):
                # Take slices along mesh_dir at beginning and end
                slc_first = [slice(None)] * len(shape)
                slc_last = [slice(None)] * len(shape)
                slc_first[mesh_dir] = 0
                slc_last[mesh_dir] = -1
                slc_first = tuple(slc_first) + (k_comp,)
                slc_last = tuple(slc_last) + (k_comp,)
                vals_first = grid[slc_first]
                vals_last = grid[slc_last]

                # Compare the arrays to detect PBC
                # Check if the difference is close to 1.0 (wraps BZ)
                delta = vals_last - vals_first
                if np.allclose(delta, 1.0, atol=1e-8):
                    logger.info(f"Auto-imposing PBC in mesh direction {mesh_dir} for k-component {k_comp}")
                    self.impose_pbc(mesh_dir, k_comp)


    def solve_on_grid(self, start_k=None):
        r"""Solve a tight-binding model on a regular mesh of k-points.

        The regular mesh of k-points covers the entire reciprocal-space unit cell. 
        Both points at the opposite sides of reciprocal-space unit cell are included 
        in the array. The spacing between points is defined by the mesh size specified 
        upon initialization. The end point is ``[start_k[0]+1, start_k[1]+1]``.

        Parameters
        ----------
        start_k : array-like (dim_k,), optional
            The starting point of the k-mesh in reciprocal space. If not specified,
            defaults to [0, 0] for 2D systems, [0, 0, 0] for 3D systems, etc. The
            starting point along each dimension must be in the range [-0.5, 0.5].

        Returns
        -------
        gaps : ndarray
            The minimal direct bandgap between `n`-th and `n+1`-th band on 
            all the k-points in the mesh.

        See Also
        --------
        :func:`pythtb.WFArray.impose_pbc`

        Notes
        -----
        One may have to use a dense k-mesh to resolve the highly-dispersive crossings.

        This function also automatically imposes periodic boundary
        conditions on the eigenfunctions. See also the discussion in
        :func:`pythtb.WFArray.impose_pbc`.

        Examples
        --------
        Solve eigenvectors on a regular grid anchored at ``[-0.5, -0.5]``
        so that the mesh is defined from ``[-0.5, -0.5]`` to ``[0.5, 0.5]``.

        >>> wf.solve_on_grid([-0.5, -0.5])

        """
        if start_k is None:
            start_k = [0] * self.ndims
            start_k = np.asarray(start_k, dtype=float)
        else:
            start_k = np.asarray(start_k, dtype=float)
            # check dimensionality
            if start_k.ndim != 1 or start_k.shape[0] != self.ndims:
                raise ValueError(
                    f"Expected start_k to have shape ({self.ndims},), "
                    f"but got shape {start_k.shape}."
                )
            
        # check values
        if np.any(start_k < -0.5) or np.any(start_k > 0.5):
            raise ValueError(
                f"Expected start_k to be in the range [-0.5, 0.5], "
                f"but got {start_k}."
            )
        
        # check dimensionality
        if self.ndims != self.model.dim_k:
            raise Exception(
                "If using solve_on_grid method, dimension of WFArray must equal"
                "dim_k of the tight-binding model."
            )

        # check number of states
        if self.nstates != self.model.nstate:
            raise ValueError(
                "\n\nWhen initializing this object, you specified nstates to be "
                + str(self.nstates)
                + ", but"
                "\nthis does not match the total number of bands specified in the model,"
                "\nwhich was "
                + str(self.model.nstate)
                + ".  If you wish to use the solve_on_grid method, do"
                "\nnot specify the nstates parameter when initializing this object.\n\n"
            )

        # store start_k
        self._start_k = start_k
        self._nks = tuple(
            nk - 1 for nk in self.mesh_shape
        )  # number of k-points in each direction

        # we use a mesh shape of (nk-1) because the last point in each direction will be
        # the same as the first one, so we only need (nk-1) points
        mesh_shape = tuple(nk - 1 for nk in self.mesh_shape)
        k_axes = [
            np.linspace(start_k[idx], start_k[idx] + 1, nk, endpoint=False)
            for idx, nk in enumerate(mesh_shape)
        ]
        # stack into a grid of shape (nk1-1, nk2-1, ..., nkd-1, dim_k)
        k_pts_sq = np.stack(np.meshgrid(*k_axes, indexing="ij"), axis=-1)
        # flatten the grid
        k_pts = k_pts_sq.reshape(-1, self.ndims)

        # store for later
        self._k_mesh_square = k_pts_sq
        self._k_mesh_flat = k_pts

        # solve the model on the k-mesh
        evals, evecs = self._model.solve_ham(k_pts, return_eigvecs=True)

        # reshape to back into a full (nk1, nk2, ..., nkd, nstate) mesh
        full_shape = tuple(mesh_shape) + (self.nstates,)
        evals = evals.reshape(full_shape)
        evecs = evecs.reshape(
            full_shape + (self.norb,) + ((self.nspin,) if self.nspin > 1 else ())
        )

        self._energies = evals  # store energies in the WFArray

        # reshape to square mesh: (nk1, nk2, ..., nkd-1, nstate, nstate) for evecs

        # Store all wavefunctions in the WFArray
        idx_arr = np.ndindex(*mesh_shape)
        for idx in idx_arr:
            self[idx] = evecs[idx]

        # impose periodic boundary conditions along all directions
        for dir in range(self.ndims):
            # impose periodic boundary conditions
            self.impose_pbc(dir, self.model.per[dir])

        if self.nstates > 1:
            gaps = evals[..., 1:] - evals[..., :-1]
            return gaps.min(axis=tuple(range(self.ndims)))
        else:
            return None

    def solve_on_one_point(self, kpt, mesh_indices):
        r"""Solve a tight-binding model on a single k-point.

        Solve a tight-binding model on a single k-point and store the eigenvectors
        in the *WFArray* object in the location specified by *mesh_indices*.

        Parameters
        ----------
        kpt : List specifying desired k-point to solve the model on.

        mesh_indices : List specifying associated set of mesh indices to assign the wavefunction to.

        Examples
        --------
        Solve eigenvectors on a sphere of radius kappa surrounding
        point `k_0` in 3d k-space and pack into a predefined 2d WFArray

        >>> n = 10
        >>> m = 10
        >>> wf = WFArray(model, [n, m])
        >>> kappa = 0.1
        >>> k_0 = [0, 0, 0]
        >>> for i in range(n + 1):
        >>>     for j in range(m + 1):
        >>>         theta = np.pi * i / n
        >>>         phi = 2 * np.pi * j / m
        >>>         kx = k_0[0] + kappa * np.sin(theta) * np.cos(phi)
        >>>         ky = k_0[1] + kappa * np.sin(theta) * np.sin(phi)
        >>>         kz = k_0[2] + kappa * np.cos(theta)
        >>>         wf.solve_on_one_point([kx, ky, kz], [i, j])
        """

        _, evec = self.model.solve_ham(kpt, return_eigvecs=True)
        if _is_int(mesh_indices):
            self._wfs[(mesh_indices,)] = evec
        else:
            self._wfs[tuple(mesh_indices)] = evec


    def choose_states(self, subset):
        r"""

        Create a new *WFArray* object containing a subset of the
        states in the original one.

        Parameters
        ----------
        subset : array-like of int 
            State indices to keep.

        Returns
        -------
        wf_new : WFArray
            Identical in all respects except that a subset of states have been kept.

        Examples
        --------
        Make new *WFArray* object containing only two states

        >>> wf_new = wf.choose_states([3, 5])

        """
        # make a full copy of the WFArray
        wf_new = copy.deepcopy(self)

        subset = np.array(subset, dtype=int)
        if subset.ndim != 1:
            raise ValueError("Parameter subset must be a one-dimensional array.")

        wf_new._nstates = subset.shape[0]
        if self._model.nspin == 2:
            wf_new._wfs = wf_new._wfs[..., subset, :, :]
        elif self._model.nspin == 1:
            wf_new._wfs = wf_new._wfs[..., subset, :]
        else:
            raise ValueError(
                "WFArray object can only handle spinless or spin-1/2 models."
            )

        return wf_new

    def empty_like(self, nstates=None):
        r"""Create a new empty *WFArray* object based on the original.

        Parameters
        ----------
        nstates : int, optional
            Specifies the number of states (or bands) to be stored in the array.
            Defaults to the same as the original *WFArray* object.

        Returns
        -------
        wf_new : WFArray
            WFArray except that array elements are uninitialized and 
            the number of states may have changed.

        Examples
        --------
        Make new empty WFArray object containing 6 bands per k-point
        
        >>> wf_new=wf.empty_like(nstates=6)

        """

        # make a full copy of the WFArray
        wf_new = copy.deepcopy(self)

        if nstates is None:
            wf_new._wfs = np.empty_like(wf_new._wfs)
        else:
            wf_shape = list(wf_new._wfs.shape)
            # modify numer of states (after k indices & before orb and spin)
            wf_shape[self._ndims] = nstates
            wf_new._wfs = np.empty_like(wf_new._wfs, shape=wf_shape)

        return wf_new
    
    
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
            Array of phase factors with shape [nk1, nk2, ..., nkd, norb, (nspin)].
            The last dimension is present only if the model has spin.
        """
        lam = -1 if inverse else 1
        nks = self.mesh_shape
        flat_mesh = self.mesh.flat

        per_dir = list(range(flat_mesh.shape[-1]))
        # slice second dimension to only keep only periodic dimensions in orb
        per_orb = self.model.orb_vecs[:, per_dir]

        # compute a list of phase factors: exp(pm i k . tau) of shape [k_val, orbital]
        phases = np.exp(lam * 1j * 2 * np.pi * per_orb @ flat_mesh.T, dtype=complex).T
        # reshape phases to have shape: [nk1, nk2, ..., nkd, norb]
        phases = phases.reshape(*nks, self.norb)

        if self._nspin == 1:
            # newaxis along state dimension
            phases = phases[..., np.newaxis, :]
        elif self._nspin == 2:
            # newaxis along state and spin dimension
            phases = phases[..., np.newaxis, :, np.newaxis]

        return phases


    def impose_pbc(self, mesh_dir: int, k_dir: int):
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

        See Also
        --------
        :ref:`3site-cycle-nb` : For an example where the periodic boundary 
        condition is applied only along one direction of *WFArray*.

        :ref:`formalism` : Section 4.4 and equation 4.18

        Notes
        -----
        If the *WFArray* object was populated using the
        :func:`pythtb.WFArray.solve_on_grid` method, this function
        should not be used since it will be called automatically by
        the code.

        This function will impose these periodic boundary conditions along
        one direction of the array. We are assuming that the k-point
        mesh increases by exactly one reciprocal lattice vector along
        this direction. This is currently **not** checked by the code;
        it is the responsibility of the user. Currently *WFArray*
        does not store the k-vectors on which the model was solved;
        it only stores the eigenvectors (wavefunctions).

        The eigenfunctions :math:`\psi_{n {\bf k}}` are by convention
        chosen to obey a periodic gauge, i.e.,
        :math:`\psi_{n,{\bf k+G}}=\psi_{n {\bf k}}` not only up to a
        phase, but they are also equal in phase. It follows that
        the cell-periodic Bloch functions are related by
        :math:`u_{n,{\bf k_0+G}}=e^{-i{\bf G}\cdot{\bf r}} u_{n {\bf k_0}}`.
        See :download:`notes on tight-binding formalism </misc/pythtb-formalism.pdf>` 
        section 4.4 and equation 4.18 for more detail.

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

        if k_dir not in self._model._per:
            raise Exception(
                "Periodic boundary condition can be specified only along periodic directions!"
            )

        if not _is_int(mesh_dir):
            raise TypeError("mesh_dir should be an integer!")
        if mesh_dir < 0 or mesh_dir >= self.ndims:
            raise IndexError("mesh_dir outside the range!")

        self._pbc_axes.append(mesh_dir)

        # Compute phase factors from orbital vectors dotted with G parallel to k_dir
        phase = np.exp(-2j * np.pi * self._orb[:, k_dir])
        phase = phase if self.nspin == 1 else phase[:, np.newaxis]

        # mesh_dir is the direction of the array along which we impose pbc
        # and it is also the direction of the k-vector along which we
        # impose pbc e.g.
        # mesh_dir=0 corresponds to kx, mesh_dir=1 to ky, etc.
        # mesh_dir=2 corresponds to lambda, etc.

        ### Define slices in a way that is general for arbitrary dimensions ###
        # Example: mesh_dir = 2 (2 defines the axis in Python counting)
        # add one for Python counting and one for ellipses
        slc_lft = [slice(None)] * (mesh_dir + 2)  # e.g., [:, :, :, :]
        slc_rt = [slice(None)] * (mesh_dir + 2)  # e.g., [:, :, :, :]
        # last element along mesh_dir axis
        slc_lft[mesh_dir] = -1  # e.g., [:, :, -1, :]
        # first element along mesh_dir axis
        slc_rt[mesh_dir] = 0  # e.g., [:, :, 0, :]
        # take all components of remaining axes with ellipses
        slc_lft[mesh_dir + 1] = Ellipsis  # e.g., [:, :, -1, ...]
        slc_rt[mesh_dir + 1] = Ellipsis  # e.g., [:, :, 0, ...]

        # Set the last point along mesh_dir axis equal to first
        # multiplied by the phase factor
        self._wfs[tuple(slc_lft)] = self._wfs[tuple(slc_rt)] * phase

    def impose_loop(self, mesh_dir):
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

        >>> wf.impose_loop(mesh_dir = 2)

        """
        if not _is_int(mesh_dir):
            raise TypeError("mesh_dir must be an integer.")
        if mesh_dir < 0 or mesh_dir >= self.ndims:
            raise ValueError(
                f"mesh_dir must be between 0 and {self.ndims-1}, got {mesh_dir}."
            )

        self._loop_axes.append(mesh_dir)

        slc_lft = [slice(None)] * (mesh_dir + 2)  # e.g., [:, :, :, :]
        slc_rt = [slice(None)] * (mesh_dir + 2)  # e.g., [:, :, :, :]

        slc_lft[mesh_dir] = -1  # e.g., [:, :, -1, :]
        slc_rt[mesh_dir] = 0  # e.g., [:, :, 0, :]
        slc_lft[mesh_dir + 1] = Ellipsis  # e.g., [:, :, -1, ...]
        slc_rt[mesh_dir + 1] = Ellipsis  # e.g., [:, :, 0, ...]
        # set the last point in the mesh_dir direction equal to the first one
        self._wfs[tuple(slc_lft)] = self._wfs[tuple(slc_rt)]

    def position_matrix(self, k_idx, occ, dir):
        r"""Position matrix for a given k-point and set of states.

        Position operator is defined in reduced coordinates.
        The returned object :math:`X` is

        .. math::

          X_{m n {\bf k}}^{\alpha} = \langle u_{m {\bf k}} \vert
          r^{\alpha} \vert u_{n {\bf k}} \rangle

        Here :math:`r^{\alpha}` is the position operator along direction
        :math:`\alpha` that is selected by `dir`.

        This routine can be used to compute the position matrix for a
        given k-point and set of states (which can be all states, or
        a specific subset).

        Parameters
        ----------
        k_idx: array-like of int 
            Set of integers specifying the k-point of interest in the mesh.
        occ: array-like, 'all'
            List of states to be included (can be 'all' to include all states).
        dir: int
            Direction along which to compute the position matrix.

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
        in addition to specifying `dir`, one also has to specify `k_idx` (k-point of interest) 
        and `occ` (list of states to be included, which can optionally be 'all').
        """

        # Check for special case of parameter occ
        if isinstance(occ, str) and occ.lower() == "all":
            occ = np.arange(self.nstates, dtype=int)
        elif isinstance(occ, (list, np.ndarray, tuple, range)):
            occ = list(occ)
            occ = np.array(occ, dtype=int)
        else:
            raise TypeError(
                "occ must be a list, numpy array, tuple, or 'all' defining "
                "band indices of itnterest."
            )

        if occ.ndim != 1:
            raise Exception(
                """\n\nParameter occ must be a one-dimensional array or string "All"."""
            )

        # check if model came from w90
        if not self._model._assume_position_operator_diagonal:
            _offdiag_approximation_warning_and_stop()
        
        evec = self.wfs[tuple(k_idx)][occ]
        return self.model.position_matrix(evec, dir)

    def position_expectation(self, dir, mesh_idx=None, occ='all'):
        r"""Position expectation value for a given k-point and set of states.

        These elements :math:`X_{n n}` can be interpreted as an
        average position of n-th Bloch state ``evec[n]`` along
        direction `dir`. 

        This routine can be used to compute the position expectation value for a
        given k-point and set of states (which can be all states, or
        a specific subset). 

        Parameters
        ----------
        dir: int
            Direction along which to compute the position expectation value.
        mesh_idx: array-like of int, optional
            Set of integers specifying the k-point of interest in the mesh.
        occ: array-like, 'all', optional
            List of states to be included (can be 'all' to include all states).

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
        in addition to specifying *dir*, one also has to specify *k_idx* (k-point of interest)
        and *occ* (list of states to be included, which can optionally be 'all').

        Generally speaking these centers are _not_
        hybrid Wannier function centers (which are instead
        returned by :func:`position_hwf`).
        """

        # Check for special case of parameter occ
        if isinstance(occ, str) and occ.lower() == "all":
            occ = np.arange(self.nstates, dtype=int)
        elif isinstance(occ, (list, np.ndarray, tuple, range)):
            occ = list(occ)
            occ = np.array(occ, dtype=int)
            if occ.ndim != 1:
                raise ValueError(
                    """Parameter occ must be a one-dimensional array or string "all"."""
                )
        else:
            raise TypeError(
                "occ must be a list, numpy array, tuple, or 'all' defining "
                "band indices of interest."
            )

        # check if model came from w90
        if not self.model._assume_position_operator_diagonal:
            _offdiag_approximation_warning_and_stop()
        
        if mesh_idx is None:
            pos_exp = np.zeros((*self.mesh_shape, self.nstates), dtype=float)
            # loop over all mesh points
            for idx in np.ndindex(*self.mesh_shape):
                evec = self.wfs[tuple(idx)][occ]
                pos_exp[idx] = self.model.position_expectation(evec, dir)
            return pos_exp
        else:
            evec = self.wfs[tuple(mesh_idx)][occ]
            return self.model.position_expectation(evec, dir)

    def position_hwf(self, mesh_idx, occ, dir, hwf_evec=False, basis="wavefunction"):
        r"""Eigenvalues and eigenvectors of the position operator in a given basis.

        Parameters
        ----------
        k_idx: array-like of int
            Set of integers specifying the k-point of interest in the mesh.
        occ: array-like, 'all'
            List of states to be included (can be 'all' to include all states).
        dir: int
            Direction along which to compute the position operator.
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

        hwf : np.ndarray
            Eigenvectors of the position operator matrix :math:`X`.
            (also called hybrid Wannier functions).  These are returned only if
            parameter ``hwf_evec=True``.

            The shape of this array is ``[h,x]`` or ``[h,x,s]`` depending on value of
            `basis` and `nspin`.  
            
            - If `basis` is "bloch" then `x` refers to indices of
              Bloch states `evec`.  
            - If `basis` is "orbital" then `x` (or `x` and `s`)
              correspond to orbital index (or orbital and spin index if `nspin` is 2).

        See Also
        --------
        :ref:`haldane-hwf-nb` : For an example.
        position_matrix : For the definition of the matrix :math:`X`.
        position_expectation : For the position expectation value.
        :func:`pythtb.TBModel.position_hwf`

        Notes
        -----
        Similar to :func:`pythtb.TBModel.position_hwf`, except that
        in addition to specifying *dir*, one also has to specify
        *k_idx*, the k-point of interest, and *occ*, a list of states to
        be included (typically the occupied states).

        For backwards compatibility the default value of *basis* here is different
        from that in :func:`pythtb.TBModel.position_hwf`.
        """
        # Check for special case of parameter occ
        if isinstance(occ, str) and occ.lower() == "all":
            occ = np.arange(self.nstates, dtype=int)
        elif isinstance(occ, (list, np.ndarray, tuple, range)):
            occ = list(occ)
            occ = np.array(occ, dtype=int)
        else:
            raise TypeError(
                "occ must be a list, numpy array, tuple, or 'all' defining "
                "band indices of itnterest."
            )
        if occ.ndim != 1:
            raise Exception(
                """\n\nParameter occ must be a one-dimensional array or string "all"."""
            )

        # check if model came from w90
        if not self.model._assume_position_operator_diagonal:
            _offdiag_approximation_warning_and_stop()

        evec = self.wfs[tuple(mesh_idx)][occ]
        return self.model.position_hwf(evec, dir, hwf_evec, basis)

    def get_links(self, state_idx=None, dirs=None):
        r"""Compute the overlap links (unitary matrices) for the wavefunctions.

        .. versionadded:: 2.0.0

        The overlap links for the wavefunctions in the `WFArray` object
        along a given direction are defined as the unitary part of the overlap
        between the wavefunctions and their neighbors in the forward direction along each
        mesh directions. Specifcally, the overlap matrices are computed as

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
        dirs : list of int, optional
            List of directions along which to compute the links.
            If not provided, links will be computed for all directions in the mesh.

        Returns
        -------
        U_forward (np.ndarray):
            Array of shape [dim, nk1, nk2, ..., nkd, n_states, n_states]
            where dim is the number of dimensions of the mesh,
            (nk1, nk2, ..., nkd) are the sizes of the mesh in each dimension, 
            and n_states is the number of states in the *WFArray* object. The first 
            axis corresponds to :math:`\mu`, the last two axes are the matrix elements,
            and the remaining axes are the mesh points.

        Notes
        -----
        The last points in the mesh of `U_forward` should be treated carefully. Periodic boundary
        conditions are always implied here, so that the 0'th wavefunction is the forward neighbor of
        the last wavefunction (-1'st element) along each direction. If the `WFArray` mesh has already
        been defined with periodic boundary conditions, with either :func:`impose_pbc` or :func:`impose_loop`,
        then the last points are identified with the first points. This means the overlap links at the boundary
        should be disregarded, since the overlap is not between neighbors. If the last and first wavefunctions 
        are not neighbors, then the forward neighbor at the boundary is undefined and the value 
        of :math:`U_{nk}^{\mu}` at the boundary can again be disregarded. The only time these points should not be 
        disregarded is when the last and first wavefunctions are truly neighbors, which would only happen if
        the wavefunctions on the mesh were manually populated that way.
        """
        wfs = self.get_states(flatten_spin=True)

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

        if dirs is None:
            # If no specific directions are provided, compute links for all directions
            dirs = list(range(self.ndims))

        U_forward = []
        for mu in dirs:
            # print(f"Computing links for direction mu={mu}")
            wfs_shifted = np.roll(wfs, -1, axis=mu)

            # <u_nk| u_m k+delta_mu>
            ovr_mu = wfs.conj() @ wfs_shifted.swapaxes(-2, -1)

            U_forward_mu = np.zeros_like(ovr_mu, dtype=complex)
            V, _, Wh = np.linalg.svd(ovr_mu, full_matrices=False)
            U_forward_mu = V @ Wh
            U_forward.append(U_forward_mu)

        return np.array(U_forward)

    @staticmethod
    def wilson_loop(wfs_loop, evals=False):
        r"""Wilson loop unitary matrix

        .. versionadded:: 2.0.0
        
        Compute Wilson loop unitary matrix and its eigenvalues for multiband Berry phases.
        The Wilson loop is a geometric quantity that characterizes the topology of the
        band structure. It is defined as the product of the overlap matrices between
        neighboring wavefunctions in the loop. Specifically, it is given by

        .. math::

            U_{Wilson} = \prod_{n} U_{n}

        where :math:`U_{n}` is the unitary part of the overlap matrix between neighboring wavefunctions
        in the loop, and the index :math:`n` labels the position in the loop 
        (see :func:`get_links` for more details).

        Multiband Berry phases always returns numbers between :math:`-\pi` and :math:`\pi`.

        Parameters
        ----------
        wfs_loop : np.ndarray
            Has format [loop_idx, band, orbital(, spin)] and loop has to be one dimensional.
            Assumes that first and last loop-point are the same. Therefore if
            there are n wavefunctions in total, will calculate phase along n-1
            links only!
        berry_evals : bool, optional
            If berry_evals is True then will compute phases for
            individual states, these corresponds to 1d hybrid Wannier
            function centers. Otherwise just return one number, Berry phase.

        Returns
        -------
        np.ndarray
            If berry_evals is True then will return phases for individual states.
            If berry_evals is False then will return one number, the Berry phase.

        See Also
        --------
        :func:`berry_loop`
        :func:`get_links`
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
        if evals:
            evals = np.linalg.eigvals(U_wilson)  # Wilson loop eigenvalues
            eval_pha = -np.angle(evals)  # Multiband  Berrry phases
            return U_wilson, eval_pha
        else:
            return U_wilson

    @staticmethod
    def berry_loop(wfs_loop, evals=False):
        r"""Berry phase along a one-dimensional loop of wavefunctions.

        The Berry phase is computed as the logarithm of the determinant
        of the product of the overlap matrices between neighboring
        wavefunctions in the loop. In otherwords, the Berry phase is
        given by the formula:

        .. math::

            \phi = -\text{Im} \ln \det U_{\rm Wilson}

        where :math:`U` is the Wilson loop unitary matrix obtained from
        :func:`wilson_loop`. The Berry phase is returned as a
        single number, which is the total Berry phase for the loop.

        Parameters
        ----------
        wfs_loop : np.ndarray
            Wavefunctions in the loop, with shape `[loop_idx, band, orbital, spin]`. 
            The first and last points in the loop are assumed to be the same.
        evals : bool, optional
            Default is `False`. If `True`, will return the eigenvalues
            of the Wilson loop unitary matrix instead of the total Berry phase.
            The eigenvalues correspond to the "maximally localized Wannier centers" or
            "Wilson loop eigenvalues". If False, will return the total
            Berry phase for the loop.

        Returns
        -------
        np.ndarray, float:
            If evals is True, returns the eigenvalues of the Wilson loop
            unitary matrix, which are the Berry phases for each band.
            If evals is False, returns the total Berry phase for the loop,
            which is a single number.

        See Also
        --------
        :func:`berry_phase`
        :func:`get_links`
        :func:`wilson_loop`
        :ref:`formalism` : Section 4.5

        Notes
        -----
        The loop is assumed to be one-dimensional, meaning that the first 
        and last points in the loop are assumed to be the same, and the wavefunctions
        at these points are also assumed to be the same. The wavefunctions in the loop
        should be ordered such that the first point corresponds to the first wavefunction,
        the second point to the second wavefunction, and so on, up to the last point,
        which corresponds to the last wavefunction.
        """

        U_wilson = WFArray.wilson_loop(wfs_loop, evals=evals)

        if evals:
            hwf_centers = U_wilson[1]
            return hwf_centers
        else:
            berry_phase = -np.angle(np.linalg.det(U_wilson))
            return berry_phase

    def berry_phase(self, occ="All", dir=None, contin=True, berry_evals=False):
        r"""Berry phase along a given array direction.

        .. versionadded:: 2.0.0

        Computes the Berry phase along a given array direction
        and for a given set of states. These are typically the
        occupied Bloch states, but can also include unoccupied
        states if desired. 
        
        By default, the function returns the Berry phase traced
        over the specified set of bands, but optionally the individual
        phases of the eigenvalues of the global unitary rotation
        matrix (corresponding to "maximally localized Wannier
        centers" or "Wilson loop eigenvalues") can be requested
        by setting the parameter *berry_evals* to `True`.

        For a one-dimensional WFArray (i.e., a single string), the
        computed Berry phases are always chosen to be between :math:`-\pi` 
        and :math:`\pi`. For a higher dimensional WFArray, the Berry phase 
        is computed for each one-dimensional string of points, and an array of
        Berry phases is returned. The Berry phase for the first string
        (with lowest index) is always constrained to be between :math:`-\pi` and
        :math:`\pi`. The range of the remaining phases depends on the value of
        the input parameter `contin`.

        Parameters
        ----------
        occ : array-like, "all"
            Optional array of indices of states to be included
            in the subsequent calculations, typically the indices of
            bands considered occupied. If 'all', all states are selected.
            Default is all bands.

        dir : int
            Index of WFArray direction along which Berry phase is
            computed. This parameters needs not be specified for
            a one-dimensional WFArray.

        contin : bool, optional
            If True then the branch choice of the Berry phase (which is indeterminate
            modulo :math:`2\pi`) is made so that neighboring strings (in the
            direction of increasing index value) have as close as
            possible phases. The phase of the first string (with lowest
            index) is always constrained to be between :math:`-\pi` and :math:`\pi`.
            If False, the Berry phase for every string is constrained to be
            between :math:`-\pi` and :math:`\pi`. The default value is True.

        berry_evals : bool, optional
            If True then will compute and return the phases of the eigenvalues of the
            product of overlap matrices. (These numbers correspond also
            to hybrid Wannier function centers.) These phases are either
            forced to be between :math:`-\pi` and :math:`\pi` (if *contin* is *False*) or
            they are made to be continuous (if *contin* is True).

        Returns
        -------
        pha :
            If *berry_evals* is False (default value) then
            returns the Berry phase for each string. For a
            one-dimensional WFArray this is just one number. For a
            higher-dimensional `WFArray` *pha* contains one phase for
            each one-dimensional string in the following format. For
            example, if *WFArray* contains k-points on mesh with
            indices `[i,j,k]` and if direction along which Berry phase
            is computed is *dir=1* then *pha* will be two dimensional
            array with indices `[i,k]`, since Berry phase is computed
            along second direction. If *berry_evals* is True then for
            each string returns phases of all eigenvalues of the
            product of overlap matrices. In the convention used for
            previous example, *pha* in this case would have indices
            `[i,k,n]` where *n* refers to index of individual phase of
            the product matrix eigenvalue.

        See Also
        ---------
        :ref:`haldane-bp-nb` : For an example
        :ref:`cone-nb` : For an example
        :ref:`3site-cycle-nb` : For an example
        :func:`berry_loop` : For a function that computes Berry phase in a 1d loop.
        :ref:`formalism` : Sec. 4.5 for the discretized formula used to compute Berry phase.

        Notes
        -----
        For an array of size *N* in direction $dir$, the Berry phase
        is computed from the *N-1* inner products of neighboring
        eigenfunctions. This corresponds to an "open-path Berry
        phase" if the first and last points have no special
        relation. If they correspond to the same physical
        Hamiltonian, and have been properly aligned in phase using
        :func:`pythtb.WFArray.impose_pbc` or :func:`pythtb.WFArray.impose_loop`,
        then a closed-path Berry phase will be computed.

        In the case *occ* should range over all occupied bands,
        the occupied and unoccupied bands should be well separated in energy; 
        it is the responsibility of the user to check that this is satisfied.

        Examples
        ---------
        Computes Berry phases along second direction for three lowest
        occupied states. For example, if wf is threedimensional, then
        ``pha[2, 3]`` would correspond to Berry phase of string of states
        along ``wf[2, :, 3]``

        >>> pha = wf.berry_phase([0, 1, 2], 1)
        """
        # Get wavefunctions in the array, flattening spin if present
        # wfs is of shape [nk1, nk2, ..., nkd, nstate, nstate]
        wfs = self.get_states(flatten_spin=True)

        # Check for special case of parameter occ
        if isinstance(occ, str) and occ.lower() == "all":
            occ = np.arange(self.nstates, dtype=int)
        elif isinstance(occ, (list, np.ndarray, tuple, range)):
            occ = np.array(list(occ), dtype=int)
        else:
            raise TypeError(
                "occ must be a list, numpy array, tuple, or 'all' defining "
                "band indices of itnterest."
            )

        if occ.ndim != 1:
            raise ValueError(
                """Parameter occ must be a one-dimensional array or "all"."""
            )

        # check if model came from w90
        if not self.model._assume_position_operator_diagonal:
            _offdiag_approximation_warning_and_stop()

        # number of mesh dimensions is total dims minus band and orbital axes
        mesh_axes = wfs.ndim - 2
        # Validate dir parameter
        if dir is None:
            if mesh_axes != 1:
                raise ValueError(
                    "If dir is not specified, the mesh must be one-dimensional."
                )
            dir = 0
        if dir is not None and (dir < 0 or dir >= mesh_axes):
            raise ValueError("dir must be between 0 and number of mesh dimensions - 1")

        # Prepare wavefunctions: select occupied bands and bring loop dimension first
        wf = wfs[..., occ, :]
        wf = np.moveaxis(wf, dir, 0)  # shape: (N_loop, *rest, nbands)

        N_loop = wf.shape[0]
        rest_shape = wf.shape[1:-2]
        nbands = wf.shape[-2]
        norb = wf.shape[-1]

        # Allocate output with the un-flattened mesh shape
        if len(rest_shape) == 0:
            # Only a single string: compute directly
            ret = self.berry_loop(wf, evals=berry_evals)
        else:
            if berry_evals:
                ret = np.empty((*rest_shape, nbands), dtype=float)
            else:
                ret = np.empty(rest_shape, dtype=float)

            # Iterate over the remaining mesh indices without flattening
            for idx in np.ndindex(*rest_shape):
                slicer = (slice(None),) + idx + (slice(None), slice(None))  # (N_loop, nbands, norb)
                slice_wf = wf[slicer]
                val = self.berry_loop(slice_wf, evals=berry_evals)
                ret[idx] = val

        ret = np.array(ret)

        if contin:
            if len(rest_shape) == 0:
                # Make phases continuous for each band
                # ret = np.unwrap(ret, axis=0)
                pass

            elif berry_evals:
                # 2D case
                if ret.ndim == 2:
                    ret = _array_phases_cont(ret, ret[0])
                # 3D case
                elif ret.ndim == 3:
                    for i in range(ret.shape[1]):
                        if i == 0: 
                            clos = ret[0, 0, :]
                        else: 
                            clos = ret[0, i-1, :]
                        ret[:, i] = _array_phases_cont(ret[:, i], clos)
                elif self._dim_arr != 1:
                    raise ValueError("\n\nWrong dimensionality!")

            else:
                # 2D case
                if ret.ndim == 1:
                    ret = _one_phase_cont(ret, ret[0])
                # 3D case
                elif ret.ndim == 2:
                    for i in range(ret.shape[1]):
                        if i == 0: 
                            clos = ret[0, 0]
                        else: 
                            clos = ret[0, i-1]
                        ret[:, i] = _one_phase_cont(ret[:, i], clos)
                elif self._dim_arr != 1:
                    raise ValueError("Wrong dimensionality!")

        return ret

    def berry_flux(self, state_idx=None, plane=None, abelian=True):
        r"""Berry flux tensor.

        .. versionremoved:: 2.0.0
            The `individual_phases` parameter has been removed.

        The Berry flux tensor is a measure of the geometric phase acquired by
        the wavefunction as it is adiabatically transported around a closed loop
        in parameter space. The flux is computed around the small plaquettes in
        the parameter mesh, using the product of overlap matrices around the loops.
        The Berry flux is simply the integral of the Berry curvature around the plaquette
        loop. The (non-Abelian) Berry flux tensor is defined as 

        .. math::

            \mathcal{F}_{\mu\nu}(\mathbf{k}) = 
            \mathrm{Im}\ln\det[U_{\mu}(\mathbf{k}) U_{\nu}(\mathbf{k} + \hat{\mu}) 
            U_{\mu}^{-1}(\mathbf{k} + \hat{\nu}) U_{\nu}^{-1}(\mathbf{k})].
        
        The Berry curvature can be approximated by the flux by simply dividing by the
        area of the plaquette, approximating the flux as a constant over the small loop.

        .. math::

            \Omega_{\mu\nu}(\mathbf{k}) \approx \frac{\mathcal{F}_{\mu\nu}(\mathbf{k})}{A_{\mu\nu}},

        where :math:`A_{\mu\nu}` is the area of the plaquette in parameter space. The
        Abelian Berry flux is defined as the trace over the band indices of the non-Abelian
        Berry flux tensor.

        .. math::

            \mathcal{F}_{\mu\nu}(\mathbf{k}) = \sum_{n} (\mathcal{F}_{\mu\nu}(\mathbf{k}))_{n, n}.

        In the case of a 2-dimensional *WFArray* array calculates the
        Berry curvature over the entire plane.  In higher dimensional case
        it will compute flux over all 2-dimensional slices of a 
        higher-dimensional *WFArray*.

        Parameters
        ----------
        state_idx : array_like, optional
            Optional array of indices of states to be included
            in the subsequent calculations, typically the indices of
            bands considered occupied. If not specified, or None, all bands are
            included.

        plane : array_like, optional
            Array or tuple of two indices defining the axes in the
            WFArray mesh which the Berry flux is computed over. By default,
            all directions are considered, and the full Berry flux tensor is
            returned.

        abelian : bool, optional
            If *True* then the Berry flux is computed
            using the abelian formula, which corresponds to the band-traced
            non-Abelian Berry curvature. If *False* then the non-Abelian Berry
            flux tensor is computed. Default value is *True*.


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

        Examples
        --------
        Computes Berry curvature of first three bands in 2D model

        >>> flux = wf.berry_flux([0, 1, 2]) # shape: (dim1, dim2, nk1, nk2)
        >>> flux = wf.berry_flux([0, 1, 2], plane=(0, 1)) # shape: (nk1, nk2)
        >>> flux = wf.berry_flux([0, 1, 2], plane=(0, 1), abelian=False) # shape: (nk1, nk2, n_states, n_states)

        3D model example

        >>> flux = wf.berry_flux([0, 1, 2], plane=(0, 1)) # shape: (nk1, nk2, nk3)
        """
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

        n_states = len(state_idx)  # Number of states considered
        ndims = self.ndims  # Total dimensionality of adiabatic space: d
        n_param = list(
            self.mesh_shape
        )  # Number of points in adiabatic mesh: (nk1, nk2, ..., nkd)

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

        # Unique axes for periodic boundary conditions and loops
        # pbc_axes = list(set(self._pbc_axes + self._loop_axes))
        flux_shape = n_param
        for ax in range(ndims):
            flux_shape[ax] -= 1  # Remove last link in each periodic direction

        # Initialize the Berry flux array
        if plane is None:
            shape = (
                (ndims, ndims, *flux_shape, n_states, n_states)
                if not abelian
                else (ndims, ndims, *flux_shape)
            )
            berry_flux = np.zeros(shape, dtype=complex)
            dirs = list(range(ndims))
            plane_idxs = ndims
        else:
            p, q = plane  # Unpack plane directions

            shape = (*flux_shape, n_states, n_states) if not abelian else (*flux_shape,)
            berry_flux = np.zeros(shape, dtype=float)

            dirs = [p, q]
            plane_idxs = 2

        # U_forward: Overlaps <u_{nk} | u_{n, k+delta k_mu}>
        U_forward = self.get_links(state_idx=state_idx, dirs=dirs)

        # Compute Berry flux for each pair of states
        for mu in range(plane_idxs):
            for nu in range(mu + 1, plane_idxs):
                # print(f"Computing flux in plane: mu={mu}, nu={nu}")
                U_mu = U_forward[mu]
                U_nu = U_forward[nu]

                # Shift the links along the mu and nu directions
                U_nu_shift_mu = np.roll(U_nu, -1, axis=mu)
                U_mu_shift_nu = np.roll(U_mu, -1, axis=nu)

                # Wilson loops: W = U_{mu}(k_0) U_{nu}(k_0+delta_mu) U^{-1}_{mu}(k_0+delta_mu+delta_nu) U^{-1}_{nu}(k_0)
                U_wilson = np.matmul(
                    np.matmul(
                        np.matmul(U_mu, U_nu_shift_mu),
                        U_mu_shift_nu.conj().swapaxes(-1, -2),
                    ),
                    U_nu.conj().swapaxes(-1, -2),
                )

                # Remove edge loop, if pbc or loop is imposed then this is an extra plaquette that isn't wanted
                # without pbc or loop, this loop has no physical meaning
                for ax in range(ndims):
                    U_wilson = np.delete(U_wilson, -1, axis=ax)

                if not abelian:
                    eigvals, eigvecs = np.linalg.eig(U_wilson)
                    angles = -np.angle(eigvals)
                    angles_diag = np.einsum(
                        "...i, ij -> ...ij", angles, np.eye(angles.shape[-1])
                    )
                    eigvecs_inv = np.linalg.inv(eigvecs)
                    phases_plane = np.matmul(
                        np.matmul(eigvecs, angles_diag), eigvecs_inv
                    )
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
    
    def berry_curv(
        self,
        dirs=None,
        state_idx=None,
        non_abelian=False,
        return_flux=False,
        Kubo=False
    ):
        """Berry curvature tensor.

        .. versionadded:: 2.0.0

        The difference between this function and :func:`berry_flux` is that this function computes a dimensionful
        Berry curvature tensor, while :func:`berry_flux` is dimensionless. Effectively, this function divides by
        the area of the plaquette (in reduced units) in parameter space. The reduced units are set by the mesh spacing
        in each direction.

        Parameters
        ----------
        dirs : list or 'all', optional
            Directions in parameter space to compute the Berry curvature. If 'all', compute for all pairs of directions.
            By default None, which is equivalent to 'all'.
        state_idx : int or list of int, optional
            Index or indices of the states to compute the Berry curvature for. By default None, which computes for all states.
        non_abelian : bool, optional
            Whether to compute the non-Abelian Berry curvature. By default False.
        return_flux : bool, optional
            Whether to return the Berry flux instead of the curvature. By default False.
        Kubo : bool, optional
            Whether to use the Kubo formula for the Berry curvature. By default False.

        Returns
        -------
        np.ndarray
            Berry curvature tensor with shape depending on input parameters.
        """

        nks = self.nks  # Number of mesh points per direction
        n_lambda = self.nlams  # Number of adiabatic parameters
        dim_k = self.dim_k      # Number of k-space dimensions
        dim_lam = self.dim_lambda   # Number of adiabatic dimensions
        dim_total = dim_k + dim_lam  # Total number of dimensions

        if dim_k < 2:
            raise ValueError("Berry curvature only defined for dim_k >= 2.")

        if Kubo:
            # if not self.is_energy_eigstate:
            #     raise ValueError("Must be energy eigenstate to use Kubo formula.")
            # if not hasattr(self, "_u_wfs") or not hasattr(self, "energies"):
            #     raise ValueError("Must diagonalize model first to set wavefunctions and energies.")
            if state_idx is not None:
                print("Berry curvature in Kubo formula is for all occupied bands. Using half filling for occupied bands.")
            if dim_lam != 0:
                raise ValueError("Adiabatic dimensions not yet supported for Kubo formula.")
            if return_flux:
                print("Kubo formula doesn't support flux. Will return dimensionful Berry curvature only.")

            u_wfs = self.get_bloch_states(flatten_spin=True)["cell"]
            energies = self.energies
            # flatten k_dims
            u_wfs = u_wfs.reshape(-1, u_wfs.shape[-2], u_wfs.shape[-1])
            energies = energies.reshape(-1, energies.shape[-1])
            n_states = u_wfs.shape[-2]

            if n_states != self.model.nstate:
                raise ValueError("Wavefunctions must be defined for all bands, not just a subset.")

            k_mesh = self.mesh.flat
            occ_idx = np.arange(n_states // 2)
            abelian = not non_abelian
            if dirs is None:
                dirs = 'all'
                b_curv = self.model.berry_curvature(k_mesh, evals=energies, evecs=u_wfs, occ_idxs=occ_idx, abelian=abelian)
                b_curv = b_curv.reshape(*b_curv.shape[:2], *nks, *b_curv.shape[3:])
            else:
                b_curv = self.model.berry_curvature(k_mesh, evals=energies, evecs=u_wfs, occ_idxs=occ_idx, abelian=abelian, dirs=dirs)
                b_curv = b_curv.reshape(*nks, *b_curv.shape[3:])

            return b_curv

        Berry_flux = self.berry_flux(state_idx=state_idx, non_abelian=non_abelian)
        Berry_curv = np.zeros_like(Berry_flux, dtype=complex)

        # Get delta vectors for each dimension in parameter space
        recip_lat_vecs = self.model.get_recip_lat()  # Expressed in cartesian (x,y,z) coordinates
        dks = np.zeros((dim_total, dim_total))
        dks[:dim_k, :dim_k] = recip_lat_vecs / np.array([nk-1 for nk in self.nks])[:, None]

        # set delta lam to be difference between 0th and last points along each adiabatic axis
        if dim_lam != 0:
            param_points = self.mesh.get_param_points()
            delta_lam = np.zeros(dim_lam)
            for i in range(dim_lam):
                # shape of param_points is (nl1, nl2, ..., nld, d)
                # FIX: Need to index param_points correctly for each adiabatic axis
                delta_lam[i] = param_points[(0,)*i + (-1,) + (0,)*(dim_lam - i - 1), dim_k + i] - param_points[(0,)*dim_lam, dim_k + i]
                dks[dim_k + i, dim_k + i] = delta_lam[i] / (n_lambda[i] - 1)

        dim = Berry_flux.shape[0]  # Number of dimensions in parameter space
        # Divide by area elements for the (mu, nu)-plane
        for mu in range(dim):
            for nu in range(mu+1, dim):
                A = np.vstack([dks[mu], dks[nu]])
                # area_element = np.prod([np.linalg.norm(dk[i]), np.linalg.norm(dk[j])])
                area_element = np.sqrt(np.linalg.det(A @ A.T))

                # Divide flux by the area element to get approx curvature
                Berry_curv[mu, nu] = Berry_flux[mu, nu] / area_element
                Berry_curv[nu, mu] = Berry_flux[nu, mu] / area_element

        if dirs is not None:
            Berry_curv, Berry_flux = Berry_curv[dirs], Berry_flux[dirs]
        if return_flux:
            return Berry_curv, Berry_flux
        else:
            return Berry_curv
        

    def chern_num(self, plane=(0, 1), state_idx=None):
        r"""Computes the Chern number in the specified plane.

        .. versionadded:: 2.0.0

        The Chern number is computed as the integral of the Berry flux
        over the specified plane, divided by :math:`2 \pi`.

        .. math::
            C = \frac{1}{2\pi} \sum_{\mathbf{k}_{\mu}, \mathbf{k}_{\nu}} F_{\mu\nu}(\mathbf{k}).

        The plane :math:`(\mu, \nu)` is specified by `plane`, a tuple of two indices.

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
        >>> chern = wfs.chern_num(plane=(0, 1), state_idx=np.arange(n_occ))
        >>> print(chern.shape)
        (12,)  # shape of the Chern number array
        """
        if state_idx is None:
            state_idx = np.arange(self.nstates)  # assume half-filled occupied

        # shape of the Berry flux array: (nk1, nk2, ..., nkd)
        berry_flux = self.berry_flux(state_idx=state_idx, plane=plane, abelian=True)
        # shape of chern (if plane is (0,1)): (nk3, ..., nkd)
        chern = np.sum(berry_flux, axis=plane) / (2 * np.pi)

        return chern
    
    def get_axion_angle(
            self, 
            tf_list, 
            use_curv=True, 
            return_both=False, 
            order_fd=3,
            use_tf_speedup=True
        ):
        r"""Compute the axion angle from the Berry curvature.

        .. versionadded:: 2.0.0

        The axion angle is a topological invariant in three-dimensional insulators,
        related to the magnetoelectric response. It is defined as 

        .. math::

            \theta = -\frac{1}{4\pi} \epsilon^{\mu\nu\rho} 
            \int d^3k \, \text{Tr} 
            \left[ \mathcal{A}_{\mu} \partial_{\nu} \mathcal{A}_{\rho} 
            - \frac{2i}{3} \mathcal{A}_{\mu} \mathcal{A}_{\nu} \mathcal{A}_{\rho} \right]

        Alternatively, it may be expressed 

        .. math::

        \theta = -\frac{1}{4\pi} \epsilon^{\mu\nu\rho} 
            \int d^3k \, \text{Tr} 
            \left[ \frac{1}{2} \mathcal{A}_{\mu} \hat{\Omega}_{\nu\rho} 
            + \frac{i}{3} \mathcal{A}_{\mu} \mathcal{A}_{\nu} \mathcal{A}_{\rho} \right]

        The latter form has the benefit that errors introduced by finite difference approximations
        of :math:`\partial_{\nu} \mathcal{A}_{\rho}` can be avoided by using the Kubo formula for
        computing the Berry curvature :math:`\hat{\Omega}_{\nu\rho}` directly.

        The axion angle is only gauge-invariant modulo :math:`2\pi`, and its precise value can depend 
        on the choice of gauge. Because of this, we must fix the gauge
        choice by using the projection method, often used in the context of Wannier functions. This
        involves projecting the occupied (and conduction) states onto a set of trial wavefunctions to 
        obtain a smooth gauge. The trial wavefunctions should be chosen to have the same symmetry
        properties as the occupied states, and should be linearly independent to ensure a well-defined
        projection. They should be chosen to capture the essential features of the occupied subspace,
        such as the orbital character and spatial localization.

        Parameters
        ----------
        tf_list : list
            List of trial wavefunctions for projection.
        use_curv : bool, optional
            Whether to use the Berry curvature in the calculation. Default is True.
        return_both : bool, optional
            Whether to return both the Berry curvature and the axion angle. Default is False.
        order_fd : int, optional
            Order of the finite difference used in the calculation. Default is 3.
        use_tf_speedup : bool, optional
            Whether to use TensorFlow for speedup. Default is True.

        Returns
        -------
        theta : float
            The computed axion angle.

        Notes
        ------
        The axion angle is only defined for three-dimensional k-space models. It must be ensured by the user
        that the `WFArray` is defined on a 3D k-space mesh, and the underlying model is also 3D. It must also
        be ensured that the `WFArray` is populated by energy eigenstates on the mesh.

        If the system has a non-trivial :math:`\mathbb{Z}_2` index, there is an obstruction to choosing a smooth
        and periodic gauge choice. In this case, one must pick a set of trial wavefunctions that break time-reversal
        symmetry. 

        """
        from .wannier import Wannier
        from .utils import levi_civita, fin_diff

        if self.dim_k != 3:
            raise ValueError("Axion angle is only defined for 3D k-space models.")
        if self.dim_lambda != 0:
            raise ValueError("Adiabatic dimensions not yet supported for axion angle.")

        model = self.model
        wfa = self
        mesh = wfa.mesh
        flat_mesh = mesh.flat
        E_nk = wfa.energies

        nks = self.nks
        n_states = self.nstates   # Total number of states
        n_occ = n_states // 2     # Number of occupied states

        # velocity operator
        v_k = model.grad_ham(flat_mesh)
        # axes for each k-dimension, expand k-dimensions
        v_k = v_k.reshape(model.dim_r, *nks, n_states, n_states)

        # Energy eigensates (flattened spin and unflattened)
        state_dict_flat = wfa.get_bloch_states(flatten_spin=True)

        u_nk_flat = state_dict_flat["cell"]
        psi_nk_flat = state_dict_flat["bloch"]        

        # Getting spin flattened occupied and conduction states and energies
        psi_occ_flat = psi_nk_flat[..., :n_occ, :]
        psi_con_flat = psi_nk_flat[..., n_occ:, :]

        ### Projection
        WF = Wannier(model, wfa)

        # For tilde (projection) gauge states
        bloch_tilde = WF.tilde_states
        bloch_tilde.nstates = len(tf_list)  # Set number of states to number of trial functions

        twfs = WF.get_trial_wfs(tf_list)
        # Flatten spin axis
        twfs_flat = twfs.reshape((*twfs.shape[:1], -1))

        # Overlap matrix S_nm = <psi_nk| g_m> with occupied bands
        S_occ = np.einsum("...nj, mj -> ...nm", psi_occ_flat.conj(), twfs_flat)
        # Overlap matrix S_nm = <psi_nk| g_m> with conduction bands
        S_con = np.einsum("...nj, mj -> ...nm", psi_con_flat.conj(), twfs_flat)

        if use_tf_speedup:
            from tensorflow import convert_to_tensor
            from tensorflow import einsum as tfeinsum
            from tensorflow import complex64 as tfcomplex64
            from tensorflow.linalg import svd as tfsvd

            S_tf = convert_to_tensor(S_occ, dtype=tfcomplex64)

            # batched SVD on Metal
            D, W, V = tfsvd(S_tf, full_matrices=True)

            # back to NumPy for the rest
            W, D, V = W.numpy(), D.numpy(), V.numpy()
            Vh = V.conj().swapaxes(-1,-2)

        else:
            # Use NumPy SVD
            W, D, Vh = np.linalg.svd(S_occ, full_matrices=True)
            V = Vh.conj().swapaxes(-1, -2)

        D_mat = np.einsum("...i, ij -> ...ij", D, np.eye(D.shape[-1])) # Make a diagonal matrix
        # Unitary part
        U_SVD = W @ Vh
        # Semi-positive definite Hermitian part
        P = V @ D_mat @ Vh

        # Use unitary to rotate occupied bands into tilde basis for smooth gauge
        psi_tilde = np.einsum("...mn, ...mj -> ...nj", U_SVD, psi_occ_flat) # shape: (*nks, states, orbs*n_spin])
        # print(psi_tilde.shape)
        # set wfs in Bloch class
        bloch_tilde.set_wfs(psi_tilde, cell_periodic=False, spin_flattened=True, set_projectors=False)

        occ_idxs = np.arange(n_occ)
        cond_idxs = np.setdiff1d(np.arange(n_states), occ_idxs)  # Identify conduction bands

        # Velocity operator in energy eigenbasis
        evecs_conj_tf = u_nk_flat.conj()
        evecs_T_tf = u_nk_flat.swapaxes(-1,-2)  # (n_kpts, n_beta, n_state, n_state)

        v_k_rot = np.matmul(
                evecs_conj_tf[None, ...],  # (1, n_kpts, n_state, n_state)
                np.matmul(
                    v_k,                       # (dim_k, n_kpts, n_state, n_state)
                    evecs_T_tf[None, ...]  # (1, n_kpts, n_beta, n_state, n_state)
                )
            )

        # Compute energy denominators
        delta_E = E_nk[..., None, :] - E_nk[..., :, None]
        delta_E_occ_cond = np.take(np.take(delta_E, occ_idxs, axis=-2), cond_idxs, axis=-1)
        inv_delta_E_occ_cond_tf = 1 / delta_E_occ_cond

        v_occ_cond = np.take(np.take(v_k_rot, occ_idxs, axis=-2), cond_idxs, axis=-1)
        v_cond_occ = np.take(np.take(v_k_rot, cond_idxs, axis=-2), occ_idxs, axis=-1)
        v_occ_cond = v_occ_cond * -inv_delta_E_occ_cond_tf
        v_cond_occ = v_cond_occ * -inv_delta_E_occ_cond_tf

        # vhat
        vhat = v_occ_cond

        orb_vecs = model.orb_vecs
        r_mu_twfs = 2*np.pi * (orb_vecs.T[:, None, :, None] * twfs).reshape(3, 2, 4)
        # rhat
        rhat = np.einsum("...nj, amj -> a...nm", psi_occ_flat.conj(), r_mu_twfs)

        term =  S_occ.conj().swapaxes(-1,-2) @ (rhat + 1j*vhat@ S_con)
        term = Vh @ term @ Vh.conj().swapaxes(-1,-2)

        for a in range(term.shape[-2]):
            for b in range(term.shape[-1]):
                term[..., a, b] *= (1 / (D[..., a] + D[..., b]))

        term = Vh.conj().swapaxes(-1,-2) @ term @ Vh
        term  = term - term.conj().swapaxes(-1,-2)

        # Berry connection in projection gauge
        A_til = (
            U_SVD.conj().swapaxes(-1,-2) @ (rhat + 1j*vhat @ S_con)
            - term
        ) @ np.linalg.inv(P)

        # subtract one since endpoint is included
        dks = [1/(nk-1) for nk in nks]

        # CS Axion angle
        epsilon = levi_civita(3,3)

        if use_curv or return_both:
            omega_kubo = wfa.berry_curv(non_abelian=True, Kubo=True)
            omega_til = np.swapaxes(U_SVD.conj(), -1,-2) @ omega_kubo @ U_SVD

            if use_tf_speedup:
                A_til_tf = convert_to_tensor(A_til, tfcomplex64)
                Omega_til_tf = convert_to_tensor(omega_til, tfcomplex64)
                AOmega = tfeinsum('i...ab,jk...ba->ijk...', A_til_tf, Omega_til_tf)
                AAA = tfeinsum('i...ab,j...bc,k...ca->ijk...', A_til_tf, A_til_tf, A_til_tf)#.numpy()
                integrand = tfeinsum("ijk, ijk... -> ...", epsilon, (1/2) * AOmega + (1j/3) * AAA).numpy()
            else:
                AOmega = np.einsum('i...ab,jk...ba->ijk...', A_til, omega_til)
                AAA = np.einsum('i...ab,j...bc,k...ca->ijk...', A_til, A_til, A_til)
                integrand = np.einsum("ijk, ijk... -> ...", epsilon, (1/2) * AOmega + (1j/3) * AAA)

            # keep all but last k-point along each k-dimension
            integrand = integrand[:-1, :-1, :-1, ...]

            theta = -(4*np.pi)**(-1) * np.sum(integrand) * np.prod(dks)

            if not return_both:
                return theta.real

        A_til_par = A_til[:, :-1, :-1, :-1]
        # Finite difference of A
        parx_A = fin_diff(A_til_par, mu=1, dk_mu=dks[0], order_eps=order_fd)
        pary_A = fin_diff(A_til_par, mu=2, dk_mu=dks[1], order_eps=order_fd)
        parz_A = fin_diff(A_til_par, mu=3, dk_mu=dks[2], order_eps=order_fd)
        par_A = np.array([parx_A, pary_A, parz_A])

        if use_tf_speedup:
            A_til_tf = convert_to_tensor(A_til_par, tfcomplex64)
            AdA = tfeinsum('i...ab,jk...ba->ijk...', A_til_tf, par_A)#.numpy()
            AAA = tfeinsum('i...ab,j...bc,k...ca->ijk...', A_til_tf, A_til_tf, A_til_tf)#.numpy()
            integrand = tfeinsum("ijk, ijk... -> ...", epsilon, AdA - (2j/3) * AAA).numpy()
        else:
            AdA = np.einsum('i...ab,jk...ba->ijk...', A_til_par, par_A)
            AAA = np.einsum('i...ab,j...bc,k...ca->ijk...', A_til_par, A_til_par, A_til_par)
            integrand = np.einsum("ijk, ijk... -> ...", epsilon, AdA - (2j/3) * AAA)

        theta2 = -(4*np.pi)**(-1) * np.sum(integrand) * np.prod(dks)

        if return_both:
            return theta.real, theta2.real
        else:
            return theta2.real


def _no_2pi(phi, ref):
    """Shift phase phi by integer multiples of 2π so it is closest to ref."""
    return ref + ((phi - ref + np.pi) % (2 * np.pi) - np.pi)


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


class Bloch(WFArray):
    def __init__(self, model: TBModel, *param_dims):
        """Class for storing and manipulating Bloch like wavefunctions.

        Wavefunctions are defined on a semi-full reciprocal space mesh.
        """
        super().__init__(model, param_dims)
        assert (
            len(param_dims) >= model.dim_k
        ), "Number of dimensions must be >= number of reciprocal space dimensions"

        self.model: TBModel = model
        # model attributes
        self._n_orb = model.norb
        self._nspin = self.model.nspin
        self._n_states = self.nstates
        self.dim_k = model.dim_k
        self.nks = param_dims[: self.dim_k]
        # set k_mesh
        self.model.set_k_mesh(*self.nks)
        # stores k-points on a uniform mesh, calculates nearest neighbor points given the model lattice
        self.k_mesh: Mesh = model.k_mesh

        # adiabatic dimension
        self.dim_lam = len(param_dims) - self.dim_k
        self.n_lambda = param_dims[self.dim_k :]

        # Total adiabatic parameter space
        self.dim_param = self.dim_adia = self.dim_k + self.dim_lam
        self.n_param = self.n_adia = (*self.nks, *self.n_lambda)

        # periodic boundary conditions assumed True unless specified
        self.pbc_lam = True

        # axes indexes
        self.k_axes = tuple(range(self.dim_k))
        self.lambda_axes = tuple(range(self.dim_k, self.dim_param))

        if self._nspin == 2:
            self.spin_axis = -1
            self.orb_axis = -2
            self.state_axis = -3
        else:
            self.spin_axis = None
            self.orb_axis = -1
            self.state_axis = -2

        # wavefunction shapes
        if self.dim_lam > 0:
            if self._nspin == 2:
                self._wf_shape = (
                    *self.nks,
                    *self.n_lambda,
                    self._n_states,
                    self._n_orb,
                    self._nspin,
                )
            else:
                self._wf_shape = (
                    *self.nks,
                    *self.n_lambda,
                    self._n_states,
                    self._n_orb,
                )
        else:
            if self._nspin == 2:
                self._wf_shape = (*self.nks, self._n_states, self._n_orb, self._nspin)
            else:
                self._wf_shape = (*self.nks, self._n_states, self._n_orb)

        # self.set_Bloch_ham()

    @property
    def u_wfs(self):
        return self._u_wfs

    @property
    def psi_wfs(self):
        return self._psi_wfs

    def get_wf_axes(self):
        dict_axes = {
            "wf shape": self._wf_shape,
            "Number of axes": len(self._wf_shape),
            "k-axes": self.k_axes,
            "lambda-axes": self.lambda_axes,
            "spin-axis": self.spin_axis,
            "orbital axis": self.orb_axis,
            "state axis": self.state_axis,
        }
        return dict_axes

    def set_pbc_lam(self):
        self.pbc_lam = True

    def solve_model(self, model_fxn=None, lambda_vals=None):
        """
        Solves for the eigenstates of the Bloch Hamiltonian defined by the model over a semi-full
        k-mesh, e.g. in 3D reduced coordinates {k = [kx, ky, kz] | k_i in [0, 1)}.

        Args:
            model_fxn (function, optional):
                A function that returns a model given a set of parameters.
            param_vals (dict, optional):
                Dictionary of parameter values for adiabatic evoltuion. Each key corresponds to
                a varying parameter and the values are arrays
        """

        if lambda_vals is None:
            # compute eigenstates and eigenenergies on full k_mesh
            eigvals, eigvecs = self.model.solve_ham(
                self.k_mesh.flat_mesh, return_eigvecs=True
            )
            eigvecs = eigvecs.reshape(*self.k_mesh.nks, *eigvecs.shape[1:])
            eigvals = eigvals.reshape(*self.k_mesh.nks, *eigvals.shape[1:])
            self.set_wfs(eigvecs)
            self.energies = eigvals
            self.is_energy_eigstate = True
            return

        lambda_keys = list(lambda_vals.keys())
        lambda_ranges = list(lambda_vals.values())
        lambda_shape = tuple(len(vals) for vals in lambda_ranges)
        dim_lambda = len(lambda_keys)

        n_kpts = self.k_mesh.Nk
        n_orb = self.model.norb
        n_spin = self.model.nspin
        n_states = n_orb * n_spin

        # Initialize storage for wavefunctions and energies
        if n_spin == 1:
            u_wfs = np.zeros((*lambda_shape, n_kpts, n_states, n_states), dtype=complex)
        elif n_spin == 2:
            u_wfs = np.zeros(
                (*lambda_shape, n_kpts, n_states, n_orb, n_spin), dtype=complex
            )

        energies = np.zeros((*lambda_shape, n_kpts, n_states))

        for idx, param_set in enumerate(np.ndindex(*lambda_shape)):
            param_dict = {
                lambda_keys[i]: lambda_ranges[i][param_set] for i in range(dim_lambda)
            }

            # Generate the model with modified parameters
            modified_model = model_fxn(**param_dict)

            # Solve for eigenstates
            eigvals, eigvecs = modified_model.solve_ham(
                self.k_mesh.flat_mesh, return_eigvecs=True
            )

            # Store results
            energies[param_set] = eigvals
            u_wfs[param_set] = eigvecs

        # Reshape for compatibility with existing Berry curvature methods
        new_axes = (dim_lambda,) + tuple(range(dim_lambda)) + (dim_lambda + 1,)
        energies = np.transpose(energies, axes=new_axes)
        if self._nspin == 1:
            new_axes = (
                (dim_lambda,)
                + tuple(range(dim_lambda))
                + tuple(range(dim_lambda + 1, dim_lambda + 3))
            )
        else:
            new_axes = (
                (dim_lambda,)
                + tuple(range(dim_lambda))
                + tuple(range(dim_lambda + 1, dim_lambda + 4))
            )
        u_wfs = np.transpose(u_wfs, axes=new_axes)

        if self._nspin == 1:
            new_shape = (*self.k_mesh.nks, *lambda_shape, n_states, n_states)
        else:
            new_shape = (*self.k_mesh.nks, *lambda_shape, n_states, n_orb, n_spin)
        u_wfs = u_wfs.reshape(new_shape)
        energies = energies.reshape((*self.k_mesh.nks, *lambda_shape, n_states))

        self.set_wfs(u_wfs, cell_periodic=True)
        self.energies = energies
        self.is_energy_eigstate = True

    def get_nbr_projector(self, return_Q=False):
        assert hasattr(
            self, "_P_nbr"
        ), "Need to call `solve_model` or `set_wfs` to initialize Bloch states"
        if return_Q:
            return self._P_nbr, self._Q_nbr
        else:
            return self._P_nbr

    def get_energies(self):
        assert hasattr(
            self, "energies"
        ), "Need to call `solve_model` to initialize energies"
        return self.energies

    def get_Bloch_Ham(self):
        """Returns the Bloch Hamiltonian of the model defined over the semi-full k-mesh."""
        if hasattr(self, "H_k"):
            return self.H_k
        else:
            self.set_Bloch_ham()
            return self.H_k

    def get_overlap_mat(self):
        """Returns overlap matrix.

        Overlap matrix defined as M_{n,m,k,b} = <u_{n, k} | u_{m, k+b}>
        """
        assert hasattr(
            self, "_M"
        ), "Need to call `solve_model` or `set_wfs` to initialize overlap matrix"
        return self._M

    def set_wfs(
        self, wfs, cell_periodic: bool = True, spin_flattened=False, set_projectors=True
    ):
        """
        Sets the Bloch and cell-periodic eigenstates as class attributes.

        Args:
            wfs (np.ndarray):
                Bloch (or cell-periodic) eigenstates defined on a semi-full k-mesh corresponding
                to nks passed during class instantiation. The mesh is assumed to exlude the
                endpoints, e.g. in reduced coordinates {k = [kx, ky, kz] | k_i in [0, 1)}.
        """
        if spin_flattened and self._nspin == 2:
            self._n_states = wfs.shape[-2]
        else:
            self._n_states = wfs.shape[self.state_axis]

        if self.dim_lam > 0:
            if self._nspin == 2:
                self._wf_shape = (
                    *self.nks,
                    *self.n_lambda,
                    self._n_states,
                    self._n_orb,
                    self._nspin,
                )
            else:
                self._wf_shape = (
                    *self.nks,
                    *self.n_lambda,
                    self._n_states,
                    self._n_orb,
                )
        else:
            if self._nspin == 2:
                self._wf_shape = (*self.nks, self._n_states, self._n_orb, self._nspin)
            else:
                self._wf_shape = (*self.nks, self._n_states, self._n_orb)

        wfs = wfs.reshape(self._wf_shape)

        if cell_periodic:
            self._u_wfs = wfs
            self._psi_wfs = self._apply_phase(wfs)
        else:
            self._psi_wfs = wfs
            self._u_wfs = self._apply_phase(wfs, inverse=True)

        if self.dim_lam == 0 and set_projectors:
            # overlap matrix
            self._M = self._get_self_overlap_mat()
            # band projectors
            self._set_projectors()

    # TODO: allow for projectors onto subbands
    # TODO: possibly get rid of nbr by storing boundary states
    def _set_projectors(self):
        num_nnbrs = self.k_mesh.num_nnbrs
        nnbr_idx_shell = self.k_mesh.nnbr_idx_shell

        if self._nspin == 2:
            u_wfs = self.get_states(flatten_spin=True)["Cell periodic"]
        else:
            u_wfs = self.get_states()["Cell periodic"]

        # band projectors
        self._P = np.einsum("...ni, ...nj -> ...ij", u_wfs, u_wfs.conj())
        self._Q = np.eye(self._n_orb * self._nspin) - self._P

        # NOTE: lambda friendly
        self._P_nbr = np.zeros(
            (self._P.shape[:-2] + (num_nnbrs,) + self._P.shape[-2:]), dtype=complex
        )
        self._Q_nbr = np.zeros_like(self._P_nbr)

        # NOTE: not lambda friendly
        # self._P_nbr = np.zeros((*nks, num_nnbrs, self._n_orb*self._nspin, self._n_orb*self._nspin), dtype=complex)
        # self._Q_nbr = np.zeros((*nks, num_nnbrs, self._n_orb*self._nspin, self._n_orb*self._nspin), dtype=complex)

        # TODO need shell to iterate over extra lambda dims also, shift accordingly
        for idx, idx_vec in enumerate(nnbr_idx_shell[0]):  # nearest neighbors
            # accounting for phase across the BZ boundary
            states_pbc = (
                np.roll(u_wfs, shift=tuple(-idx_vec), axis=self.k_axes)
                * self.k_mesh.bc_phase[..., idx, np.newaxis, :]
            )
            self._P_nbr[..., idx, :, :] = np.einsum(
                "...ni, ...nj -> ...ij", states_pbc, states_pbc.conj()
            )
            self._Q_nbr[..., idx, :, :] = (
                np.eye(self._n_orb * self._nspin) - self._P_nbr[..., idx, :, :]
            )

        return

    # TODO: allow for subbands and possible lamda dim
    def _get_self_overlap_mat(self):
        """Compute the overlap matrix of the cell periodic eigenstates.

        Overlap matrix of the form

        M_{m,n}^{k, k+b} = < u_{m, k} | u_{n, k+b} >

        Assumes that the last u_wf along each periodic direction corresponds to the
        next to last k-point in the mesh (excludes endpoints).

        Returns:
            M (np.array):
                Overlap matrix with shape [*nks, num_nnbrs, n_states, n_states]
        """

        # Assumes only one shell for now
        _, idx_shell = self.k_mesh.get_k_shell(N_sh=1, report=False)
        idx_shell = idx_shell[0]
        bc_phase = self.k_mesh.bc_phase

        # TODO: Not lambda friendly
        # overlap matrix
        M = np.zeros(
            (*self.k_mesh.nks, len(idx_shell), self._n_states, self._n_states),
            dtype=complex,
        )

        if self._nspin == 2:
            u_wfs = self.get_states(flatten_spin=True)["Cell periodic"]
        else:
            u_wfs = self.get_states()["Cell periodic"]

        for idx, idx_vec in enumerate(idx_shell):  # nearest neighbors
            # introduce phases to states when k+b is across the BZ boundary
            states_pbc = (
                np.roll(
                    u_wfs,
                    shift=tuple(-idx_vec),
                    axis=[i for i in range(self.k_mesh.dim)],
                )
                * bc_phase[..., idx, np.newaxis, :]
            )
            M[..., idx, :, :] = np.einsum(
                "...mj, ...nj -> ...mn", u_wfs.conj(), states_pbc
            )

        return M

    def berry_curv(
        self,
        dirs=None,
        state_idx=None,
        non_abelian=False,
        delta_lam=1,
        return_flux=False,
        Kubo=False,
    ):

        nks = self.nks  # Number of mesh points per direction
        n_lambda = self.n_lambda
        dim_k = len(nks)  # Number of k-space dimensions
        dim_lam = len(n_lambda)  # Number of adiabatic dimensions
        dim_total = dim_k + dim_lam  # Total number of dimensions

        if dim_k < 2:
            raise ValueError("Berry curvature only defined for dim_k >= 2.")

        if Kubo:
            if not self.is_energy_eigstate:
                raise ValueError("Must be energy eigenstate to use Kubo formula.")
            if not hasattr(self, "_u_wfs") or not hasattr(self, "energies"):
                raise ValueError(
                    "Must diagonalize model first to set wavefunctions and energies."
                )
            if state_idx is not None:
                print(
                    "Berry curvature in Kubo formula is for all occupied bands. Using half filling for occupied bands."
                )
            if dim_lam != 0 or delta_lam != 1:
                raise ValueError(
                    "Adiabatic dimensions not yet supported for Kubo formula."
                )
            if return_flux:
                print(
                    "Kubo formula doesn't support flux. Will return dimensionful Berry curvature only."
                )

            u_wfs = self.get_states(flatten_spin=True)["Cell periodic"]
            energies = self.energies
            # flatten k_dims
            u_wfs = u_wfs.reshape(-1, u_wfs.shape[-2], u_wfs.shape[-1])
            energies = energies.reshape(-1, energies.shape[-1])
            n_states = u_wfs.shape[-2]

            if n_states != self.model.nstate:
                raise ValueError(
                    "Wavefunctions must be defined for all bands, not just a subset."
                )

            k_mesh = self.k_mesh.flat_mesh
            occ_idx = np.arange(n_states // 2)
            abelian = not non_abelian
            if dirs is None:
                dirs = "all"
                b_curv = self.model.berry_curvature(
                    k_mesh,
                    evals=energies,
                    evecs=u_wfs,
                    occ_idxs=occ_idx,
                    abelian=abelian,
                )
                b_curv = b_curv.reshape(*b_curv.shape[:2], *nks, *b_curv.shape[3:])
            else:
                b_curv = self.model.berry_curvature(
                    k_mesh,
                    evals=energies,
                    evecs=u_wfs,
                    occ_idxs=occ_idx,
                    abelian=abelian,
                    dirs=dirs,
                )
                b_curv = b_curv.reshape(*nks, *b_curv.shape[3:])

            return b_curv

        Berry_flux = self.berry_flux_plaq(state_idx=state_idx, non_abelian=non_abelian)
        Berry_curv = np.zeros_like(Berry_flux, dtype=complex)

        dim = Berry_flux.shape[0]  # Number of dimensions in parameter space
        recip_lat_vecs = (
            self.model.get_recip_lat()
        )  # Expressed in cartesian (x,y,z) coordinates

        dks = np.zeros((dim_total, dim_total))
        dks[:dim_k, :dim_k] = recip_lat_vecs / np.array(self.nks)[:, None]
        if self.dim_lam > 0:
            np.fill_diagonal(dks[dim_k:, dim_k:], delta_lam / np.array(self.n_lambda))

        # Divide by area elements for the (mu, nu)-plane
        for mu in range(dim):
            for nu in range(mu + 1, dim):
                A = np.vstack([dks[mu], dks[nu]])
                # area_element = np.prod([np.linalg.norm(dk[i]), np.linalg.norm(dk[j])])
                area_element = np.sqrt(np.linalg.det(A @ A.T))

                # Divide flux by the area element to get approx curvature
                Berry_curv[mu, nu] = Berry_flux[mu, nu] / area_element
                Berry_curv[nu, mu] = Berry_flux[nu, mu] / area_element

        if dirs is not None:
            Berry_curv, Berry_flux = Berry_curv[dirs], Berry_flux[dirs]
        if return_flux:
            return Berry_curv, Berry_flux
        else:
            return Berry_curv

    # TODO allow for subbands
    def trace_metric(self):
        P = self._P
        Q_nbr = self._Q_nbr

        nks = Q_nbr.shape[:-3]
        num_nnbrs = Q_nbr.shape[-3]
        w_b, _, _ = self.k_mesh.get_weights(N_sh=1)

        T_kb = np.zeros((*nks, num_nnbrs), dtype=complex)
        for nbr_idx in range(num_nnbrs):  # nearest neighbors
            T_kb[..., nbr_idx] = np.trace(
                P[..., :, :] @ Q_nbr[..., nbr_idx, :, :], axis1=-1, axis2=-2
            )

        return w_b[0] * np.sum(T_kb, axis=-1)

    # TODO allow for subbands
    def omega_til(self):
        M = self._M
        w_b, k_shell, idx_shell = self.k_mesh.get_weights(N_sh=1)
        w_b = w_b[0]
        k_shell = k_shell[0]

        nks = M.shape[:-3]
        Nk = np.prod(nks)
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

    def interp_op(self, O_k, k_path, plaq=False):
        k_mesh = np.copy(self.k_mesh.square_mesh)
        k_idx_arr = self.k_mesh.idx_arr
        nks = self.k_mesh.nks
        dim_k = len(nks)
        Nk = np.prod([nks])

        supercell = list(
            product(
                *[range(-int((nk - nk % 2) / 2), int((nk - nk % 2) / 2)) for nk in nks]
            )
        )

        if plaq:
            # shift by half a mesh point to get the center of the plaquette
            k_mesh += np.array([(1 / nk) / 2 for nk in nks])

        # Fourier transform to real space
        O_R = np.zeros((len(supercell), *O_k.shape[dim_k:]), dtype=complex)
        for idx, pos in enumerate(supercell):
            for k_idx in k_idx_arr:
                R_vec = np.array(pos)
                phase = np.exp(-1j * 2 * np.pi * np.vdot(k_mesh[k_idx], R_vec))
                O_R[idx] += O_k[k_idx] * phase / Nk

        # interpolate to arbitrary k
        O_k_interp = np.zeros((k_path.shape[0], *O_k.shape[dim_k:]), dtype=complex)
        for k_idx, k in enumerate(k_path):
            for idx, pos in enumerate(supercell):
                R_vec = np.array(pos)
                phase = np.exp(1j * 2 * np.pi * np.vdot(k, R_vec))
                O_k_interp[k_idx] += O_R[idx] * phase

        return O_k_interp

    def interp_energy(self, k_path, return_eigvecs=False):
        H_k_proj = self.get_proj_ham()
        H_k_interp = self.interp_op(H_k_proj, k_path)
        if return_eigvecs:
            u_k_interp = self.interp_op(self._u_wfs, k_path)
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
    def get_proj_ham(self):
        if not hasattr(self, "H_k_proj"):
            self.set_Bloch_ham()
        H_k_proj = self.u_wfs.conj() @ self.H_k @ np.swapaxes(self.u_wfs, -1, -2)
        return H_k_proj
