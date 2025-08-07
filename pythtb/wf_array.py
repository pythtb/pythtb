from .utils import _is_int, _offdiag_approximation_warning_and_stop
from .tb_model import TBModel
from .mesh import Mesh
import numpy as np
import copy  # for deepcopying
from itertools import product

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
        # TODO: We would like to have a KMesh object associated with the WFArray
        # this way we can store information about the k-points corresponding to each
        # point in the WFArray, and also the k-points can be used to impose PBC automatically.
        # To do this, the user needs to specify the k-points when constructing the WFArray.
        # Some dimensions of the mesh may be adiabatic parameters, or paths in k-space. Somehow
        # this should be distinguished from the regular k-mesh.

        # check that model is of type TBModel
        if not isinstance(model, TBModel):
            raise TypeError("model must be of type TBModel")
         # store model
        self._model = model

        # check that mesh is of type Mesh
        if not isinstance(mesh, Mesh):
            raise TypeError("mesh must be of type Mesh")
        
        # store mesh
        self._mesh = mesh
        
        # derive mesh dimensions from the Mesh object
        # mesh.grid has shape (*dims, coord_dim)
        self._mesh_size = np.array(self._mesh.grid.shape[:-1], dtype=int)
        self._dim_mesh = self._mesh_size.size

        # ensure each mesh dimension is at least 2
        # all dimensions should be 2 or larger, because pbc can be used
        if True in (self._mesh_size <= 1).tolist():
            raise ValueError(
                "Dimension of WFArray object in each direction must be 2 or larger."
            )

        # number of electronic states for each k-point
        if nstates is None:
            self._nstates = model.nstate  # this = norb*nspin = no. of bands
            # note: 'None' means to use the default, which is all bands!
        else:
            if not _is_int(nstates):
                raise TypeError("Argument nstates is not an integer.")
            self._nstates = nstates  # set by optional argument

        # number of spin components
        self._nspin = model.nspin
        # number of orbitals
        self._norb = model.norb
        # store orbitals from the model
        self._orb = model.orb_vecs

        self._pbc_axes = []  # axes along which periodic boundary conditions are imposed
        self._loop_axes = []  # axes along which loops are imposed
        # generate temporary array used later to generate object ._wfs
        wfs_dim = np.copy(self._mesh_size)
        wfs_dim = np.append(wfs_dim, self._nstates)
        wfs_dim = np.append(wfs_dim, self._norb)
        if self._nspin == 2:
            wfs_dim = np.append(wfs_dim, self._nspin)

        # store wavefunctions in the form [kx_index, ky_index,..., state, orb, spin]
        self._wfs = np.zeros(wfs_dim, dtype=complex)
        self._energies = np.zeros(tuple(self._mesh_size) + (self._nstates,), dtype=float)

    def __getitem__(self, key):
        self._check_key(key)
        return self._wfs[key]

    def __setitem__(self, key, value):
        self._check_key(key)
        self._wfs[key] = np.array(value, dtype=complex)

    def _check_key(self, key):
        # key is an index list specifying the grid point of interest
        if self._dim_mesh == 1:
            if isinstance(key, (tuple, list, np.ndarray)):
                assert len(key) == 1, "Key should be an integer or a tuple of length 1!"
                key = key[0]  # convert to integer
            elif not isinstance(key, (int, np.integer)):
                raise TypeError("Key should be an integer!")
            if key < (-1) * self._mesh_size[0] or key >= self._mesh_size[0]:
                raise IndexError("Key outside the range!")
        else:
            if len(key) != self._dim_mesh:
                raise TypeError("Wrong dimensionality of key!")
            for i, k in enumerate(key):
                if not _is_int(k):
                    raise TypeError("Key should be set of integers!")
                if k < (-1) * self._mesh_size[i] or k >= self._mesh_size[i]:
                    raise IndexError("Key outside the range!")

    @property
    def wfs(self):
        """The wavefunctions stored in the *WFArray* object."""
        return self._wfs

    @property
    def shape(self):
        """The shape of the wavefunction array."""
        return self._wfs.shape

    @property
    def mesh_size(self):
        """The mesh dimensions of the *WFArray* object."""
        return self._mesh_size

    @property
    def dim_mesh(self):
        """The number of dimensions of the *WFArray* object."""
        return self._dim_mesh

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

    @property
    def nspin(self):
        """The number of spin components stored in the *WFArray* object."""
        return self._nspin

    @property
    def norb(self):
        """The number of orbitals stored in the *WFArray* object."""
        return self._norb

    @property
    def model(self):
        """The underlying TBModel object associated with the *WFArray*."""
        return self._model

    @property
    def param_path(self):
        """The parameter path (e.g., k-points) along which the model was solved.
        This is only set if the model was solved along a path using `solve_on_path`."""
        return getattr(self, "_param_path", None)

    @property
    def flat_k_mesh(self):
        r"""Returns a flattened version of the k-mesh used in the *WFArray*."""
        return getattr(self, "_k_mesh_flat", None)

    @property
    def k_mesh(self):
        r"""Returns the KMesh object associated with the *WFArray*."""
        return getattr(self, "_k_mesh_square", None)

    @property
    def energies(self):
        """Returns the energies of the states stored in the *WFArray*."""
        return getattr(self, "_energies", None)

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
        """Returns Bloch and cell-periodic states from the WFArray.

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
        if self.dim_mesh != self.model.dim_k:
            raise ValueError(
                f"WFArray is initialized for a mesh of dimensions {self.dim_mesh}, "
                f"but the model has dim_k = {self.model.dim_k}. Bloch states assumes the"
                f"WFArray is defined on a regular k-mesh."
            )

        psi_wfs = self._apply_phase(wfs=u_wfs, inverse=False)

        # flatten last two axes together to condense spin and orbital indices
        if flatten_spin:
            u_wfs = u_wfs.reshape((*u_wfs.shape[:-2], -1))
            psi_wfs = psi_wfs.reshape((*psi_wfs.shape[:-2], -1))

        return_states = {
            "cell": u_wfs,
            "bloch": psi_wfs,
        }
        return return_states

    def get_projectors(self, return_Q=False):
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

        # band projectors
        P = np.einsum("...ni, ...nj -> ...ij", u_wfs, u_wfs.conj())
        Q = np.eye(self.nstates) - P

        if return_Q:
            return P, Q
        return P
    
    def solve_k_mesh(self, lambda_idx=None):
        """Solve the Hamiltonian on the k-mesh for a given parameter slice."""
        dim_k = self._mesh.dim_k
        shape_k = self._mesh.shape_k or ()
        shape_param = self._mesh.shape_param or ()
        Nk = int(np.prod(shape_k)) if shape_k else 1
        Np = int(np.prod(shape_param)) if shape_param else 1

        # Parameter index check
        if self._mesh.dim_param > 0:
            if lambda_idx is None:
                raise ValueError("lambda_idx must be provided when mesh has parameter dimensions")
            if not (0 <= lambda_idx < Np):
                raise IndexError(f"lambda_idx {lambda_idx} out of range [0, {Np})")
            
            k_pts = self._mesh.grid[..., lambda_idx, :dim_k] if self._mesh.dim_param > 0 else self._mesh.grid
            # flatten
            k_pts = k_pts.reshape(-1, dim_k)
        else:
            # ignore lambda_idx if no parameter dimensions
            lambda_idx = None

            k_pts = self._mesh.flat

        # Solve Hamiltonian
        evals, evecs = self._model.solve_ham(k_pts, return_eigvecs=True)

        evals_shape = tuple(shape_k) + (self.model.nstate,)
        if self.model.nspin > 1:
            evecs_shape = tuple(shape_k) + (self.model.nstate, self.model.norb, self.model.nspin)
        else:
            evecs_shape = tuple(shape_k) + (self.model.nstate, self.model.nstate)

        evecs = evecs.reshape(evecs_shape)
        evals = evals.reshape(evals_shape)

        # Now set the WFArray at the lambda_idx
        if lambda_idx is not None:
            slice_wfs = tuple([slice(None)]*len(shape_k)) + (lambda_idx,)
        else:
            slice_wfs = tuple([slice(None)]*len(shape_k))

        self._wfs[slice_wfs] = evecs
        self._energies[slice_wfs] = evals


    # TODO: Figure out how to solve over lambda as well
    # May want to pass model constructor to generate Hamiltonians along lambda 
    # dimension
    def solve_k_mesh(self, lambda_idx=None, auto_detect_pbc=True):
        """Solve the Hamiltonian on the k-mesh for a given parameter slice.

        .. versionadded:: 2.0.0

        Parameters
        ----------
        lambda_idx : int, optional
            The index of the parameter slice to solve for. If None, solves for all slices.

        auto_detect_pbc : bool, optional
            If True, automatically detects and imposes periodic boundary conditions (PBC) for the k-mesh.

        """
        dim_k = self._mesh.dim_k
        shape_k = self._mesh.shape_k or ()
        shape_param = self._mesh.shape_param or ()
        Nk = int(np.prod(shape_k)) if shape_k else 1
        Np = int(np.prod(shape_param)) if shape_param else 1

        # Parameter index check
        if self._mesh.dim_param > 0:
            if lambda_idx is None:
                raise ValueError("lambda_idx must be provided when mesh has parameter dimensions")
            if not (0 <= lambda_idx < Np):
                raise IndexError(f"lambda_idx {lambda_idx} out of range [0, {Np})")
            
            k_pts = self._mesh.grid[..., lambda_idx, :dim_k] if self._mesh.dim_param > 0 else self._mesh.grid
            # flatten
            k_pts = k_pts.reshape(-1, dim_k)
        else:
            # ignore lambda_idx if no parameter dimensions
            lambda_idx = None

            k_pts = self._mesh.flat

        # Solve Hamiltonian
        evals, evecs = self._model.solve_ham(k_pts, return_eigvecs=True)

        evals_shape = tuple(shape_k) + (self.model.nstate,)
        if self.model.nspin > 1:
            evecs_shape = tuple(shape_k) + (self.model.nstate, self.model.norb, self.model.nspin)
        else:
            evecs_shape = tuple(shape_k) + (self.model.nstate, self.model.nstate)

        evecs = evecs.reshape(evecs_shape)
        evals = evals.reshape(evals_shape)

        # Now set the WFArray at the lambda_idx
        if lambda_idx is not None:
            slice_wfs = tuple([slice(None)]*len(shape_k)) + (lambda_idx,)
        else:
            slice_wfs = tuple([slice(None)]*len(shape_k))

        self._wfs[slice_wfs] = evecs
        self._energies[slice_wfs] = evals

        # auto-detect and impose PBC for each k-component axis
        if auto_detect_pbc:
            grid = self._mesh.grid
            shape = grid.shape[:-1]  # mesh shape
            dim_k = self._mesh.dim_k
            for mesh_dir, k_comp in enumerate(self._mesh.k_axes):
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
                    print(f"Auto-imposing PBC in mesh direction {mesh_dir} for k-component {k_comp}")
                    self.impose_pbc(mesh_dir, k_comp)


    def solve_on_path(self, k_arr):
        """
        Solve the model along a 1D parameter path (e.g., k-points).
        Stores eigenvectors and eigenvalues along this path.

        Parameters
        ----------
        k_arr : array-like, shape (n_points, dim_k)
            Sequence of points (e.g., k-points) at which to solve the model.
            Must match the model's dim_k and the WFArray's mesh_size[0].

        Returns
        -------
        energies : ndarray, shape (n_points, nstates)
            Eigenvalues at each point along the path.
        wfs : ndarray, shape (n_points, nstates, norb[, nspin])
            Corresponding eigenvectors stored in the WFArray.
        """
        k_arr = np.asarray(k_arr, dtype=float)
        # check dimensionality
        if k_arr.ndim != 2 or k_arr.shape[1] != self.model.dim_k:
            raise ValueError(
                f"Expected k_arr to have shape (n_points, {self.model.dim_k}), "
                f"but got shape {k_arr.shape}."
            )
        n_points = k_arr.shape[0]
        # check that the number of points matches the mesh size
        if self.dim_mesh != 1 or self.mesh_size[0] != n_points:
            raise ValueError(
                f"WFArray is initialized for a mesh size of {self.mesh_size[0]}, "
                f"but k_arr has {n_points} points."
            )
        self._param_path = k_arr.copy()

        eigvals, eigvecs = self.model.solve_ham(k_arr, return_eigvecs=True)
        for idx, pt in enumerate(k_arr):
            self._energies[idx] = eigvals[idx]
            self._wfs[(idx,)] = eigvecs[idx]


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
            start_k = [0] * self.dim_mesh
            start_k = np.asarray(start_k, dtype=float)
        else:
            start_k = np.asarray(start_k, dtype=float)
            # check dimensionality
            if start_k.ndim != 1 or start_k.shape[0] != self.dim_mesh:
                raise ValueError(
                    f"Expected start_k to have shape ({self.dim_mesh},), "
                    f"but got shape {start_k.shape}."
                )
            
        # check values
        if np.any(start_k < -0.5) or np.any(start_k > 0.5):
            raise ValueError(
                f"Expected start_k to be in the range [-0.5, 0.5], "
                f"but got {start_k}."
            )
        
        # check dimensionality
        if self.dim_mesh != self.model.dim_k:
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
            nk - 1 for nk in self.mesh_size
        )  # number of k-points in each direction

        # we use a mesh size of (nk-1) because the last point in each direction will be
        # the same as the first one, so we only need (nk-1) points
        mesh_size = tuple(nk - 1 for nk in self.mesh_size)
        k_axes = [
            np.linspace(start_k[idx], start_k[idx] + 1, nk, endpoint=False)
            for idx, nk in enumerate(mesh_size)
        ]
        # stack into a grid of shape (nk1-1, nk2-1, ..., nkd-1, dim_k)
        k_pts_sq = np.stack(np.meshgrid(*k_axes, indexing="ij"), axis=-1)
        # flatten the grid
        k_pts = k_pts_sq.reshape(-1, self.dim_mesh)

        # store for later
        self._k_mesh_square = k_pts_sq
        self._k_mesh_flat = k_pts

        # solve the model on the k-mesh
        evals, evecs = self._model.solve_ham(k_pts, return_eigvecs=True)

        # reshape to back into a full (nk1, nk2, ..., nkd, nstate) mesh
        full_shape = tuple(mesh_size) + (self.nstates,)
        evals = evals.reshape(full_shape)
        evecs = evecs.reshape(
            full_shape + (self.norb,) + ((self.nspin,) if self.nspin > 1 else ())
        )

        self._energies = evals  # store energies in the WFArray

        # reshape to square mesh: (nk1, nk2, ..., nkd-1, nstate, nstate) for evecs

        # Store all wavefunctions in the WFArray
        idx_arr = np.ndindex(*mesh_size)
        for idx in idx_arr:
            self[idx] = evecs[idx]

        # impose periodic boundary conditions along all directions
        for dir in range(self.dim_mesh):
            # impose periodic boundary conditions
            self.impose_pbc(dir, self.model.per[dir])

        if self.nstates > 1:
            gaps = evals[..., 1:] - evals[..., :-1]
            return gaps.min(axis=tuple(range(self.dim_mesh)))
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
            wf_shape[self._dim_mesh] = nstates
            wf_new._wfs = np.empty_like(wf_new._wfs, shape=wf_shape)

        return wf_new

    def _apply_phase(self, inverse=False):
        """
        Change between cell periodic and Bloch wfs by multiplying exp(\pm i k . tau)

        Assumes that the WFArray was populated using a regular mesh
        of k-points and none of the states are at the same k-point. This means
        there should be no adiabatic lambda points in the mesh.

        Returns
        -------
        wfsxphase : np.ndarray
            wfs with orbitals multiplied by phase factor
        """
        lam = -1 if inverse else 1  # overall minus if getting cell periodic from Bloch

        # create a regular mesh of k-points
        # NOTE: Assumes that the WFArray was populated using a regular mesh
        # of k-points

        nks = self.mesh_size
        dim_k = len(nks)
        if dim_k != self.model.dim_k:
            raise ValueError(
                f"WFArray has {dim_k} k-dimensions, but model has {self.model.dim_k}!"
            )
        # create a mesh of k-points in the range [0, 1) for each k-dimension
        end_pts = [0, 1]
        k_vals = [np.linspace(end_pts[0], end_pts[1], nk, endpoint=False) for nk in nks]
        flat_mesh = np.stack(np.meshgrid(*k_vals, indexing="ij"), axis=-1)
        flat_mesh = flat_mesh.reshape(-1, dim_k)
        # flat_mesh is now of shape [k_val, dim_k], where k_val is the total number of k-points

        per_dir = list(range(flat_mesh.shape[-1]))
        # slice second dimension to only keep only periodic dimensions in orb
        per_orb = self.model.orb_vecs[:, per_dir]

        # compute a list of phase factors: exp(pm i k . tau) of shape [k_val, orbital]
        phases = np.exp(lam * 1j * 2 * np.pi * per_orb @ flat_mesh.T, dtype=complex).T
        # reshape phases to have shape: [nk1, nk2, ..., nkd, norb]
        phases = phases.reshape(*nks, self.norb)

        # wfs is of shape [nk1, nk2, ..., nkd, nband, norb, nspin]
        wfs = self.wfs

        # broadcasting to match dimensions of wfs
        if self._nspin == 1:
            # newaxis along state dimension
            phases = phases[..., np.newaxis, :]
        elif self._nspin == 2:
            # newaxis along state and spin dimension
            phases = phases[..., np.newaxis, :, np.newaxis]

        return wfs * phases

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
        if mesh_dir < 0 or mesh_dir >= self.dim_mesh:
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
        if mesh_dir < 0 or mesh_dir >= self.dim_mesh:
            raise ValueError(
                f"mesh_dir must be between 0 and {self.dim_mesh-1}, got {mesh_dir}."
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
        #
        evec = self.wfs[tuple(k_idx)][occ]
        return self.model.position_matrix(evec, dir)

    def position_expectation(self, k_idx, occ, dir):
        """Position expectation value for a given k-point and set of states.

        These elements :math:`X_{n n}` can be interpreted as an
        average position of n-th Bloch state ``evec[n]`` along
        direction `dir`. 

        This routine can be used to compute the position expectation value for a
        given k-point and set of states (which can be all states, or
        a specific subset). 

        Parameters
        ----------
        k_idx: array-like of int
            Set of integers specifying the k-point of interest in the mesh.
        occ: array-like, 'all'
            List of states to be included (can be 'all' to include all states).
        dir: int
            Direction along which to compute the position expectation value.

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

        evec = self.wfs[tuple(k_idx)][occ]
        return self.model.position_expectation(evec, dir)

    def position_hwf(self, k_idx, occ, dir, hwf_evec=False, basis="wavefunction"):
        """Eigenvalues and eigenvectors of the position operator in a given basis.

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

        evec = self.wfs[tuple(k_idx)][occ]
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
            dirs = list(range(self.dim_mesh))

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
        N_loop, *rest_shape, nbands, norb = wf.shape
        # Flatten redundant param dimensions intermediately
        wf_flat = wf.reshape(
            N_loop, -1, nbands, norb
        )  # shape: (N_loop, rest_shape, nbands, norb)

        # Compute Berry phase for each slice along other dimensions
        results = []
        # loop over all other parameter values other than the loop dimension
        for idx in range(wf_flat.shape[1]):
            slice_wf = wf_flat[:, idx, :, :]
            results.append(self.berry_loop(slice_wf, evals=berry_evals))

        ret = np.array(results)

        if contin:
            # Make phases continuous
            ret = np.unwrap(ret, axis=0)

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
            `[dim_mesh, dim_mesh, *flux_shape, n_states, n_states]`. The
            shape will depend on the parameters passed to the function.

            If plane is `None` (default), then the first two axes
            `(dim_mesh, dim_mesh)` correspond to the plane directions, otherwise,
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
        dim_mesh = self.dim_mesh  # Total dimensionality of adiabatic space: d
        n_param = list(
            self.mesh_size
        )  # Number of points in adiabatic mesh: (nk1, nk2, ..., nkd)

        # Validate plane
        if plane is not None:
            if not isinstance(plane, (list, tuple, np.ndarray)):
                raise TypeError("plane must be None, a list, tuple, or numpy array.")
            if len(plane) != 2:
                raise ValueError("plane must contain exactly two directions.")
            if any(p < 0 or p >= dim_mesh for p in plane):
                raise ValueError(f"Plane indices must be between 0 and {dim_mesh-1}.")
            if plane[0] == plane[1]:
                raise ValueError("Plane indices must be different.")

        # Unique axes for periodic boundary conditions and loops
        # pbc_axes = list(set(self._pbc_axes + self._loop_axes))
        flux_shape = n_param
        for ax in range(dim_mesh):
            flux_shape[ax] -= 1  # Remove last link in each periodic direction

        # Initialize the Berry flux array
        if plane is None:
            shape = (
                (dim_mesh, dim_mesh, *flux_shape, n_states, n_states)
                if not abelian
                else (dim_mesh, dim_mesh, *flux_shape)
            )
            berry_flux = np.zeros(shape, dtype=complex)
            dirs = list(range(dim_mesh))
            plane_idxs = dim_mesh
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
                for ax in range(dim_mesh):
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

    def set_Bloch_ham(self, lambda_vals=None, model_fxn=None):
        if lambda_vals is None:
            H_k = self.model.hamiltonian(
                k_pts=self.k_mesh.flat_mesh
            )  # [Nk, norb, norb]
            # [nk1, nk2, ..., norb, norb]
            self.H_k = H_k.reshape(*[nk for nk in self.k_mesh.nks], *H_k.shape[1:])
            return

        lambda_keys = list(lambda_vals.keys())
        lambda_ranges = list(lambda_vals.values())
        lambda_shape = tuple(len(vals) for vals in lambda_ranges)
        dim_lambda = len(lambda_keys)

        n_kpts = self.k_mesh.Nk
        n_orb = self._n_orb
        n_spin = self._n_spin
        n_states = n_orb * n_spin

        # Initialize storage for wavefunctions and energies
        if n_spin == 1:
            H_kl = np.zeros((*lambda_shape, n_kpts, n_states, n_states), dtype=complex)
        elif n_spin == 2:
            H_kl = np.zeros(
                (*lambda_shape, n_kpts, n_orb, n_spin, n_orb, n_spin), dtype=complex
            )

        for idx, param_set in enumerate(np.ndindex(*lambda_shape)):
            # kwargs for model_fxn with specified parameter values
            param_dict = {
                lambda_keys[i]: lambda_ranges[i][param_set] for i in range(dim_lambda)
            }

            # Generate the model with modified parameters
            modified_model: TBModel = model_fxn(**param_dict)

            H_kl[param_set] = modified_model.hamiltonian(k_pts=self.k_mesh.flat_mesh)

        # Reshape for compatibility with existing Berry curvature methods

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
                + tuple(range(dim_lambda + 1, dim_lambda + 5))
            )
        H_kl = np.transpose(H_kl, axes=new_axes)

        if self._nspin == 1:
            new_shape = (*self.k_mesh.nks, *lambda_shape, n_states, n_states)
        else:
            new_shape = (*self.k_mesh.nks, *lambda_shape, n_states, n_orb, n_spin)
        H_kl = H_kl.reshape(new_shape)

        self.H_k = H_kl

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
