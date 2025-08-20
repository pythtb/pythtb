from .utils import _is_int, _offdiag_approximation_warning_and_stop
from .tb_model import TBModel
from .mesh import Mesh
import numpy as np
import copy  # for deepcopying
from itertools import product
from functools import partial
import warnings
import functools
import logging


logging.basicConfig(level=logging.INFO)
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

        # ensure each mesh dimension is at least 2
        # all dimensions should be 2 or larger, because pbc can be used
        if True in (np.array(self.mesh_shape, dtype=int) <= 1).tolist():
            raise ValueError(
                "Dimension of WFArray object in each direction must be 2 or larger.\n"
                "This is required for periodic boundary conditions (PBC) to be applied.\n"
                "Maybe you need to build the mesh first?"
            )

        # number of electronic states for each k-point
        if nstates is None:
            self._nstates = self.model.nstate  # this = norb*nspin = no. of bands
            # note: 'None' means to use the default, which is all bands!
        else:
            if not _is_int(nstates):
                raise TypeError("Argument nstates is not an integer.")
            self._nstates = nstates  # set by optional argument

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
        if not isinstance(value, (list, np.ndarray)):
            raise TypeError("Value must be a list or numpy array!")
        
        value = np.array(value, dtype=complex)

        if self.nspin == 2:
            if value.ndim == self.ndims + 2:
                if value.shape[-1] != self.norb*2:
                    raise ValueError("Value shape does not match expected shape for spin-1/2 model!")
                value = value.reshape(*value.shape[:-1], self.norb, 2)

        else:
            if value.shape != self.shape[len(self.mesh_shape):]:
                raise ValueError("Incompatible shape for wavefunction!")
        
        self._wfs[key] = value

    def _check_key(self, key):
        # Normalize to a tuple of ints
        if self.ndims == 1:
            if isinstance(key, (tuple, list, np.ndarray)):
                if len(key) != 1:
                    raise TypeError("Key should be an integer or a tuple of length 1!")
                key = key[0]
            if not _is_int(key):
                raise TypeError("Key should be an integer!")
            idxs = (int(key),)
        else:
            if not isinstance(key, (tuple, list, np.ndarray)):
               raise TypeError("Key should be a tuple, list, or ndarray!")
            if len(key) != self.ndims:
                raise TypeError("Wrong dimensionality of key!")
            if not all(_is_int(k) for k in key):
                raise TypeError("Key should be set of integers!")
            idxs = tuple(int(k) for k in key)

        for i, k in enumerate(idxs):
            lo, hi = -self.mesh_shape[i], self.mesh_shape[i]
            if k < lo or k >= hi:
                raise IndexError("Key outside the range!")

    @property
    def model(self):
        """The underlying TBModel object associated with the *WFArray*."""
        return self._model
    
    @property
    def mesh(self):
        """The mesh object associated with the *WFArray*."""
        return self._mesh

    @property
    def wfs(self):
        """The (cell-periodic) wavefunctions stored in the *WFArray* object."""
        return self._wfs

    @property
    def filled(self):
        """Whether the wavefunctions are filled (i.e., not empty)."""
        # if uninitialzed, wfs will be np.empty
        return self._wfs.size > 0
    
    @property
    def u_nk(self):
        """The cell-periodic wavefunctions stored in the *WFArray* object."""
        if not self.filled:
            raise ValueError("Wavefunctions are not initialized.")
        if self.dim_k == 0:
            raise ValueError("Cell-periodic wavefunctions are not defined for 0D k-space.")

        return getattr(self, "_u_nk", None)

    @property
    def psi_nk(self):
        """The Bloch wavefunctions stored in the *WFArray* object."""
        if not self.filled:
            raise ValueError("Wavefunctions are not initialized.")
        if self.dim_k == 0:
            raise ValueError("Bloch wavefunctions are not defined for 0D k-space.")
        
        return getattr(self, "_psi_nk", None)
    
    @property
    def Mmn(self):
        """The overlap matrix of the wavefunctions stored in the *WFArray* object."""
        if not self.filled:
            raise ValueError("Wavefunctions are not initialized.")
        if not self.mesh.is_grid:
            raise ValueError("Overlap matrix is only defined for regular grids.")
        if self.dim_k == 0:
            raise ValueError("Overlap matrix is not defined for 0D k-space.")
        
        return self._Mmn

    @property
    def energies(self):
        """Returns the energies of the states stored in the *WFArray*."""
        if not self.filled:
            raise ValueError("Wavefunctions are not initialized.")
        if self.hamiltonian is None:
            raise ValueError("Hamiltonian is not set. Use set_ham() to set it.")
        if self._energies.size == 0:
            raise ValueError("Energies are not initialized.")
        
        return self._energies

    @property
    def hamiltonian(self):
        r"""Returns the Hamiltonian matrix in (k,\lambda)-space."""
        return getattr(self, "_H_k", None)
    
    @property
    def mesh_shape(self):
        """The mesh dimensions of the *WFArray* object."""
        return self.mesh.shape_mesh

    @property
    def shape(self):
        """The shape of the wavefunction array."""
        wfs_dim = np.array(self.mesh_shape, dtype=int)
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
        return self.mesh.dim_lambda

    @property
    def ndims(self):
        """The number of dimensions of the *WFArray* object."""
        return self.mesh.num_axes

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
        return self.model.nspin

    @property
    def norb(self):
        """The number of orbitals stored in the *WFArray* object."""
        return self.model.norb
    
    @property
    def nks(self):
        """The number of k-points stored in the *WFArray* mesh."""
        return self.mesh.shape_k
    
    @property
    def nlams(self):
        """The number of lambda points stored in the *WFArray* mesh."""
        return self.mesh.shape_lambda

    @property
    def k_points(self):
        """The k-space mesh associated with the *WFArray*."""
        return self.mesh.get_k_points()
    
    @property
    def param_points(self):
        """The parameter mesh associated with the *WFArray*."""
        return self.mesh.get_param_points()
    

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
        r"""Pick a subset of states to keep in the `WFArray`.

        This method modifies the existing `WFArray` in place to keep only the specified states.

        Parameters
        ----------
        state_idxs : array-like of int 
            State indices to keep.

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
    

    @deprecated(
        "The 'empty_like' method is deprecated and will be removed in a future release. " \
        "Create a new `WFArray` object with the same `Mesh` and `TBModel` instead."
    )
    def empty_like(self, nstates=None):
        r"""
        .. deprecated:: 2.0.0
            `empty_like` has been deprecated and will be removed in a future release. " \
            "Create a new `WFArray` object with the same `Mesh` and `TBModel` instead."
        """
        # make a full copy of the WFArray
        wf_new = WFArray(self.model, self.mesh, nstates=nstates)
        return wf_new
    

    def get_k_shell(self, n_shell: int, report: bool = False):
        """Generates shells of k-points around the Gamma point.

        Returns array of vectors connecting the origin to nearest 
        neighboring k-points in the mesh. The function

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
            connecting nearest neighbor k-mesh points. Length is `n_shell`.
        idx_shell : list[np.ndarray[int]]
            Each entry is an array of integer shifts that takes a k-point 
            index in the mesh to its nearest neighbors.
            Length is `n_shell`.
        """
        if not isinstance(n_shell, int) or n_shell < 1:  
            raise ValueError("Invalid n_shell: must be a positive integer.")
        
        recip_lat_vecs = self.model.get_recip_lat()
        dim_k = self.dim_k
        nks = self.nks

        if dim_k != len(nks):
            raise ValueError("Mesh is not full, cannot generate k-shells.")

        # basis vectors connecting neighboring mesh points (in inverse Cartesian units)
        dk = np.array([recip_lat_vecs[i] / nk for i, nk in enumerate(nks)])

        # array of integers e.g. in 2D for n_shell = 1 would be 
        # [-1,-1], [-1,0], [-1,1], [0,-1], [0,1], [1,-1], [1,0], [1,1]
        nnbr_idx = list(product(range(-n_shell, n_shell + 1), repeat=dim_k))
        nnbr_idx.remove((0,) * dim_k)
        nnbr_idx = np.array(nnbr_idx)
        
        # Vectors connecting k-points near Gamma point (in inverse Cartesian units)
        # (M, dim_k) @ (dim_k, dim_k) -> (M, dim_k)
        b_vecs = nnbr_idx @ dk 

        # Squared norms
        d2 = np.einsum('ij,ij->i', b_vecs, b_vecs)
        # remove numerical noise
        d2r = np.round(d2, 12)

        # Sort by increasing radius^2
        sorted_idxs = np.argsort(d2r)
        d2r_sorted = d2r[sorted_idxs]
        b_sorted = b_vecs[sorted_idxs]
        idx_sorted = nnbr_idx[sorted_idxs]

        # Unique radii^2 in order; take first n_shell shells
        unique_d2 = sorted(list(set(d2r_sorted)))  # removes repeated distances
        unique_d2 = unique_d2[:n_shell]  # keep only distances up to n_shell away

        # keep only b_vecs in n_shell shells
        k_shell = [
            b_sorted[np.isin(d2r_sorted, unique_d2[i])]
            for i in range(len(unique_d2))
        ]
        idx_shell = [
            idx_sorted[np.isin(d2r_sorted, unique_d2[i])]
            for i in range(len(unique_d2))
        ]

        if report:
            # Pretty report
            lines = []
            lines.append("k-shell report")
            lines.append("═" * 46)
            lines.append(f"dim_k: {dim_k}   nks: {nks}")
            # Compact step info: show |dk_i| and vectors
            step_norms = [np.linalg.norm(dk[i]) for i in range(dim_k)]
            steps_str  = ", ".join(f"|dk_{i}|={step_norms[i]:.6g}" for i in range(dim_k))
            lines.append(f"step sizes: {steps_str}")
            # Optional: show dk rows
            lines.append("dk vectors:")
            for i in range(dim_k):
                lines.append(f"  dk[{i}] = {np.array2string(dk[i], precision=6, floatmode='maxprec_equal', suppress_small=True)}")

            lines.append("")
            lines.append("Shells (by increasing |b|):")
            for si, (B, I) in enumerate(zip(k_shell, idx_shell), start=1):
                deg = B.shape[0]
                radius = np.sqrt(unique_d2[si-1])
                lines.append(f"  • shell {si:>2}: |b|={radius:.6g}   degeneracy={deg}")
                # Show a few representatives from this shell
                head = min(deg, 6)
                for j in range(head):
                    b_str = np.array2string(B[j], precision=6, floatmode='maxprec_equal', suppress_small=True)
                    i_str = np.array2string(I[j], formatter={'int':lambda x: f"{x:>2}"})
                    lines.append(f"      idx={i_str}   b={b_str}")
                if deg > head:
                    lines.append(f"      … (+{deg-head} more)")
            print("\n".join(lines))

        return k_shell, idx_shell

    def get_shell_weights(self, n_shell : int = 1, report: bool = False):
        r"""Generates the finite difference weights on a k-shell.

        This function uses the k-shells generated by :func:`get_k_shell` 
        to compute the  weights for finite difference approximations of 
        :math:`\nabla_{\mathbf{k}}` on a Monkhorst-Pack k-mesh. To linear
        order, the following expression must be satisfied

        .. math::

            \sum_{s}^{N_{\rm sh}} w_s \sum_{i}^{M_s} b_{\alpha}^{i,s}
            b_{\beta}^{i,s} = \delta_{\alpha,\beta}

        where :math:`N_{\rm sh} \equiv` ``n_shell` is the number of shells
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
        k_shell : list[np.ndarray[float]]
            List of :math:`\mathbf{b}` vectors in inverse units of lattice vectors
            connecting nearest neighbor k-mesh points. Length is `n_shell`.
        idx_shell : list[np.ndarray[int]]
            Each entry is an array of integer shifts that takes a k-point 
            index in the mesh to its nearest neighbors.
            Length is `n_shell`.
        """
        from itertools import combinations_with_replacement as comb

        k_shell, idx_shell = self.get_k_shell(n_shell=n_shell, report=report)
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
        if report:
            print(f"Finite difference weights: {w}")
        return w, k_shell, idx_shell
    
      
    def set_ham(self, model_func=None, fixed_params: dict={}):
        r"""Sets the Hamiltonian for the wavefunction array.

        .. versionadded:: 2.0.0

        If a model function is provided, it is used to generate the Hamiltonian
        with the current mesh parameters. If no parameters are defined in the
        mesh, a warning is issued and the `TBModel` passed upon instantiation
        is used to generate the Hamiltonian.

        Parameters
        ----------
        model_func : callable, optional
            A function that takes mesh parameters as input and returns a
            :class:`TBModel`.
        fixed_params : dict, optional
            A dictionary of fixed parameters to be passed to the model function.
            It has the structure ``{<param_name>: <param_value>, ...}``. 
            ``param_name`` must match the names of the parameters expected by 
            the model function.

        Notes
        ------
        If there are parameter dimensions in the Mesh, then the ``model_func`` must be used.
        The ``axis_names`` argument in the ``Mesh`` constructor are used to assign axes to the
        variables to be passed to the model function. The names must match the parameter names
        expected by the model function.

        If `axis_names` were not specified in the Mesh initialization, they will be
        set to the default values of 'k_i' and 'l_i', where i is the
        index of the axis.

        Examples
        --------
        Say we have a model function defined as follows:

        >>> def model_func(param1, param2):
        ...     lat = [[1, 0], [0, 1]]
        ...     orb = [[0,0], [0.5, 0.5]]
        ...     model = TBModel(dim_k=2, dim_r=2, lat=lat, orb=orb, nspin=1)
        ...     # Some model calculations with parameters
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
        is so that we have the lattice structure stored in the ``WFArray``. We pass the reference model and mesh
        to the ``WFArray`` constructor.

        >>> ref_model = model_func(0, 0)
        >>> wfa = WFArray(ref_model, mesh)

        Now say we want to keep ``param_1`` fixed to be 1. We can do this by setting the `fixed_params` argument 
        when calling `set_ham`.

        >>> wfa.set_ham(model_func=model_func, fixed_params={'param1': 1})

        The Hamiltonian now has the correct form with respect to the fixed parameters. The model is 
        spinless and has 2 orbitals, so the shape of the Hamiltonian is:

        >>> wfa.hamiltonian.shape
        (20, 20, 5, 2, 2)

        """

        if model_func is not None and self.mesh.num_lambda_axes == self.dim_lambda == 0:
            logger.warning("Model function is provided but no parameters are defined in Mesh.")
        if model_func is None and self.mesh.num_lambda_axes > 0:
            raise ValueError("Model function is required when mesh has parameters.")

        flat = self.mesh.flat
        n_orb = self.norb
        n_spin = self.nspin
        n_states = self.nstates

        if model_func is None:
            Hk = self.model.hamiltonian(k_pts=flat)

            if self.mesh.is_grid:
                shape_k = self.mesh.shape_k
                Hk = Hk.reshape(*shape_k, *Hk.shape[1:])
            
            self._H_k = Hk
            return

        model_gen = partial(model_func, **fixed_params)
        self.fixed_params = fixed_params
        self.model_func = model_func
        
        dim_lambda = self.mesh.dim_lambda
        lambda_shape = self.nlams
        lambda_keys = self.mesh.axis_names[self.mesh.num_k_axes:]
        lambda_ranges = list(self.mesh.get_param_ranges().values())

        if self.dim_k > 0:
            n_kpts = np.prod(self.nks)
            shape_k = self.mesh.shape_k
            k_grid = self.k_points
            k_flat = k_grid.reshape(n_kpts, -1)

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
            perm = (
                list(range(dim_lambda, dim_lambda + n_k_axes)) 
                + list(range(0, dim_lambda)) 
                + list(range(dim_lambda + n_k_axes, total_axes))
            )
            H_kl = np.transpose(H_kl, axes=perm)

        else:
            if n_spin == 1:
                H_kl = H_kl.reshape(*lambda_shape, n_orb, n_orb)
            elif n_spin == 2:
                H_kl = H_kl.reshape(*lambda_shape, n_orb, n_spin, n_orb, n_spin)

        self._H_k = H_kl

    def get_states(
            self, 
            state_idx=None,
            flatten_spin=False, 
            return_psi=False):
        r"""Return cell-periodic (and optionally Bloch) states.

        .. versionadded:: 2.0.0

        Shapes:
            cell-periodic u : [nk..., nl..., nstate, norb(,nspin)]
            bloch psi       : same as u
            If *flatten_spin* and nspin==2, the last two axes are flattened.

        Parameters
        ----------
        state_idx : int, array-like, optional
            Index or indices of the states to return. If None, all states are returned.
        flatten_spin : bool, optional
            If True, the spin and orbital indices are flattened into a single index.
            Default is False.
        return_psi : bool, optional
            If True, the function also returns the Bloch wavefunctions.

        Returns
        -------
        u_nk : np.ndarray
            Cell-periodic states (periodic in real space) :math:`u_{n\mathbf{k}}(\mathbf{r})`
        psi_nk : np.ndarray, optional
            Bloch states (periodic in k-space) :math:`\psi_{n\mathbf{k}}(\mathbf{r})`

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

        if flatten_spin and self.nspin == 2:
            u = u.reshape((*u.shape[:-2], -1))
            if psi is not None:
                psi = psi.reshape((*psi.shape[:-2], -1))

        return (u, psi) if return_psi else u

        
    def _set_wfs(
        self, 
        wfs, 
        cell_periodic: bool = True, 
        spin_flattened=False, 
        set_projectors=True
    ):
        """Sets the wavefunctions in the *WFArray* object.

        This function sets the Bloch and cell-periodic eigenstates as class attributes
        when `wfs` is defined on the a k-mesh. When the model is finite, only the
        ``.wfs`` attribute is set.

        Parameters
        ----------
        wfs : np.ndarray
            Wavefunctions to populate the mesh with. The shape must match the expected
            shape for the given mesh and spin configuration.
        cell_periodic : bool, optional
            If True, the wavefunctions are treated as cell-periodic (Bloch states).
            Default is True.
        spin_flattened : bool, optional
            If True, the spin and orbital indices are flattened into a single index.
            Default is False. This must match the shape of the input ``wfs``.
        """
        if not isinstance(wfs, np.ndarray):
            raise TypeError("wfs must be a numpy ndarray.")
        # Check the shape of wfs
        if spin_flattened and self.nspin == 2:
            if wfs.shape != self.mesh_shape + (self.nstates, self.norb * self.nspin):
                raise ValueError(
                    f"wfs shape {wfs.shape} does not match expected shape for flattened spin: "
                    f"{self.mesh_shape + (self.nstates, self.norb * self.nspin)}"
                )
            self._nstates = wfs.shape[-2]
            
        if not spin_flattened and self.nspin == 2:
            if wfs.shape != self.mesh_shape + (self.nstates, self.norb, self.nspin):
                raise ValueError(
                    f"wfs shape {wfs.shape} does not match expected shape for non-flattened spin: "
                    f"{self.mesh_shape + (self.nstates, self.norb, self.nspin)}"
                )
            self._nstates = wfs.shape[-3]
    
        elif self.nspin == 1:
            if wfs.shape != self.mesh_shape + (self.nstates, self.norb):
                raise ValueError(
                    f"wfs shape {wfs.shape} does not match expected shape for spinless model: "
                    f"{self.mesh_shape + (self.nstates, self.norb)}"
                )
            self._nstates = wfs.shape[-2]
            

        wfs = wfs.reshape(self.shape)

        if self.dim_k > 0:
            if cell_periodic:
                phases = self._get_phases(inverse=False)
                psi_nk = wfs * phases
                self._u_wfs = self._wfs = wfs
                self._psi_nk = psi_nk
            else:
                phases = self._get_phases(inverse=True)
                u_nk = wfs * phases
                self._u_nk = self._wfs = u_nk
                self._psi_nk = wfs

            if self.mesh.is_grid:
                self._Mmn = self.get_overlap_mat()

        elif not cell_periodic:
            raise ValueError("Cannot set non-cell-periodic wavefunctions for 0D k-space.")
        
        else:
            self._wfs = wfs
    
        if set_projectors:
            self._set_projectors()


    def _set_projectors(self):
        P, Q = self.get_projectors(return_Q=True)
        self._P = P
        self._Q = Q

        if self.dim_k > 0 and self.mesh.is_grid:
            _, nnbr_idx_shell = self.get_k_shell(n_shell=1, report=False)
            num_nnbrs = nnbr_idx_shell[0].shape[0]

            self._P_nbr = np.zeros(
                (P.shape[:-2] + (num_nnbrs,) + P.shape[-2:]), dtype=complex
            )
            self._Q_nbr = np.zeros_like(self._P_nbr)

            for idx, idx_vec in enumerate(nnbr_idx_shell[0]):  # nearest neighbors
                u_shifted = self.roll_states_with_bc(idx_vec, flatten_spin=True)
                P = np.einsum("...ni, ...nj -> ...ij", u_shifted, u_shifted.conj())
                Q = np.eye(u_shifted.shape[-1]) - P
                self._P_nbr[..., idx, :, :] = P
                self._Q_nbr[..., idx, :, :] = Q


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

        u_nk = self.get_states(flatten_spin=True)

        if state_idx is not None:
            u_nk = u_nk[..., state_idx, :]

        # band projectors
        P = np.einsum("...ni, ...nj -> ...ij", u_nk, u_nk.conj())
        Q = np.eye(u_nk.shape[-1]) - P

        if return_Q:
            return P, Q
        return P

    def solve_mesh(
            self, 
            model_func=None, 
            fixed_params: dict=None, 
            use_metal=False
            ):
        r"""Diagonalizes the Hamiltonian over the `Mesh` points.

        .. versionadded:: 2.0.0

        Solves for the eigenstates and eigenenergies of the Hamiltonian defined 
        by the `TBModel` on the points set in `Mesh`.

        If the `Mesh` has parametric dimensions, a `model_func` must be provided that returns
        the modified model. The varying arguments of the function must match the :math:`\lambda`
        axis names defined in the `Mesh`. Some of the arguments in the `model_func` may be kept fixed
        by specifying their names as keys and values as the values in the `fixed_params` dictionary.

        .. note::

           When the `Mesh` grid includes endpoints in k-space, one must note that when using
           the :func:`berry_flux` function the last plaquette will be omitted. See Notes for 
           further details

        Parameters
        ----------
        model_func : function, optional
            A function that returns a model given a set of parameters.
        param_vals (dict, optional):
            Dictionary of parameter values for adiabatic evolution. Each key corresponds to
            a varying parameter and the values are arrays

        See Also
        --------
        `set_ham`

        Notes
        ------
        The eigenfunctions :math:`\psi_{n {\bf k}}` are by convention
        chosen to obey a periodic gauge, i.e.,
        :math:`\psi_{n,{\bf k+G}}=\psi_{n {\bf k}}` not only up to a
        phase, but they are also equal in phase. It follows that
        the cell-periodic Bloch functions are related by
        :math:`u_{n,{\bf k_0+G}}=e^{-i{\bf G}\cdot{\bf r}} u_{n {\bf k_0}}`.
        See :download:`notes on tight-binding formalism </misc/pythtb-formalism.pdf>` 
        section 4.4 and equation 4.18 for more detail.

        This routine automatically finds the directions in the `Mesh` that include endpoints of
        the Brillouin zone, meaning the value of one of the components of the k-vector (`k_dir`) 
        differ at the beginning and end by a reciprocal lattice vector along that axis 
        (1 in reduced units).

        If these points are included explicitly, periodic boundary conditions are automatically 
        imposed. This sets the cell-periodic Bloch function at the end of the mesh in this direction
        equal to the first, multiplied by a phase factor.
        Explicitly, this means we set :math:`u_{n,{\bf k_0+G}}=e^{-i{\bf G}\cdot{\bf r}} u_{n {\bf k_0}}` 
        for the corresponding reciprocal lattice vector :math:`\mathbf{G} = \mathbf{b}_{\texttt{k_dir}}`,
        where :math:`\mathbf{b}_{\texttt{k_dir}}` is the reciprocal lattice basis vector corresponding to the
        direction `k_dir`. The state :math:`u_{n{\bf k_0}}` is the state populated in the first element
        of the mesh along the `mesh_dir` axis.
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
            logger.info("Attempting to use tensorflow for eigenvalue calculation")

            from tensorflow import convert_to_tensor
            from tensorflow import complex64 as tfcomplex64
            from tensorflow.linalg import eigh as tfeigh

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

        self._set_wfs(evec, cell_periodic=True, spin_flattened=False)
        self._energies = eval

        if self.nstates > 1:
            gaps = eval[..., 1:] - eval[..., :-1]
            self.gaps = gaps.min(axis=tuple(range(self.ndims)))
        else:
            self.gaps = None

        # These contain endpoints (k_i = 1 in reduced units)
        # This means we need to impose periodic boundary conditions explicitly
        closed_axes = self.mesh.closed_axes
        for ax, comp in closed_axes:
            if ax in self.mesh.k_axes:
                # impose periodic boundary conditions for k-components
                logger.info(f"Imposing PBC in mesh direction {ax} for k-component {comp}")
                self.impose_pbc(ax, comp)


    @deprecated(
            "This function is deprecated and will be removed in a future version."
    )
    def solve_on_grid(self, start_k=None):
        r"""
        """
        # deprecation warning
        return
    
    @deprecated(
            "This function is deprecated and will be removed in a future version."
    )
    def solve_on_one_point(self, kpt, mesh_indices):
        r"""
        """
        # deprecation warning
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
        per_orb = self.model.orb_vecs[:, per_dir]

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

        if k_dir not in self.model.per:
            raise Exception(
                "Periodic boundary condition can be specified only along periodic directions!"
            )

        if not _is_int(mesh_dir):
            raise TypeError("mesh_dir should be an integer!")
        if mesh_dir < 0 or mesh_dir >= self.ndims:
            raise IndexError("mesh_dir outside the range!")
        
        orb_vecs = self.model.orb_vecs
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
        :func:`pythtb.WFArray.solve_mesh` method, this function
        should not be used since it will be called automatically by
        the code.

        This function will impose these periodic boundary conditions along
        one direction of the array. We are assuming that the k-point
        mesh increases by exactly one reciprocal lattice vector along
        this direction. 

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
        if self.dim_k == 0:
            raise ValueError(
                "Cannot impose periodic boundary conditions in 0D k-space.\n"
                "Use `impose_loop` instead."
                )
        if k_dir not in self.model.per:
            raise ValueError(
                "Periodic boundary condition can be specified only along periodic directions!"
            )

        self._pbc_axes.append(mesh_dir)
        self.mesh.loop_axis(mesh_dir, k_dir)
        self.mesh.close_axis(mesh_dir, k_dir)

        phase, slc_lft, slc_rt = self._get_pbc_phases(mesh_dir, k_dir)

        # Set the last point along mesh_dir axis equal to first
        # multiplied by the phase factor
        self._wfs[slc_lft] = self._wfs[slc_rt] * phase

        if self.u_nk is not None:
            # Set the last point along mesh_dir axis equal to first
            # multiplied by the phase factor
            self._u_wfs[slc_lft] = self._u_wfs[slc_rt] * phase
            self._psi_wfs[slc_lft] = self._psi_wfs[slc_rt]

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
        if mesh_dir in self.mesh.k_axes and self.mesh.is_k_torus:
            raise ValueError("Cannot impose loop condition on periodic k-space axis.")

        self._loop_axes.append(mesh_dir)
        self.mesh.close_axis(mesh_dir, mesh_dir)

        slc_lft, slc_rt = self._edge_slices(mesh_dir)
        self._wfs[slc_lft] = self._wfs[slc_rt]
        if self.u_nk is not None:
            self._u_nk[slc_lft] = self._u_nk[slc_rt]
        if self.psi_nk is not None:
            self._psi_nk[slc_lft] = self._psi_nk[slc_rt]

    def _unit_shift(self, axis: int):
        """Return an integer shift vector with +1 along *axis* over sampling axes."""
        v = [0] * self.ndims
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
    
    def _boundary_phase_for_shift(self, idx_vec):
        """Compute exp(-i G·r) mask for a multi-axis integer shift.

        The returned array is broadcast to match the stored state tensor shape
        (including lambda axes and the state axis). For spinful models, it
        returns a last axis of size norb*nspin when flatten_spin=True, or
        (..., norb, nspin) when flatten_spin=False.
        """
        nks = np.array(self.nks)
        dim_k = nks.size
        if dim_k == 0:
            return np.array(1.0, dtype=complex)

        mesh = self.mesh

        # Use shift only on sampling axes that are *periodic*
        k_shift = []
        for local_k_axis, mesh_axis in enumerate(mesh.k_axes):
            sh = int(idx_vec[mesh_axis])
            if mesh.is_axis_periodic(mesh_axis):
                k_shift.append(sh)
            elif mesh.is_axis_closed(mesh_axis):
                print(f"Warning: axis {mesh_axis} is closed; removing shift.")
                k_shift.append(0)
            else:
                print(f"Warning: axis {mesh_axis} is not periodic; removing shift.")
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
        for local_k_axis, mesh_axis in enumerate(mesh.k_axes):
            for c in range(dim_k):
                if mesh.is_axis_periodic(mesh_axis, c):
                    M[local_k_axis, c] = 1

        # Project signed wraps from k-axes to k-components
        G_comp = np.einsum('...i, ic -> ...c', G_axes, M, dtype=int)

        # Dot with orbital positions τ (reduced coords), shape (norb, dim_k)
        # Use only the real-space components that actually correspond to k-space
        # directions, in the *model.per* order. This matters when dim_k < dim_r or
        # when the periodic axes are not the first dim_k Cartesian components.
        per = getattr(self.model, "per", None)
        if per is None:
            # Fallback to the first dim_k components, but warn because this may be wrong
            if self.dim_k != self.model.dim_r:
                logger.warning(
                    "WFArray._boundary_phase_for_shift: model.per is missing; "
                    "falling back to first dim_k real-space components."
                )
            orb = self.model.orb_vecs[:, :dim_k]
        else:
            per = np.asarray(per, dtype=int)
            if per.size < dim_k:
                raise ValueError(
                    f"model.per lists only {per.size} periodic directions but dim_k={dim_k}."
                )
            # Select exactly dim_k periodic components in the order used for k-space
            orb = self.model.orb_vecs[:, per[:dim_k]]

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

    
    def roll_states_with_bc(self, idx_vec, flatten_spin=True):
        """Toroidal/open neighbor access with correct boundary phase and shape."""
        states = self.wfs 
        mesh = self.mesh

        if len(idx_vec) < mesh.num_k_axes:
            raise ValueError("idx_vec must have at least as many elements as k-axes in the mesh.")
        elif len(idx_vec) > mesh.num_axes:
            raise ValueError("idx_vec must have at most as many elements as total axes in the mesh.")

        rolled = states
        for ax, sh in enumerate(idx_vec):
            if not sh:
                continue
            if not mesh.is_axis_closed(ax):
                rolled = np.roll(rolled, shift=-int(sh), axis=ax)
            else:
                print(f"Applying bounded shift {sh} to axis {ax} without wrapping.")
                rolled = self._bounded_shift(rolled, axis=ax, sh=-int(sh))

        phase = self._boundary_phase_for_shift(tuple(idx_vec))
        rolled = rolled * phase

        if flatten_spin and self.nspin == 2:
            rolled = rolled.reshape(*rolled.shape[:-2], self.norb * self.nspin)
        return rolled 
    

    def get_overlap_mat(self):
        r"""Compute the overlap matrix of the cell periodic eigenstates on nearest neighbor k-shell.

        Overlap matrix is of the form

        .. math::
            M_{m,n}^{\mathbf{b}}(\mathbf{k}, \lambda) 
            = \langle u_{m, \mathbf{k}, \lambda} | u_{n, \mathbf{k+b}, \lambda} \rangle

        where :math:`\mathbf{b}` is a displacement vector connecting nearest neighbor k-points. 

        Returns
        -------
        M : np.ndarray
            Overlap matrix with shape [nk1-1, ..., nkd-1, *shape_lambda, num_nnbrs, n_states, n_states]

        Notes
        -----
        The overlap matrix is computed using periodic boundary conditions, and the
        last point in the mesh along each k-direction is excluded from the computation.
        """

        # Assumes only one shell for now
        _, idx_shell = self.get_k_shell(n_shell=1, report=False)
        idx_shell = idx_shell[0]

        # overlap matrix
        M = np.zeros(
            (*self.mesh_shape, len(idx_shell), self.nstates, self.nstates),
            dtype=complex,
        )

        u_nk = self.get_states(flatten_spin=True)
        for idx, idx_vec in enumerate(idx_shell):  # nearest neighbors
            # introduce phases to states when k+b is across the BZ boundary
            states_pbc = self.roll_states_with_bc(idx_vec, flatten_spin=True)
            M[..., idx, :, :] = np.einsum("...mj, ...nj -> ...mn", u_nk.conj(), states_pbc)

        return M
    

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
        The last points in the mesh of `U_forward` along the axes in ``dirs`` are excluded from the computation.

        If the ``Mesh`` has already been defined with periodic boundary conditions, or you have manually imposed
        PBC with either :func:`impose_pbc` or :func:`impose_loop`,
        then the last points are equal to the first points (modulo a phase factor in the PBC case). The last
        element of the links will be between the first and last wavefunctions in the original mesh.

        The overlap matrix is between the mesh of wavefunctions and their 'rolled' version along each
        axis in ``dirs``. In the rolled mesh of wavefunctions, the first element becomes the last. The overlap
        between this first element and last element in the original mesh are not between neighbors, since they
        are identical to each other. In the case where the ``Mesh`` does not have PBC, and the states at the beginning 
        and end are not neighbors, then this overlap at the edge is also not between neighbors and should be disregarded.

        In the situation where the first element and last element are truly neighbors, which would happen in the case
        where the ``Mesh`` excludes endpoints and still has a periodic topology, then the overlap at the edge should be 
        manually computed and appended to the links. Be aware that a phase factor must be multiplied if the state is beyond 
        the first Brillouin zone. It is often preferred to use a full ``Mesh`` that includes the endpoints of the Brillouin zone, 
        in which case the last element of the link is the overlap between the edge of the BZ and the beginning. 
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
            wfs_shifted = self.roll_states_with_bc(idx_vec=self._unit_shift(mu), flatten_spin=True)
            wfs_shifted = np.take(wfs_shifted, state_idx, axis=-2)

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
        
        Computes the Wilson loop unitary matrix and its eigenvalues for multiband Berry phases.
        The Wilson loop is a geometric quantity that characterizes the topology of the
        band structure. It is defined as the product of the overlap matrices between
        neighboring wavefunctions in the loop. Specifically, it is given by

        .. math::

            U_{Wilson} = \prod_{n} U_{n}

        where :math:`U_{n}` is the unitary part of the overlap matrix between neighboring wavefunctions
        in the loop, and the index :math:`n` labels the position in the loop 
        (see :func:`get_links` for more details).

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

        Notes
        ------
        Multiband Berry phases always returns numbers between :math:`-\pi` and :math:`\pi`.
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
            eval_pha = -np.angle(evals)  # Multiband Berry phases
            # sort the eigenvalues
            eval_pha = np.sort(eval_pha)
            return U_wilson, eval_pha
        else:
            return U_wilson

    @staticmethod
    def berry_loop(wfs_path, evals=False):
        r"""Berry phase along a one-dimensional path of wavefunctions.

        When ``evals=False``, the Berry phase is computed as the logarithm
        of the determinant of the product of the overlap matrices between
        neighboring wavefunctions in the path. In otherwords, the Berry phase is
        given by the formula:

        .. math::

            \phi = -\text{Im} \ln \det U_{\rm Wilson}

        where :math:`U` is the Wilson loop unitary matrix obtained from
        :func:`wilson_loop`. 

        When ``evals=True``, the function returns an array of
        the individual phases (multiband Berry phases) for each band. 
        They are computed as

        .. math::

            \phi_n = -\text{Im} \ln \lambda_n

        where :math:`\lambda_n` are the eigenvalues of the Wilson loop 
        unitary matrix. These multiband Berry phases correspond to the
        "maximally localized Wannier centers" or "Wilson loop eigenvalues".

        Parameters
        ----------
        wfs_loop : np.ndarray
            Wavefunctions in the path, with shape ``(path_idx, band, orbital, spin)``. 
        evals : bool, optional
            Default is `False`. If `True`, will return the argument of the eigenvalues
            of the Wilson loop unitary matrix instead of the total Berry phase.
            If False, will return the total Berry phase for the loop.

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

        U_wilson = WFArray.wilson_loop(wfs_path, evals=evals)

        if evals:
            hwf_centers = U_wilson[1]
            return hwf_centers
        else:
            berry_phase = -np.angle(np.linalg.det(U_wilson))
            return berry_phase
        
        
    def berry_phase(
            self,
            mu: int, 
            state_idx=None, 
            berry_evals: bool = False,
            contin: bool = True
            ):
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
        mu : int
            Index of WFArray direction along which Berry phase is
            computed. This parameters needs not be specified for
            a one-dimensional WFArray.

        state_idx : int, array-like, optional
            Optional band index or array of band indices to be included
            in the subsequent calculations. If unspecified, all bands are 
            included.
        
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
            forced to be between :math:`-\pi` and :math:`\pi` (if ``contin=False``) or
            they are made to be continuous (if ``contin=True``).

        Returns
        -------
        phase :
            If ``berry_evals=False`` (default value) then
            returns the Berry phase for each string. For a
            one-dimensional WFArray this is just one number. For a
            higher-dimensional `WFArray` *pha* contains one phase for
            each one-dimensional string in the following format. For
            example, if *WFArray* contains k-points on mesh with
            indices ``[i,j,k]`` and if direction along which Berry phase
            is computed is ``mu=1`` then `phase` will be two dimensional
            array with indices ``[i,k]``, since Berry phase is computed
            along second direction.

            If ``berry_evals = True`` then for
            each string returns phases of all eigenvalues of the
            product of overlap matrices. In the convention used for
            previous example, `phase` in this case would have indices
            ``[i,k,n]`` where ``n`` refers to index of individual phase of
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
        if not isinstance(mu, int) or mu < 0 or mu >= self.ndims:
            raise ValueError(f"mu must be an integer in [0, {self.ndims-1}]")

        # States (optionally restricted to a subspace)
        u = self.get_states(state_idx=state_idx, flatten_spin=True)
        u_expanded = self.get_states(state_idx=state_idx, flatten_spin=False)

        u_loop = u  # init wf loop

        # Check for open periodic boundary conditions
        for ax, comp in self.mesh.periodic_axes:
            # If mu axis is periodic and open, we need to append 
            # the first state to the end
            if ax == mu and not self.mesh.is_axis_closed(ax, comp):
                # If component is along k, apply PBC phase
                if comp < self.dim_k:
                    phase, _, _ = self._get_pbc_phases(ax, comp)
                    u_first = np.take(u_expanded, 0, axis=mu)
                    state_last = u_first * phase
                # If ax is not a k-axis, no phase is applied
                else:
                    state_last = np.take(u_expanded, 0, axis=mu)

                # flatten spin
                if self.nspin == 2:
                    state_last = state_last.reshape(*state_last.shape[:-2], -1)
            
                u_loop = np.concatenate([u_loop, np.expand_dims(state_last, axis=mu)], axis=mu)

        # Bring loop axis first for easy slicing over transverse axes
        u_loop = np.moveaxis(u_loop, mu, 0)
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

            val = self.berry_loop(wf_line, evals=berry_evals)
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
                    raise ValueError("\n\nWrong dimensionality!")

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
        

    def berry_flux(self, state_idx=None, plane=None, abelian=True):
        r"""Berry flux tensor.

        .. versionremoved:: 2.0.0
            The `individual_phases` parameter has been removed.

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
            \mathrm{Im}\ln\det[U_{\mu}(\mathbf{k}) U_{\nu}(\mathbf{k} + \hat{\mu}) 
            U_{\mu}^{-1}(\mathbf{k} + \hat{\nu}) U_{\nu}^{-1}(\mathbf{k})].

        The (non-Abelian) Berry flux tensor is computed by taking the 
        imaginary part of the matrix logarithm of the product of the link matrices
        around the plaquettes. It is defined as

        .. math::

            \mathcal{F}_{\mu\nu}(\mathbf{k}) =
            \mathrm{Im} \,\ln \Big[
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

        ndims = self.ndims  # Total dimensionality of adiabatic space: d
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
            self.mesh_shape
        )  # Number of points in adiabatic mesh: (nk1, nk2, ..., nkd)

        for ax in range(ndims):
            if not self.mesh.is_axis_closed(ax) and not self.mesh.is_axis_periodic(ax):
                # If the axis is not closed, remove the last point in that direction
                # to avoid computing flux across the boundary
                flux_shape[ax] -= 1

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
            dirs = [p, q]
            plane_idxs = 2

            shape = (*flux_shape, n_states, n_states) if not abelian else (*flux_shape,)
            berry_flux = np.zeros(shape, dtype=float)

        # U_forward: Unitary part of overlaps <u_{nk} | u_{n, k+delta k_mu}>
        U_forward = self.get_links(state_idx=state_idx, dirs=dirs)

        # Compute Berry flux for each pair of states
        for mu in range(plane_idxs):
            for nu in range(mu + 1, plane_idxs):
                U_mu = U_forward[mu]
                U_nu = U_forward[nu]

                # Shift the links along the mu and nu directions
                U_nu_shift_mu = np.roll(U_nu, -1, axis=mu)
                U_mu_shift_nu = np.roll(U_mu, -1, axis=nu)

                # Wilson loops: W = U_{mu}(k_0) U_{nu}(k_0+delta_mu) U^{-1}_{mu}(k_0+delta_mu+delta_nu) U^{-1}_{nu}(k_0)
                U_wilson = (
                    U_mu
                    @ U_nu_shift_mu
                    @ U_mu_shift_nu.conj().swapaxes(-1, -2)
                    @ U_nu.conj().swapaxes(-1, -2)
                )

                for ax in range(ndims):
                    if self.mesh.is_axis_closed(ax):
                        print(f"Axis {ax} is closed (includes endpoint). "
                              "Removing last point in the flux array to avoid overcounting.")
                        U_wilson = np.delete(U_wilson, -1, axis=ax)

                if not abelian:
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
    
    def berry_curv(
        self,
        dirs=None,
        state_idx=None,
        non_abelian=False,
        return_flux=False,
        Kubo=False
    ):
        r"""Berry curvature tensor.

        .. versionadded:: 2.0.0

        The difference between this function and :func:`berry_flux` is that this function computes a dimensionful
        Berry curvature tensor, while :func:`berry_flux` is dimensionless. Effectively, this function divides by
        the area of the plaquette (in reduced units) in parameter space. The reduced units are set by the mesh spacing
        in each direction.

        The Berry curvature can be approximated by the flux by simply dividing by the
        area of the plaquette, approximating the flux as a constant over the small loop.

        .. math::

            \Omega_{\mu\nu}(\mathbf{k}) \approx \frac{\mathcal{F}_{\mu\nu}(\mathbf{k})}{A_{\mu\nu}},

        where :math:`A_{\mu\nu}` is the area (in Cartesian units) of the plaquette in parameter space. 

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

        #TODO: Get rid of `Kubo` flag
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

            u_nk = self.get_states(flatten_spin=True)
            energies = self.energies
            # flatten k_dims
            u_nk = u_nk.reshape(-1, u_nk.shape[-2], u_nk.shape[-1])
            energies = energies.reshape(-1, energies.shape[-1])
            n_states = u_nk.shape[-2]

            if n_states != self.model.nstate:
                raise ValueError("Wavefunctions must be defined for all bands, not just a subset.")

            k_mesh = self.mesh.flat
            occ_idx = np.arange(n_states // 2)
            abelian = not non_abelian
            if dirs is None:
                dirs = 'all'
                b_curv = self.model.berry_curvature(k_mesh, evals=energies, evecs=u_nk, occ_idxs=occ_idx, abelian=abelian)
                b_curv = b_curv.reshape(*b_curv.shape[:2], *nks, *b_curv.shape[3:])
            else:
                b_curv = self.model.berry_curvature(k_mesh, evals=energies, evecs=u_nk, occ_idxs=occ_idx, abelian=abelian, dirs=dirs)
                b_curv = b_curv.reshape(*nks, *b_curv.shape[3:])

            return b_curv

        Berry_flux = self.berry_flux(state_idx=state_idx, non_abelian=non_abelian)
        Berry_curv = np.zeros_like(Berry_flux, dtype=complex)

        # Get delta vectors for each dimension in parameter space
        recip_lat_vecs = self.model.get_recip_lat()  # Expressed in inverse cartesian (x,y,z) coordinates
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
        mesh_idx: array-like of int
            Set of integers specifying the index of interest in the mesh.
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

        hwf : np.ndarray, optional
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
    

    # TODO allow for subbands
    def trace_metric(self):
        P = self._P
        Q_nbr = self._Q_nbr

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

    def interp_op(self, O_k, k_path, plaq=False):
        k_mesh = np.copy(self.mesh.grid)
        k_idx_arr = self.mesh.idx_arr
        nks = self.mesh.shape_k
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
