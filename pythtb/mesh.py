import numpy as np
from .tb_model import TBModel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tb_model import TBModel

__all__ = [
    "Mesh",
]


def _interpolate_path(nodes: np.ndarray, n_interp: int) -> np.ndarray:
    """
    Given `nodes` shape (R, D), returns a linear interpolation
    along each consecutive pair, totalling R*n_interp points.
    """
    segments = []
    for i in range(len(nodes)-1):
        start, end = nodes[i], nodes[i+1]
        t = np.linspace(0,1,n_interp,endpoint=False)
        segments.append(start[None,:] + (end-start)[None,:]*t[:,None])
    # add the final node
    segments.append(nodes[-1:,:])
    return np.vstack(segments)


"""
 def __init__(self, model: "TBModel", *nks):
        self.model = model
        self.nks = nks
        self.Nk = np.prod(nks)
        self.dim: int = len(nks)
        self.recip_lat_vecs = model.get_recip_lat()
        idx_grid = np.indices(nks, dtype=int)
        idx_arr = idx_grid.reshape(len(nks), -1).T
        self.idx_arr: list = idx_arr  # 1D list of all k_indices (integers)
        self.square_mesh: np.ndarray = self.gen_k_mesh(
            flat=False, endpoint=False
        )  # each index is a direction in k-space
        self.flat_mesh: np.ndarray = self.gen_k_mesh(
            flat=True, endpoint=False
        )  # 1D list of k-vectors

        # nearest neighbor k-shell
        self.nnbr_w_b, _, self.nnbr_idx_shell = self.get_weights(N_sh=1)
        self.num_nnbrs = len(self.nnbr_idx_shell[0])

        # matrix of e^{-i G . r} phases
        self.bc_phase = self.get_boundary_phase()
        self.orb_phases = self.get_orb_phases()
"""

class Mesh:
    def __init__(
            self,  
            dim_k: int, 
            dim_param: int, 
            axis_types: list[str],
            axis_names: list[str] = None
            ):
        r"""Initialize a Mesh object for a given TBModel.

        This class is responsible for constructing the mesh in k-space and parameter space.
        It provides methods to build both grid and path representations of the mesh, or a custom mesh
        with user-defined points.

        Parameters
        ----------
        model : TBModel
            The tight-binding model to use.
        axis_types : list[str]
            A list of axis types, which can be 'k', 'param'.
        axis_names : list[str], optional
            A list of axis names, which can be used to label the axes in plots or output.
        dim_param : int, optional
            The dimensionality of the parameter space. If not specified, it will be inferred from the axis types.

        See Also
        --------
        :ref:`haldane-bp-nb`
        :ref:`kane-mele-nb`
        :ref:`3site-cycle-nb`
        :ref:`3site-cycle-fin-nb`
        :ref:`cubic-slab-hwf-nb`
        :ref:`haldane-hwf-nb

        Notes
        -----
        The choice of 'l' for naming parameter axes is arbitrary but follows the convention of adiabatic parameters
        being labeled with :math:`\lambda`.

        Paths through a mixed parameter and k-space are not currently supported. 
        """
        
        self._dim_k = dim_k

        self._k_axes = [i for i, at in enumerate(axis_types) if at == 'k']
        self._param_axes = [i+self.num_k_axes for i, at in enumerate(axis_types) if at == 'param']

        if axis_names is None:
            axis_names = [f"k_{i}" for i in range(self.dim_k)] + [f"l_{i}" for i in range(self.num_param_axes)]
        elif len(axis_types) != len(axis_names):
            raise ValueError("Axis types and axis names must have the same length.")
        if not all(at in ['k', 'param'] for at in axis_types):
            raise ValueError("Axis types must be either 'k', 'param', or 'path'.")

        self._axis_types = axis_types
        self._axis_names = axis_names

 
        if self.num_k_axes > dim_k:
            raise ValueError(f"Number of k axes ({self.num_k_axes}) cannot exceed model dimension ({dim_k}).")

        if dim_param is None:
            self._dim_param = self.num_param_axes
        else:
            self._dim_param = dim_param

        self._points = np.empty((0,) + (self._dim_k + self._dim_param,), dtype=float)

        # for grids
        self._shape_k = (0,) * self.num_k_axes
        self._shape_param = (0,) * self.num_param_axes

        # for paths
        self._nodes = None

        self.is_path = False
        self.is_grid = False
        self.is_custom = False

    @property
    def points(self):
        return self._points

    @property
    def flat(self):
        return self._points
    
    @property
    def grid(self):
        # if self.is_path:
        #     raise ValueError("Mesh is a path, not a grid. Use .points or .flat instead.")
        # else:
        #     return self._points.reshape(*self.grid_shape)
        
        return self._points.reshape(*self.grid_shape)
        
    @property
    def grid_shape(self):
        return self.shape_k + self.shape_param + (self.dim_k + self.dim_param,)
        
    @property
    def num_points(self):
        return self._points.shape[0]
    
    @property
    def k_axes(self):
        return self._k_axes

    @property
    def param_axes(self):
        return self._param_axes

    @property
    def num_k_axes(self):
        return len(self._k_axes)

    @property
    def num_param_axes(self):
        return len(self._param_axes)

    @property
    def shape_k(self):
        return self._shape_k

    @property
    def shape_param(self):
        return self._shape_param
    
    @property
    def dim_param(self):
        return self._dim_param

    @property
    def dim_k(self):
        return self._dim_k

    @property
    def dim_total(self):
        return self.dim_k + self.dim_param

    @property
    def axis_names(self):
        return self._axis_names
    
    @property
    def axis_types(self):
        return self._axis_types

    @property
    def nodes(self):
        return self._nodes 

    def build_path(self,
        nodes: np.ndarray,
        n_interp: int = 1
    ):
        """
        Build a k-path in the Brillouin zone.

        The `nodes` array must have the following shape:
            - shape ``(N_nodes, dim)`` for any k/parameter-path.

        Parameters
        ----------
        nodes : np.ndarray
            The k/parameter-path points in reduced coordinates.
        n_interp : int
            The number of interpolation points between each pair of nodes.
        """
        if self.num_k_axes + self.num_param_axes > 1:
            raise ValueError("For a path, must only have one axis type.")
        
        nodes = np.asarray(nodes, dtype=float)
        # make sure nodes are the right shape
        if nodes.ndim != 2:
            raise ValueError(f"Expected 2D array for nodes, got {nodes.ndim}D array.")
        if nodes.shape[1] != self._dim_k + self._dim_param:
            raise ValueError(f"Expected shape (N_nodes, {self._dim_k + self._dim_param}), got {nodes.shape}")
        self._nodes = nodes

        path = _interpolate_path(nodes, n_interp)
        self._points = path
        self.is_path = True

        if self.axis_types[0] == 'k':
            self._shape_k = path.shape[:-1]
        if self.axis_types[0] == 'param':
            self._shape_param = path.shape[:-1]


    def build_full_grid(self,
        shape: tuple | list,
        gamma_centered: bool = False,
        k_endpoints: bool = True,
        param_endpoints: bool = True
    ):
        """ Build a regular k-space and parameter space grid.

        The grid is a set of points in reduced units that form a cubic/square
        lattice in k-space and parameter space. The exact nature of the grid
        (e.g., the number of points, the spacing) is determined by the input
        parameters.

        .. warning::
            You must pass *either* ``full_grid=True``, or the ``points`` array.

        After calling this function, the ``.grid`` attribute will be:
            - shape ``(*shape_k, *shape_param, dim_k+dim_param)`` for full-grid,
        while the ``.flat`` attribute will be the flattened version:
            - shape ``(N_k*N_p, dim_k+dim_param)``.

        Parameters
        ----------
        points : np.ndarray
            The points in k-space and parameter space. The shape should be 
            ``(*shape_k, *shape_param, dim_k + dim_param)``
        shape_k : list or tuple of int with length dim_k
            The shape of the k-space grid.
        shape_param : list or tuple of int with length dim_param, optional
            The shape of the parameter space grid.
        full_grid : bool, optional
            If True, build a full grid in k-space and parameter space.
        gamma_centered : bool, optional
            If True, center the k-space grid at the Gamma point. This
            makes the grid axes go from -0.5 to 0.5.
        exclude_k_endpoints : bool, optional
            If True, exclude the endpoints of the k-space grid.
        exclude_param_endpoints : bool, optional
            If True, exclude the endpoints of the parameter space grid.
        """
        if not isinstance(shape, (tuple, list)):
            raise ValueError(f"Expected tuple or list for shape, got {type(shape)}")
        if len(shape) != self.num_k_axes + self.num_param_axes:
            raise ValueError(f"Expected {self.num_k_axes + self.num_param_axes} dimensions, got {len(shape)}")

        shape_k = shape[:self.num_k_axes]
        shape_param = shape[self.num_k_axes:]

        self._shape_k = shape_k
        self._shape_param = shape_param

        if len(shape_param) == 0:
            grid = self.gen_hyper_cube(
                *shape_k,
                centered=gamma_centered,
                flat=False,
                endpoint=k_endpoints
            )

            flat = grid.reshape(-1, grid.shape[-1])
        elif len(shape_k) == 0:
            grid = self.gen_hyper_cube(
                *shape_param,
                centered=False,
                flat=False,
                endpoint=param_endpoints
            )
            flat = grid.reshape(-1, grid.shape[-1])
        else:
            # generate k-space grid
            k_flat = self.gen_hyper_cube(
                *shape_k,
                centered=gamma_centered,
                flat=True,
                endpoint=k_endpoints
            )

            # generate parameter space grid
            p_flat = self.gen_hyper_cube(
                *shape_param,
                centered=False,
                flat=True,
                endpoint=param_endpoints
            )

            Nk, Np = k_flat.shape[0], p_flat.shape[0]

            k_rep = np.repeat(k_flat, Np, axis=0)
            p_rep = np.tile(p_flat, (Nk, 1))
            flat = np.hstack([k_rep, p_rep])

        self._points = flat
        self.is_grid = True

        # --- precompute k-space phases if present ---
        if self.dim_k > 0:
            pass
            # self.recip_lat_vecs = model.get_recip_lat()
            # self.orb_phases = self.get_orb_phases()
            # self.bc_phase = self.get_boundary_phase()

    def build_custom(self, points, is_path=False):
        """Build a custom mesh from the given points.

        Parameters
        ----------
        points : np.ndarray
            Array of shape (N1, N2, ..., Nd, dim_total) defining the mesh points.
        axis_types : list[str]
            List of axis types ('k' or 'param') corresponding to each axis in the mesh.

        Returns
        -------
        Mesh
            A Mesh object representing the custom mesh.

        Raises
        ------
        ValueError
            If the shape of points or the length of axis_types is inconsistent.
        """
        self.is_custom = True
        self.is_path = is_path

        if not isinstance(points, np.ndarray):
            raise ValueError("Mesh points must be a numpy array.")
        if points.ndim != len(self.grid_shape):
            raise ValueError("Inconsistent dimensions between mesh points and axis types.")

        self._points = np.reshape(points, (-1, points.shape[-1]))

        shape = points.shape[:-1]
        self._shape_k = tuple(shape[i] for i in self.k_axes)
        self._shape_param = tuple(shape[i] for i in self.param_axes)

    
    def _extract_axis_range(
            self, 
            axis_index: int, 
            component_index: int, 
            assume_periodic_wrap: bool = False, 
            tol: float = 1e-12
        ) -> np.ndarray:
        """
        Extract the unique 1D range along a mesh axis/component pair.
        Only for grids, not paths.
        """
        if not (self.is_grid or self.is_custom):
            raise ValueError("Axis range extraction only supported for grid meshes.")
        if self.points.size == 0:
            raise ValueError("Mesh points are not initialized.")
        shape = self.grid_shape
        n_axes = len(shape) - 1
        if axis_index < 0 or axis_index >= n_axes:
            raise IndexError(f"axis_index {axis_index} out of bounds for mesh with {n_axes} axes.")
        idx = [0] * n_axes
        idx[axis_index] = slice(None)
        idx = tuple(idx)
        arr = self.grid[idx + (component_index,)]
        arr = np.asarray(arr)
        # arr should be 1D
        if arr.ndim != 1:
            arr = np.reshape(arr, -1)
        if assume_periodic_wrap and arr.size > 1:
            diff = np.abs(((arr[-1] - arr[0]) % 1.0))
            if diff < tol:
                arr = arr[:-1]
        return arr

    def get_k_ranges(self, assume_periodic_wrap: bool = False, tol: float = 1e-12) -> dict:
        """
        Return a dict mapping k-axis names to their 1D ranges.
        """
        if self.num_k_axes == 0:
            raise ValueError("No k-axes present in this mesh.")
        # Indices of k axes and their corresponding component indices
        k_axis_indices = list(range(self.num_k_axes))
        k_comp_indices = list(range(self.num_k_axes))
        k_names = [n for n, t in zip(self.axis_names, self.axis_types) if t == 'k']
        result = {}
        for i, comp, name in zip(k_axis_indices, k_comp_indices, k_names):
            arr = self._extract_axis_range(i, comp, assume_periodic_wrap=assume_periodic_wrap, tol=tol)
            result[name] = arr
        return result

    def get_param_ranges(self, assume_periodic_wrap: bool = False, tol: float = 1e-12) -> dict:
        """
        Return a dict mapping parameter-axis names to their 1D ranges.
        """
        if self.num_param_axes == 0:
            raise ValueError("No parameter axes present in this mesh.")
        # Indices of parameter axes and their corresponding component indices
        param_axis_indices = list(range(self.num_k_axes, self.num_k_axes + self.num_param_axes))
        param_comp_indices = list(range(self.dim_k, self.dim_k + self.dim_param))
        param_names = [n for n, t in zip(self.axis_names, self.axis_types) if t == 'param']
        result = {}
        for i, comp, name in zip(param_axis_indices, param_comp_indices, param_names):
            arr = self._extract_axis_range(i, comp, assume_periodic_wrap=assume_periodic_wrap, tol=tol)
            result[name] = arr
        return result
    

    def get_k_points(self) -> np.ndarray:
        """
        Return the unique k-point mesh from the full grid, with shape (*shape_k, dim_k).
        Only works for grid-like meshes.
        """
        if not (self.is_grid or self.is_custom):
            raise ValueError("get_k_points only supported for grid or custom meshes.")
        if self.points.size == 0:
            raise ValueError("Mesh points are not initialized.")
        G = self.grid  # shape (*shape_k, *shape_param, dim_k+dim_param)
        Gk = G[..., :self.dim_k]
        num_k_axes = self.num_k_axes
        num_param_axes = self.num_param_axes
        # Build index: [slice(None)]*num_k_axes + [0]*num_param_axes + [slice(None)]
        idx = [slice(None)]*num_k_axes + [0]*num_param_axes + [slice(None)]
        Gk_unique = Gk[tuple(idx)]
        # Ensure correct shape
        Gk_unique = np.asarray(Gk_unique)
        shape_k = self.shape_k
        if Gk_unique.shape != shape_k + (self.dim_k,):
            Gk_unique = Gk_unique.reshape(shape_k + (self.dim_k,))
        return Gk_unique

    def get_param_points(self) -> np.ndarray:
        """
        Return the unique parameter-point mesh from the full grid, with shape (*shape_param, dim_param).
        Only works for grid-like meshes.
        """
        if not (self.is_grid or self.is_custom):
            raise ValueError("get_param_points only supported for grid or custom meshes.")
        if self.points.size == 0:
            raise ValueError("Mesh points are not initialized.")
        G = self.grid  # shape (*shape_k, *shape_param, dim_k+dim_param)
        Gp = G[..., self.dim_k:]
        num_k_axes = self.num_k_axes
        num_param_axes = self.num_param_axes
        # Build index: [0]*num_k_axes + [slice(None)]*num_param_axes + [slice(None)]
        idx = [0]*num_k_axes + [slice(None)]*num_param_axes + [slice(None)]
        Gp_unique = Gp[tuple(idx)]
        # Ensure correct shape
        shape_param = self.shape_param
        if Gp_unique.shape != shape_param + (self.dim_param,):
            Gp_unique = Gp_unique.reshape(shape_param + (self.dim_param,))
        return Gp_unique

    @staticmethod
    def gen_hyper_cube(
        *n_points, centered: bool = False, flat: bool = True, endpoint: bool = False
    ) -> np.ndarray:
        """Generate a hypercube of points in the specified dimensions.

        Parameters
        ----------
        *n_points: int
            Number of points along each dimension.
    
        centered: bool, optional
            If True, mesh is defined from [-0.5, 0.5] along each direction.
            Defaults to False.
        flat: bool, optional
            If True, returns flattened array of k-points (e.g. of shape ``(n1*n2*n3 , 3)``).
            If False, returns reshaped array with axes along each k-space dimension
            (e.g. of shape ``(1, n1, n2, n3, 3)``). Defaults to True.
        endpoint: bool, optional
            If True, includes 1 (edge of BZ in reduced coordinates) in the mesh. Defaults to False.
            When Wannierizing should omit this point.

        Returns
        -------
        mesh: np.ndarray
            Array of coordinates defining the hypercube. 
        """

        end_pts = [-0.5, 0.5] if centered else [0, 1]
        vals = [
            np.linspace(end_pts[0], end_pts[1], n, endpoint=endpoint)
            for n in n_points
        ]
        flat_mesh = np.stack(np.meshgrid(*vals, indexing="ij"), axis=-1)

        return flat_mesh if not flat else flat_mesh.reshape(-1, len(vals))
    
