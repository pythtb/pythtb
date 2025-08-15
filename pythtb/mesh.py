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


class Mesh:
    def __init__(
            self,  
            dim_k: int, 
            axis_types: list[str],
            dim_lambda: int | None = None,
            axis_names: list[str] = None           
            ):
        r"""Initialize a Mesh object for a given TBModel.

        This class is responsible for constructing the mesh in k-space and parameter space.
        It provides methods to build both grid and path representations of the mesh, or a custom mesh
        with user-defined points.

        Parameters
        ----------
        dim_k : int
            The dimensionality of k-space. This will influence the dimension of
            the vector at each mesh point.
        axis_types : list[str]
            A list of axis types, which can be 'k' or 'l' for k-space and parametric
            space respectively. The length of this list will determine the number of 
            dimensions in the mesh.
        dim_lambda : int, optional
            The dimensionality of parameter space. If left unspecified, the
            parameter space dimension will be automatically inferred from the axis types.
            If creating a path through a higher-dimensional parameter space, this must be
            specified.
        axis_names : list[str], optional
            A list of axis names, which can be used for parametrically populating
            a `WFArray`. If unspecified, default names will be generated.
            See examples listed below for more details.

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
        The parameter space is assumed to be orthogonal to the k-space. This means that when varying the parameter
        along its axis, the k-components are held fixed. Paths through a mixed parameter and k-space are not 
        currently supported.

        The choice of 'l' for naming parameter axes is arbitrary but follows the convention of adiabatic parameters
        being labeled with :math:`\lambda`.
        """
        
        self._dim_k = dim_k

        # Axis types
        if not all(at in ['k', 'l'] for at in axis_types):
            raise ValueError("Axis types must be either 'k', or 'l'.")
        
        self._axis_types = axis_types

        # Naming axes
        if axis_names is None:
            axis_names = [f"k_{i}" for i in range(self.num_k_axes)] + [f"l_{i}" for i in range(self.num_lambda_axes)]
        elif len(axis_types) != len(axis_names):
            raise ValueError("Axis types and axis names must have the same length.")
        self._axis_names = axis_names 

        # Dimension of parameter space
        if dim_lambda is None:
            self._dim_lambda = self.num_lambda_axes
        else:
            self._dim_lambda = dim_lambda

        # Define component types for the last coordinate axis: first dim_k are 'k', then parameters
        self._component_types = tuple(['k'] * self.dim_k + ['l'] * self.dim_lambda)

        if self.num_k_axes > dim_k:
            raise ValueError(f"Number of k axes ({self.num_k_axes}) cannot exceed specified dimension ({dim_k}).")

        self._points = np.empty((0,) + (self._dim_k + self._dim_lambda,), dtype=float)

        # for grids
        self._shape_k = (0,) * self.num_k_axes
        self._shape_lambda = (0,) * self.num_lambda_axes

        # for paths
        self._nodes = None

        # pbc info
        self._pbc_mask = np.zeros((self.num_axes, self.dim_total), dtype=bool)

    @property
    def points(self):
        """Mesh point array of shape (N_points, dim_k + dim_lambda)."""
        return self._points

    @property
    def flat(self):
        """Alias for .points. Shape (N_points, dim_k + dim_lambda)."""
        return self._points
    
    @property
    def grid(self):
        """Mesh point array of shape (N1, ..., Nd, dim_k + dim_lambda)."""
        return self._points.reshape(*self.shape_full)

    @property
    def nodes(self):
        return self._nodes 
    
    @property
    def filled(self):
        """True if the mesh is filled (i.e., contains points)."""
        return not self.points.size == 0

    # ---- Axis properties ----
    @property
    def axis_names(self):
        """Tuple of axis names."""
        return self._axis_names
    
    @axis_names.setter
    def axis_names(self, names: list[str]):
        if len(names) != len(self.axis_types):
            raise ValueError("Axis names must match the number of axes.")
        self._axis_names = names

    @property
    def axis_types(self) -> tuple[str]:
        """Tuple of k/parameter-axis types."""
        return self._axis_types

    @property
    def axis_sizes(self) -> list[int]:
        """Tuple of axis sizes."""
        return [*self.shape_k, *self.shape_lambda]

    @property
    def shape_k(self) -> tuple[int]:
        """Mesh shape defining the layout of k-axis points."""
        return self._shape_k

    @property
    def shape_lambda(self) -> tuple[int]:
        """Mesh shape defining the layout of parameter-axis points."""
        return self._shape_lambda
    
    @property
    def shape_full(self) -> tuple[int]:
        """Shape of full grid (*shape_k, *shape_lambda, dim_k + dim_lambda)."""
        return self.shape_k + self.shape_lambda + (self.dim_k + self.dim_lambda,)

    @property
    def shape_mesh(self) -> tuple[int]:
        """Shape of mesh grid (*shape_k, *shape_lambda)."""
        return self.shape_k + self.shape_lambda

    @property
    def shape_flat(self) -> tuple[int]:
        """Shape of flattened mesh points (N_points, dim_total)."""
        return (self.num_points, self.dim_total)
    
    @property
    def num_points(self) -> int:
        """Number of mesh points."""
        return int(np.prod(self.shape_k) * np.prod(self.shape_lambda))

    @property
    def k_axes(self) -> list[int]:
        """Tuple of k-axis indices."""
        return [i for i, at in enumerate(self.axis_types) if at == 'k']

    @property
    def lambda_axes(self) -> list[int]:
        """Tuple of lambda-axis indices."""
        return [i for i, at in enumerate(self.axis_types) if at == 'l']

    @property
    def num_k_axes(self) -> int:
        """Number of k-axes."""
        return len(self.k_axes)

    @property
    def num_lambda_axes(self) -> int:
        """Number of parameter-axes."""
        return len(self.lambda_axes)
    
    @property
    def num_axes(self) -> int:
        """Number of axes (k + lambda)."""
        return self.num_k_axes + self.num_lambda_axes
    
    @property
    def is_grid(self) -> bool:
        r"""True if the mesh is a grid (as opposed to a path).

        A grid mesh has an axis for each dimension of the mesh.
        """
        return self.num_axes == self.dim_total
    
    @property
    def is_full_k(self) -> bool:
        r"""True if the mesh is a full grid along k-axes.

        A full grid mesh has an axis for each dimension of the mesh, and includes all
        points in the grid. This only considers the k-space axes/dimensions. If 
        any of the k-space axes are not periodic, then the mesh is not a full grid.
        """
        if not self.is_grid:
            return False
        
        pbc_axes = self.periodic_axes
        if len(pbc_axes) == 0:
            return False

        k_axes = np.arange(self.num_k_axes)
        periodic_axes = [ax for ax, _ in pbc_axes]
        # Check if all k axes are periodic
        for axis in k_axes:
            if axis not in periodic_axes:
                return False

        return True

    # ---- Vector component properties ----
    @property
    def dim_lambda(self) -> int:
        """Dimension of parameter space."""
        return self._dim_lambda

    @property
    def dim_k(self) -> int:
        """Dimension of k-space."""
        return self._dim_k

    @property
    def dim_total(self) -> int:
        """Dimension of the full mesh space (k + lambda)."""
        return self.dim_k + self.dim_lambda

    @property
    def component_types(self) -> tuple[str]:
        """Tuple of length dim_total labeling vector components as 'k' or 'l'."""
        return self._component_types

    @property
    def pbc_mask(self):
        """Mask array indicating periodicity per vector component."""
        if not self.filled:
            return None
        return getattr(self, '_pbc_mask', None)

    @property
    def periodic_axes(self) -> list[tuple[int, int]]:
        """List of (mesh_axis, component_index) pairs that wrap by ~1."""
        if not self.filled:
            return []
        
        mat = self._pbc_mask  # (n_axes, dim_total)
        pbc_axes = []
        for axis_idx in range(mat.shape[0]):
            for comp_idx in range(self.dim_total):
                if mat[axis_idx, comp_idx]:
                    pbc_axes.append((axis_idx, comp_idx))

        return pbc_axes

    def summary(self) -> str:
        """Human-friendly multi-line summary of this Mesh."""
        # Helpers
        def _fmt_tuple(t):
            return "(" + ", ".join(str(x) for x in t) + ")"

        def _fmt_list(lst):
            return "[" + ", ".join(str(x) for x in lst) + "]"

        def _yn(val):
            return "yes" if bool(val) else "no"

        # Mesh type
        if not self.filled:
            mesh_type = "uninitialized"
        elif getattr(self, "is_grid", False):
            mesh_type = "grid"
        else:
            mesh_type = "path"

        # Shapes
        shape_k = self.shape_k
        shape_p = self.shape_lambda
        overall_shape = self.shape_full

        # Endpoints per axis if available
        axis_eps = getattr(self, "axis_includes_endpoints", None)
        eps_str = None
        if axis_eps is not None and len(axis_eps) > 0:
            eps_str = "[" + ", ".join("•" if e else "◦" for e in axis_eps) + "]"
            # legend: • includes endpoints, ◦ excludes/unknown
            
        # Full grid (optional flag some versions have)
        is_full_k = getattr(self, "is_full_k", None)

        # Periodic axes (if you’re caching them somewhere)
        periodic_axes = self.periodic_axes
        if periodic_axes:
            pa_str = ", ".join(f"(axis {a}, comp {c})" for a, c in periodic_axes)
        else:
            pa_str = "None"

        # Count points
        num_points = self.num_points

        # Names / indices
        axis_names = getattr(self, "_axis_names", None) or getattr(self, "axis_names", None)
        k_axes = getattr(self, "k_axes", [])
        p_axes = getattr(self, "lambda_axes", [])

        lines = []
        lines.append("Mesh Summary")
        lines.append("=" * 40)
        lines.append(f"Type: {mesh_type}")
        lines.append(f"Dimensionality: {self.dim_k} k-dim(s) + {self.dim_lambda} λ-dim(s)")
        lines.append(f"Number of mesh points: {num_points}")
        lines.append(f"Full shape: {_fmt_tuple(overall_shape)}")
        lines.append(f"k-shape: {_fmt_tuple(shape_k)}")
        lines.append(f"λ-shape: {_fmt_tuple(shape_p)}")
        lines.append(f"k-axes: {_fmt_list(k_axes)}")
        lines.append(f"λ-axes: {_fmt_list(p_axes)}")
        lines.append(f"Axis names: {_fmt_list(axis_names)}")
        if eps_str is not None:
            # Add legend only once
            lines.append(f"Axis endpoints included: {eps_str}   (• yes, ◦ no/unknown)")
        # Optional full-grid flag
        if is_full_k is not None and mesh_type != "path":
            lines.append(f"Full k-space (endpoints on all k-axes): {_yn(is_full_k)}")
        # Periodicity info if you compute/cache it
        lines.append(f"Periodic axes: {pa_str}")
        return "\n".join(lines)

    def __str__(self) -> str:
        # Pretty, multi-line view for print(mesh)
        return self.summary()

    def __repr__(self) -> str:
        # Concise single-line debug view for interactive REPL / lists
        try:
            npts = getattr(self, "num_points", None)
            if npts is None:
                pts = getattr(self, "_points", None)
                npts = int(pts.shape[0]) if pts is not None else "?"
            shape = None
            G = getattr(self, "grid", None)
            if G is not None:
                shape = G.shape[:-1]
            return f"<Mesh type={'path' if getattr(self,'is_path',False) else 'grid' if getattr(self,'is_grid',False) else 'custom' if getattr(self,'is_custom',False) else 'uninitialized'} dim_k={self.dim_k} dim_lambda={self.dim_lambda} shape={shape} points={npts}>"
        except Exception:
            return "<Mesh ...>"

    def _set_pbc_info(self, components: str = "k", tol: float = 1e-8) -> np.ndarray:
        r"""
        Determine per-axis / per-component wraps purely from mesh data.

        Returns a boolean array `wraps` of shape (n_axes, dim_total) where
        wraps[a, c] is True iff sampling along axis `a` wraps component `c`
        (first slice vs last slice differ by ≈ 1 modulo 1).

        Parameters
        ----------
        components : {"k","all"}
            - "k": Only report wraps for k-components (0..dim_k-1). Param components are forced False.
            - "all": Test and report for all components (k and lambda).
        tol : float
            Tolerance for detecting a wrap ≈ 1 modulo 1. Use ~1e-8 for double.

        Notes
        -----
        - For **paths** (self.is_path == True): there is effectively one sampling axis.
        We compare the first and last node of the path for each component.
        - For **grids/custom non-path**: along each sampling axis, we compare the first
        and last hyperfaces for each component. We require that *all points* on those
        two faces differ by ≈ 1 modulo 1 for that component to be marked as wrapping.
        - This function infers wraps from the sampled data only. If your grid *does not*
        include endpoints on an axis, you won't see wraps along that axis (which is
        usually what you want to avoid double counting).
        """
        dim_k = self.dim_k
        dim_p = self.dim_lambda
        dim_total = dim_k + dim_p

        # number of sampling axes in the mesh (order: *shape_k, *shape_lambda)
        n_axes = self.num_k_axes + self.num_lambda_axes
        wraps = np.zeros((n_axes, dim_total), dtype=bool)

        # Mask which components to consider (k-only or all)
        comp_mask = np.zeros(dim_total, dtype=bool)
        if components == "k":
            comp_mask[:dim_k] = True
        elif components == "all":
            comp_mask[:] = True
        else:
            raise ValueError("components must be 'k' or 'all'")
        
        if self.points.size == 0:
            raise ValueError("Mesh points are not initialized.")

        if not self.is_grid:
            # For paths, treat as a single sampling axis (index 0).
            start = np.asarray(self.points[0], dtype=float)        # (dim_total,)
            end   = np.asarray(self.points[-1], dtype=float)       # (dim_total,)
            delta = abs(end - start)                         # (dim_total,)
            wraps_path = (abs(delta - 1) < tol) & comp_mask
            # If metadata says more than one axis (unlikely for a path), fill only first row
            wraps[0, :] = wraps_path
            self._pbc_mask = wraps
            return

        G = self.grid  # shape (*shape_k, *shape_lambda, dim_total)
        shape = G.shape[:-1]  # per-axis sizes
        if len(shape) != n_axes:
            # Inconsistent metadata; bail out safely
            self._pbc_mask = wraps
            return

        # Iterate over sampling axes; compare first vs last hyperfaces.
        for axis_idx in range(n_axes):
            # Build two index tuples differing only at 'axis_idx'
            slc_first = [0] * n_axes
            slc_last  = [-1] * n_axes
            # Turn the rest into full slices to keep all other indices
            for i in range(n_axes):
                if i != axis_idx:
                    slc_first[i] = slice(None)
                    slc_last[i]  = slice(None)
            slc_first = tuple(slc_first)
            slc_last  = tuple(slc_last)

            # Extract the two hyperfaces: shapes (..., dim_total)
            vals_first = np.asarray(G[slc_first], dtype=float)
            vals_last  = np.asarray(G[slc_last],  dtype=float)

            # Differences modulo 1 along the component axis (last axis)
            delta = abs(vals_last - vals_first) # (..., dim_total)

            # Reduce across all non-component axes: require *all* points to satisfy wrap
            if delta.ndim == 1:
                face_wraps = abs(delta - 1) < tol                           # (dim_total,)
            else:
                # all points on the hyperfaces must wrap that component
                face_wraps = np.all(abs(delta - 1) < tol, axis=tuple(range(delta.ndim - 1)))

            wraps[axis_idx, :] = face_wraps & comp_mask

        self._pbc_mask = wraps

    def make_lambda_loop(self, lambda_idx: int):
        r"""Sets the periodic boundary condition (PBC) mask for a specific lambda axis.

        Parameters
        ----------
        lambda_idx : int
            The index of the lambda axis to set the PBC mask for. Must
            be in the list of valid lambda axes in ``self.lambda_axes``.

        Notes
        -----
        This method modifies the PBC mask in place along the diagonal.
        ``mask[lambda_idx, self.dim_k + lambda_idx]`` will be set to True.
        What this means is that the ``dim_k + lambda_idx`` component of the
        mesh vector winds along the ``lambda_idx`` axis. In most cases this
        is the expected behavior.
        """
        if not self.filled:
            return None

        if lambda_idx not in self.lambda_axes:
            raise ValueError(f"Invalid lambda index: {lambda_idx}")

        self._pbc_mask[lambda_idx, self.dim_k + lambda_idx - 1] = True

    def build_path(self,
        nodes: np.ndarray,
        n_interp: int = 1
    ):
        r"""
        Build a k-path in the Brillouin zone.

        The number of points along the path is determined by the number of
        interpolation points specified. For `N` nodes, there will be `N-1`
        segments, each with `n_interp` points, plus the endpoints. Thus, the
        total number of points will be 
        `N-1 + 1 + (N-1) * n_interp = N + (N-1) * n_interp`.

        Parameters
        ----------
        nodes : np.ndarray
            The k/parameter-path points in reduced coordinates.
            Must have the shape ``(N_nodes, dim_total)`` 
            for any k/parameter-path, where `dim_total` is the total number of 
            dimensions in the mesh defined by ``dim_total = dim_k + dim_lambda``.
        n_interp : int
            The number of interpolation points between each pair of nodes.

        Examples
        --------
        We can create a k-path by specifying the nodes in reduced coordinates.

        >>> nodes = np.array([[0, 0, 0], [0.5, 0.5, 0], [1, 1, 0]])
        >>> mesh.build_path(nodes, n_interp=5)

        Since we specified 5 interpolation points between the nodes, the resulting mesh 
        will have 10 points along the path.

        >>> mesh.points.shape
        (10, 3)
        """
        if self.num_k_axes + self.num_lambda_axes > 1:
            raise ValueError("For a path, must only have one axis type.")
        
        nodes = np.asarray(nodes, dtype=float)
        # make sure nodes are the right shape
        if nodes.ndim != 2:
            raise ValueError(f"Expected 2D array for nodes, got {nodes.ndim}D array.")
        if nodes.shape[1] != self._dim_k + self._dim_lambda:
            raise ValueError(f"Expected shape (N_nodes, {self._dim_k + self._dim_lambda}), got {nodes.shape}")
        self._nodes = nodes

        path = _interpolate_path(nodes, n_interp)
        self._points = path

        if self.axis_types[0] == 'k':
            self._shape_k = path.shape[:-1]
        if self.axis_types[0] == 'l':
            self._shape_lambda = path.shape[:-1]

        self._set_pbc_info()

    def build_grid(self,
        shape: tuple | list,
        gamma_centered: bool | list = False,
        k_endpoints: bool | list = True,
        lambda_endpoints: bool | list = True,
        lambda_start: int | float | list = 0.0,
        lambda_stop: int | float | list = 1.0
    ):
        r"""Build a regular k-space and lambda space grid.

        The grid is a set of points that have an axis for each dimension in the combined
        k-space and lambda space.

        The k-points (in reduced units) range from 0 to 1 along the k-axes, 
        unless gamma_centered is True, in which case they range from -0.5 to 0.5
        along the k-axes. The last points (1 or 0.5) are included if ``k_endpoints``
        flag is set to ``True`` (default is ``True``).

        The parameter points range from ``param_start`` to ``param_stop`` 
        along the parameter axes. If these are not specified, they will default
        to 0 and 1 respectively. The endpoints are included if ``param_endpoints``
        flag is set to ``True`` (default is ``True``).

        This function populates the ``.grid`` and ``.flat``/``.points`` attributes.
        After calling this function, the ``.grid`` attribute will be:
            - shape ``(*shape, dim_k+dim_lambda)`` for full-grid,
        while the ``.flat`` attribute will be the flattened version:
            - shape ``(np.prod(*shape), dim_k+dim_lambda)``.

        Parameters
        ----------
        shape : list or tuple of int with size ``len(axis_types)``
            The number of points along each axis.
        gamma_centered : bool, list[bool] optional
            If True, center the k-space grid at the Gamma point. This
            makes the grid axes go from -0.5 to 0.5. One may also specify
            a list of booleans to control the centering for each k-axis.
        k_endpoints : bool, list[bool], optional
            If True, include the endpoints of the k-space grid, giving
            the topology of a torus. One may also specify a list of booleans
            to control the inclusion of endpoints for each k-axis.
        lambda_endpoints : bool, list[bool], optional
            If True, include the endpoints of the lambda space grid.
            One may also specify a list of booleans to control the inclusion
            of endpoints for each lambda-axis.
        lambda_start : float, list[float], optional
            The starting point for the lambda space grid. If not specified,
            defaults to 0.0. One may also specify a list of floats to control
            the starting point for each lambda-axis.
        lambda_stop : float, list[float], optional
            The stopping point for the lambda space grid. If not specified,
            defaults to 1.0. One may also specify a list of floats to control
            the stopping point for each lambda-axis.

        Examples
        --------
        We can create a full grid by specifying the shape of the grid.

        >>> mesh = Mesh(dim_k=2, dim_lambda=0, axis_types=['k', 'k'])
        >>> mesh.build_full_grid(shape=(10, 10), gamma_centered=True)
        >>> mesh.grid.shape
        (10, 10, 2)

        Or suppose we have a 3D k-space model with an additional lambda dimension.

        >>> mesh = Mesh(dim_k=3, dim_lambda=1, axis_types=['k', 'k', 'k', 'l'])
        >>> mesh.build_full_grid(shape=(10, 10, 10, 100), gamma_centered=True)
        >>> mesh.grid.shape
        (10, 10, 10, 100, 4)

        The endpoints are included by default, and since we have a gamma-centered grid,
        the k-axes go from -0.5 to 0.5.

        >>> mesh.grid[0, 0, 0, 0, 0]
        array([-0.5, -0.5, -0.5,  0. ])
        >>> mesh.grid[-1, -1, -1, -1, -1]
        array([ 0.5,  0.5,  0.5,  1. ])
        """
        # Checks
        if not isinstance(shape, (tuple, list)):
            raise ValueError(f"Expected tuple or list for shape, got {type(shape)}")
        if len(shape) != self.num_k_axes + self.num_lambda_axes:
            raise ValueError(f"Expected {self.num_k_axes + self.num_lambda_axes} dimensions, got {len(shape)}")
        
        if isinstance(gamma_centered, bool):
            gamma_centered = [gamma_centered] * self.num_k_axes
        elif isinstance(gamma_centered, list) and len(gamma_centered) != self.num_k_axes:
            raise ValueError(f"Expected {self.num_k_axes} elements in gamma_centered, got {len(gamma_centered)}")
        else:
            raise ValueError("gamma_centered must be a bool or a list of bools.")
        
        if isinstance(k_endpoints, bool):
            k_endpoints = [k_endpoints] * self.num_k_axes
        elif isinstance(k_endpoints, list) and len(k_endpoints) != self.num_k_axes:
            raise ValueError(f"Expected {self.num_k_axes} elements in k_endpoints, got {len(k_endpoints)}")
        else:
            raise ValueError("k_endpoints must be a bool or a list of bools.")

        if isinstance(lambda_endpoints, bool):
            lambda_endpoints = [lambda_endpoints] * self.num_lambda_axes
        elif isinstance(lambda_endpoints, list) and len(lambda_endpoints) != self.num_lambda_axes:
            raise ValueError(f"Expected {self.num_lambda_axes} elements in lambda_endpoints, got {len(lambda_endpoints)}")
        else:
            raise ValueError("lambda_endpoints must be a bool or a list of bools.")

        if isinstance(lambda_start, (int, float, complex)):
            lambda_start = [lambda_start] * self.num_lambda_axes
        elif isinstance(lambda_start, list) and len(lambda_start) != self.num_lambda_axes:
            raise ValueError(f"Expected {self.num_lambda_axes} elements in lambda_start, got {len(lambda_start)}")
        else:
            raise ValueError("lambda_start must be a complex, int, float or a list of them.")

        if isinstance(lambda_stop, (int, float, complex)):
            lambda_stop = [lambda_stop] * self.num_lambda_axes
        elif isinstance(lambda_stop, list) and len(lambda_stop) != self.num_lambda_axes:
            raise ValueError(f"Expected {self.num_lambda_axes} elements in lambda_stop, got {len(lambda_stop)}")
        else:
            raise ValueError("lambda_stop must be a complex, int, float or a list of them.")

        shape_k = shape[:self.num_k_axes]
        shape_lambda = shape[self.num_k_axes:]

        self._shape_k = tuple(shape_k)
        self._shape_lambda = tuple(shape_lambda)

        self._gamma_centered = gamma_centered
        k_starts = []
        k_stops = []
        for i, g in enumerate(gamma_centered):
            if g:
                k_starts.append(-0.5)
                k_stops.append(0.5)
            else:
                k_starts.append(0)
                k_stops.append(1)

        if len(shape_lambda) == 0:
            flat = self.gen_hyper_cube(
                *shape_k,
                start=k_starts,
                stop=k_stops,
                flat=True,
                endpoint=k_endpoints
            )

        elif len(shape_k) == 0:
            flat = self.gen_hyper_cube(
                *shape_lambda,
                start=lambda_start,
                stop=lambda_stop,
                flat=True,
                endpoint=lambda_endpoints
            )
        else:
            # generate k-space grid
            k_flat = self.gen_hyper_cube(
                *shape_k,
                start=k_starts,
                stop=k_stops,
                flat=True,
                endpoint=k_endpoints
            )

            # generate parameter space grid
            p_flat = self.gen_hyper_cube(
                *shape_lambda,
                start=lambda_start,
                stop=lambda_stop,
                flat=True,
                endpoint=lambda_endpoints
            )

            Nk, Np = k_flat.shape[0], p_flat.shape[0]

            k_rep = np.repeat(k_flat, Np, axis=0)
            p_rep = np.tile(p_flat, (Nk, 1))
            flat = np.hstack([k_rep, p_rep])

        self._points = flat
        self._set_pbc_info()


    def build_custom(self, points):
        """Build a custom mesh from the given points.

        This method allows for the creation of a mesh with arbitrary points,
        rather than a regular grid. The shape of the input points array must
        match the axis types defined in the ``Mesh`` object. 

        Parameters
        ----------
        points : np.ndarray
            Array of shape ``(N1, N2, ..., Nd, dim_total)``, where
            `d` is the number of axes defined by ``axis_types`` and 
            `dim_total` is the total number of dimensions in the mesh defined
            by ``dim_total = dim_k + dim_lambda``.

        Examples
        --------
        Say we have a model with two k-space dimensions (e.g., kx and ky).
        We can then build a custom mesh using arbitrary points:

        >>> custom_points = np.random.rand(4, 2)  # 4 points in 2D
        >>> mesh = Mesh(dim_k=2, dim_lambda=0, axis_types=['k'])
        >>> mesh.build_custom(custom_points)

        Suppose instead our custom points were in a 'grid' shape. We would 
        then need to initialize the ``Mesh`` with two 'k' axis types.

        >>> grid_points = np.random.rand(10, 10, 2)  # 10x10 grid in 2D
        >>> mesh = Mesh(dim_k=2, dim_lambda=0, axis_types=['k', 'k'])
        >>> mesh.build_custom(grid_points)
        """
        self.is_custom = True

        if not isinstance(points, np.ndarray):
            raise ValueError("Mesh points must be a numpy array.")
        if points.ndim != len(self.shape_full):
            raise ValueError("Inconsistent dimensions between mesh points and axis types.")

        self._points = np.reshape(points, (-1, points.shape[-1]))

        shape = points.shape[:-1]
        self._shape_k = tuple(shape[i] for i in self.k_axes)
        self._shape_lambda = tuple(shape[i] for i in self.lambda_axes)
        self._set_pbc_info()


    def _extract_axis_range(
            self, 
            axis_index: int, 
            component_index: int, 
        ) -> np.ndarray:
        """
        Extract the unique 1D range along a mesh axis/component pair.
        Only for grids, not paths.
        """
        if not (self.is_grid or self.is_custom):
            raise ValueError("Axis range extraction only supported for grid meshes.")
        if self.points.size == 0:
            raise ValueError("Mesh points are not initialized.")
        shape = self.shape_full
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
        return arr

    def get_k_ranges(self) -> dict:
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
            arr = self._extract_axis_range(i, comp)
            result[name] = arr
        return result

    def get_param_ranges(self) -> dict:
        """
        Return a dict mapping parameter-axis names to their 1D ranges.
        """
        if self.num_lambda_axes == 0:
            raise ValueError("No parameter axes present in this mesh.")
        # Indices of parameter axes and their corresponding component indices
        param_axis_indices = list(range(self.num_k_axes, self.num_k_axes + self.num_lambda_axes))
        param_comp_indices = list(range(self.dim_k, self.dim_k + self.dim_lambda))
        param_names = [n for n, t in zip(self.axis_names, self.axis_types) if t == 'l']
        result = {}
        for i, comp, name in zip(param_axis_indices, param_comp_indices, param_names):
            arr = self._extract_axis_range(i, comp)
            result[name] = arr
        return result
    

    def get_k_points(self) -> np.ndarray:
        """
        Return the unique k-point mesh from the full grid, with shape (*shape_k, dim_k).
        """
        if self.points.size == 0:
            raise ValueError("Mesh points are not initialized.")
        G = self.grid  # shape (*shape_k, *shape_lambda, dim_k+dim_lambda)
        Gk = G[..., :self.dim_k]
        num_k_axes = self.num_k_axes
        num_lambda_axes = self.num_lambda_axes
        # Build index: [slice(None)]*num_k_axes + [0]*num_lambda_axes + [slice(None)]
        idx = [slice(None)]*num_k_axes + [0]*num_lambda_axes + [slice(None)]
        Gk_unique = Gk[tuple(idx)]
        # Ensure correct shape
        Gk_unique = np.asarray(Gk_unique)
        shape_k = self.shape_k
        if Gk_unique.shape != shape_k + (self.dim_k,):
            Gk_unique = Gk_unique.reshape(shape_k + (self.dim_k,))
        return Gk_unique

    def get_param_points(self) -> np.ndarray:
        """
        Return the unique parameter-point mesh from the full grid, with shape (*shape_lambda, dim_lambda).
        """
        if self.points.size == 0:
            raise ValueError("Mesh points are not initialized.")
        G = self.grid  # shape (*shape_k, *shape_lambda, dim_k+dim_lambda)
        Gp = G[..., self.dim_k:]
        num_k_axes = self.num_k_axes
        num_lambda_axes = self.num_lambda_axes
        # Build index: [0]*num_k_axes + [slice(None)]*num_lambda_axes + [slice(None)]
        idx = [0]*num_k_axes + [slice(None)]*num_lambda_axes + [slice(None)]
        Gp_unique = Gp[tuple(idx)]
        # Ensure correct shape
        shape_lambda = self.shape_lambda
        if Gp_unique.shape != shape_lambda + (self.dim_lambda,):
            Gp_unique = Gp_unique.reshape(shape_lambda + (self.dim_lambda,))
        return Gp_unique

    @staticmethod
    def gen_hyper_cube(
        *n_points, 
        start: int | float | list[int | float] = 0.0, 
        stop: int | float | list[int | float] = 1.0, 
        endpoint: bool | list[bool] = False,
        flat: bool = True
    ) -> np.ndarray:
        """Generate a hypercube of points in the specified dimensions.

        Parameters
        ----------
        *n_points: int
            Number of points along each dimension.
        start: float, optional
            Start value for the mesh grid. Defaults to 0.0.
        stop: float, optional
            Stop value for the mesh grid. Defaults to 1.0.
        flat: bool, optional
            If True, returns flattened array of k-points (e.g. of shape ``(n1*n2*n3 , 3)``).
            If False, returns reshaped array with axes along each k-space dimension
            (e.g. of shape ``(n1, n2, n3, 3)``). Defaults to True.
        endpoint: bool, optional
            If True, includes 1 (edge of BZ in reduced coordinates) in the mesh. Defaults to False.
            When Wannierizing should omit this point.

        Returns
        -------
        mesh: np.ndarray
            Array of coordinates defining the hypercube. 
        """
        if isinstance(start, list):
            if len(start) != len(n_points):
                raise ValueError(f"Expected {len(n_points)} elements in start, got {len(start)}")
        elif not isinstance(start, (int, float)):
            raise ValueError("start must be a complex, int, float or a list of them.")
        else:
            start = [start] * len(n_points)

        if isinstance(stop, list):
            if len(stop) != len(n_points):
                raise ValueError(f"Expected {len(n_points)} elements in stop, got {len(stop)}")
        elif not isinstance(stop, (int, float)):
            raise ValueError("stop must be a complex, int, float or a list of them.")
        else:
            stop = [stop] * len(n_points)

        if isinstance(endpoint, list):
            if len(endpoint) != len(n_points):
                raise ValueError(f"Expected {len(n_points)} elements in endpoint, got {len(endpoint)}")
        elif not isinstance(endpoint, (bool)):
            raise ValueError("endpoint must be a bool or a list of bools.")
        else:
            endpoint = [endpoint] * len(n_points)

        vals = [
            np.linspace(start[idx], stop[idx], n, endpoint=endpoint[idx])
            for idx, n in enumerate(n_points)
        ]
        flat_mesh = np.stack(np.meshgrid(*vals, indexing="ij"), axis=-1)

        return flat_mesh if not flat else flat_mesh.reshape(-1, len(vals))

    

    
