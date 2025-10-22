import numpy as np
from typing import Optional

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


class Axis:
    def __init__(
            self, 
            axis_type: str, 
            name: str = None, 
            ):
        r"""Class representing a single axis in k/parameter space.

        Parameters
        ----------
        axis_type : str
            The type of the axis, either ``"k"`` for k-space or ``"l"`` for parameter space.
        name : str, optional
            The name of the axis. If not provided, a default name will be assigned.

        Notes
        -----
        This class is primarily used internally by the `Mesh` class to manage
        individual axes in the mesh.
        """
        if axis_type not in ['k', 'l']:
            raise TypeError("Axis type must be either 'k' or 'l'.")
        
        self._type = axis_type
        self._name = name if name is not None else f"{axis_type}_axis"
        self._size = 0

        self._loop_comps = []
        self._endpt_comps = []
        self._wind_bz_comps = []

        self._is_path = False

    @property
    def size(self) -> int:
        """The size of the axis."""
        return self._size
    @size.setter
    def size(self, value: int) -> None:
        if value < 0:
            raise ValueError("Axis size must be non-negative.")
        if not isinstance(value, int):
            raise TypeError("Axis size must be an integer.")
        self._size = value

    @property
    def name(self) -> str:
        """The name of the axis."""
        return self._name
    @name.setter
    def name(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("Axis name must be a string.")
        self._name = value

    @property
    def type(self) -> str:
        """The type of the axis, either 'k' or 'l'."""
        return self._type
    
    @property
    def is_k_axis(self) -> bool:
        """True if the axis is a k-axis."""
        return self._type == 'k'
    @property
    def is_lambda_axis(self) -> bool:
        """True if the axis is a lambda-axis."""
        return self._type == 'l'

    # loop
    @property
    def is_loop(self) -> bool:
        """True if the axis is a loop (i.e., winds around)."""
        return False if len(self._loop_comps) == 0 else True
    @property
    def loop_components(self) -> Optional[list[int]]:
        """The component index that this axis winds, or None if not a loop."""
        return self._loop_comps
    # allow appending loop components
    def add_loop_component(self, comp_idx: int) -> None:
        """Add a component index that this axis winds."""
        if self._loop_comps is None:
            self._loop_comps = []
        if comp_idx not in self._loop_comps:
            self._loop_comps.append(comp_idx)
    def remove_loop_component(self, comp_idx: int) -> None:
        """Remove a component index that this axis winds."""
        if self._loop_comps is None:
            return
        if comp_idx in self._loop_comps:
            self._loop_comps.remove(comp_idx)

    # endpoints
    @property
    def has_endpoint(self) -> bool:
        """True if the axis has an endpoint (i.e., first and last points are equal)."""
        return False if len(self._endpt_comps) == 0 else True
    @property
    def endpoint_components(self) -> Optional[list[int]]:
        """The component index that this axis has equal endpoints, or None if not."""
        return self._endpt_comps
    # allow appending endpoint components
    def add_endpoint_component(self, comp_idx: int) -> None:
        """Add a component index that this axis has equal endpoints."""
        if self._endpt_comps is None:
            self._endpt_comps = []
        if comp_idx not in self._endpt_comps:
            self._endpt_comps.append(comp_idx)
    def remove_endpoint_component(self, comp_idx: int) -> None:
        """Remove a component index that this axis has equal endpoints."""
        if self._endpt_comps is None:
            return
        if comp_idx in self._endpt_comps:
            self._endpt_comps.remove(comp_idx)

    # BZ winding
    @property
    def winds_bz(self) -> bool:
        """True if the axis winds the Brillouin zone."""
        return False if len(self._wind_bz_comps) == 0 else True
    @property
    def winds_bz_components(self) -> Optional[list[int]]:
        """The k-component index that this axis winds the Brillouin zone, or None if not."""
        return self._wind_bz_comps
    # allow appending BZ winding components
    def add_wind_bz_component(self, comp_idx: int) -> None:
        """Add a k-component index that this axis winds the Brillouin zone."""
        if self._wind_bz_comps is None:
            self._wind_bz_comps = []
        if comp_idx not in self._wind_bz_comps:
            self._wind_bz_comps.append(comp_idx)
    def remove_wind_bz_component(self, comp_idx: int) -> None:
        """Remove a k-component index that this axis winds the Brillouin zone."""
        if self._wind_bz_comps is None:
            return
        if comp_idx in self._wind_bz_comps:
            self._wind_bz_comps.remove(comp_idx)

    def __str__(self) -> str:
        return f"Axis(type={self.type}, name={self.name}, size={self.size})"

    def __repr__(self) -> str:
        return self.__str__()
    

class Mesh:
    r"""Initialize a Mesh object storing and managing a mesh in :math:`(k, \lambda)`-space.

    .. versionadded:: 2.0.0

    This class is responsible for constructing a mesh sampling of the combined reciprocal space and 
    additional adiabatic parameters, i.e. :math:`(k, \lambda)`-space.
    It provides methods to build both grid and path representations of the mesh, or a custom mesh
    with user-defined points. The mesh can be a pure k-space mesh, a pure parameter space mesh,
    or a mixed mesh with axes in both spaces. The mesh can also be a grid or a path. 

    A grid mesh has an axis for each dimension of the mesh, while a path mesh has a single axis
    that traces a path through the combined :math:`(k, \lambda)`-space. For example, a 2D k-space grid has two k-axes,
    while a 1D path mesh would have a single axis that moves through k-space. A mesh has axes that represent 
    the dimensions of the mesh, and the last axis represents the vector components in :math:`(k, \lambda)`-space. 
    A grid is defined such that each mesh dimension (axis) corresponds to one vector component, while a path has a 
    single axis that may vary multiple vector components simultaneously.

    Parameters
    ----------
    dim_k : int
        The dimensionality of k-space. This will influence the dimension of
        the vector at each mesh point.
    axis_types : list[str]
        A list of axis types, which can be ``"k"`` or ``"l"`` for k-space and parametric
        space respectively. The length of this list will determine the number of 
        dimensions in the mesh.
    dim_lambda : int, optional
        The dimensionality of parameter space. If left unspecified, the
        parameter space dimension will be automatically inferred from the axis types.
        If creating a path through a higher-dimensional parameter space, this must be
        specified.
    axis_names : list[str], optional
        A list of axis names, which can be used for parametrically populating
        a :class:`pythtb.WFArray`. If unspecified, default names will be generated.
        See examples listed below for more details.

    See Also
    --------
    :ref:`haldane-bp-nb`
    :ref:`kane-mele-nb`
    :ref:`three-site-thouless-nb`
    :ref:`cubic-slab-hwf-nb`
    :ref:`haldane-hwf-nb`

    Notes
    -----
    .. warning::
        The parameter space is assumed to be orthogonal to the k-space. This means that when varying the parameter
        along its axis, the k-components are held fixed. Paths through a mixed parameter and k-space are not 
        currently supported.
    """
    def __init__(
        self,  
        dim_k: int, 
        axis_types: list[str],
        dim_lambda: int | None = None,
        axis_names: list[str] = None
    ):

        
        self._dim_k = dim_k

        # Naming axes
        nk_ax = sum(1 for at in axis_types if at == 'k')
        nl_ax = sum(1 for at in axis_types if at == 'l')
        if axis_names is None:
            axis_names = [f"k_{i}" for i in range(nk_ax)] + [f"l_{i}" for i in range(nl_ax)]
        elif len(axis_types) != len(axis_names):
            raise ValueError("Axis types and axis names must have the same length.")
        
        # Initialize axes
        self._axes = [Axis(at, name) for at, name in zip(axis_types, axis_names)]

        if self.num_k_axes > dim_k:
            raise ValueError(f"Number of k axes ({self.num_k_axes}) cannot exceed specified dimension ({dim_k}).")

        # Dimension of parameter space
        if dim_lambda is None:
            self._dim_lambda = self.num_lambda_axes
        else:
            self._dim_lambda = dim_lambda

        # Define component types for the last coordinate axis: first dim_k are 'k', then parameters
        self._component_types = tuple(['k'] * self.dim_k + ['l'] * self.dim_lambda)

        self._points = np.empty((0,) + (self.dim_k + self.dim_lambda,), dtype=float)

        # for paths
        self._nodes = None

    @property
    def points(self):
        r"""Mesh point array of shape ``(N1*N2*...*Nd, dim_k + dim_lambda)``."""
        return self._points

    @property
    def flat(self):
        r"""Mesh point array of shape ``(N1*N2*...*Nd, dim_k + dim_lambda)``.

        Alias for `points` property.
        """
        return self._points
    
    @property
    def grid(self):
        r"""Mesh point array of shape ``(N1, ..., Nd, dim_k + dim_lambda)``."""
        return self._points.reshape(*self.shape_full)

    @property
    def nodes(self):
        r"""For path meshes, the original nodes used to build the path."""
        return self._nodes 
    
    @property
    def filled(self):
        """True if the mesh is filled (i.e., contains points)."""
        return not self.flat.size == 0

    # ---- Axis properties ----
    @property
    def axes(self) -> list[Axis]:
        """List of Axis objects defining the mesh axes."""
        return self._axes
    
    @property
    def k_axes(self) -> list[Axis]:
        """Tuple of k-axes."""
        return [ax for ax in self.axes if ax.type == 'k']

    @property
    def lambda_axes(self) -> list[Axis]:
        """Tuple of lambda-axes."""
        return [ax for ax in self.axes if ax.type == 'l']
    
    @property
    def k_axis_indices(self) -> list[int]:
        """List of indices of k-axes."""
        return [i for i, ax in enumerate(self.axes) if ax.type == 'k']
    
    @property
    def lambda_axis_indices(self) -> list[int]:
        """List of indices of lambda-axes."""
        return [i for i, ax in enumerate(self.axes) if ax.type == 'l']

    @property
    def axis_names(self):
        """Tuple of axis names."""
        axis_names = [ax.name for ax in self.axes]
        return axis_names

    @property
    def axis_types(self) -> tuple[str]:
        """Tuple of axis types."""
        axis_types = [ax.type for ax in self.axes]
        return axis_types

    @property
    def axis_sizes(self) -> list[int]:
        """Tuple of axis sizes."""
        return tuple([ax.size for ax in self.axes])

    @property
    def shape_k(self) -> tuple[int]:
        """Mesh shape defining the layout of k-axis points."""
        shape_k = tuple([ax.size for ax in self.axes if ax.type == 'k'])
        return shape_k

    @property
    def shape_lambda(self) -> tuple[int]:
        """Mesh shape defining the layout of parameter-axis points."""
        shape_lambda = tuple([ax.size for ax in self._axes if ax.type == 'l'])
        return shape_lambda
    
    @property
    def num_points(self) -> int:
        """Number of mesh points."""
        return int(np.prod(self.axis_sizes))

    @property
    def shape_full(self) -> tuple[int]:
        r"""Shape of full grid ``(*shape_k, *shape_lambda, dim_k + dim_lambda)``."""
        return self.axis_sizes + (self.dim_k + self.dim_lambda,)

    @property
    def shape_mesh(self) -> tuple[int]:
        r"""Shape of mesh grid ``(*shape_k, *shape_lambda)``."""
        return self.axis_sizes

    @property
    def shape_flat(self) -> tuple[int]:
        """Shape of flattened mesh points (N_points, dim_total)."""
        return (self.num_points, self.dim_total)

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
    def lambda_component_indices(self) -> list[int]:
        """List of indices of lambda components of the vector."""
        return list(range(self.dim_k, self.dim_total))
    
    @property
    def k_component_indices(self) -> list[int]:
        """List of indices of k components of the vector."""
        return list(range(self.dim_k))

    # ---- Topology properties ----

    # loop
    @property
    def loop_axes(self) -> list[Axis]:
        """List of Axis objects that wind to form a loop."""
        return [ax for ax in self.axes if ax.is_loop]
    @property
    def loop_mask(self) -> np.ndarray:
        """Boolean array of shape (num_axes, dim_total) marking which axes wind to form a loop."""
        loop_mask = np.zeros((self.num_axes, self.dim_total), dtype=bool)
        for i, ax in enumerate(self.axes):
            for c in ax.loop_components:
                loop_mask[i, c] = True
        return loop_mask
    def _get_loop_ax_comp(self) -> list[tuple[int, int]]:
        """List of (mesh_axis, component_index) pairs that wind to form a loop."""
        if not self.filled:
            return []
        
        mat = self.loop_mask  # (n_axes, dim_total)
        loop_axes = []
        for axis_idx in range(mat.shape[0]):
            for comp_idx in range(self.dim_total):
                if mat[axis_idx, comp_idx]:
                    loop_axes.append((axis_idx, comp_idx))
        return loop_axes
    
    # endpoints
    @property
    def endpoint_axes(self) -> list[Axis]:
        """List of Axis objects that have equal endpoints."""
        return [ax for ax in self.axes if ax.has_endpoint]
    @property
    def endpoint_mask(self) -> np.ndarray:
        """Boolean array of shape (num_axes, dim_total) marking which axes have equal endpoints."""
        endpt_mask = np.zeros((self.num_axes, self.dim_total), dtype=bool)
        for i, ax in enumerate(self.axes):
            for c in ax.endpoint_components:
                endpt_mask[i, c] = True
        return endpt_mask
    def _get_endpt_ax_comp(self) -> list[tuple[int, int]]:
        """List of (mesh_axis, component_index) pairs that wrap by ~1."""
        if not self.filled:
            return []

        mat = self.endpoint_mask
        endpt_axes = []
        for axis_idx in range(mat.shape[0]):
            for comp_idx in range(self.dim_total):
                if mat[axis_idx, comp_idx]:
                    endpt_axes.append((axis_idx, comp_idx))
        return endpt_axes

    # BZ winding
    @property
    def bz_winding_axes(self) -> list[Axis]:
        """List of Axis objects that wind around the BZ to form a loop."""
        return [ax for ax in self.axes if ax.winds_bz and ax.is_k_axis]
    @property
    def bz_winding_mask(self) -> np.ndarray:
        """Boolean array of shape (num_axes, dim_total) marking which axes wind around the BZ."""
        winds_bz_mask = np.zeros((self.num_axes, self.dim_total), dtype=bool)
        for i, ax in enumerate(self.axes):
            for c in ax.winds_bz_components:
                winds_bz_mask[i, c] = True
        return winds_bz_mask
    def _get_bz_wind_ax_comp(self) -> list[tuple[int, int]]:
        """List of (mesh_axis, component_index) pairs that wind around the BZ."""
        if not self.filled:
            return []

        mat = self.bz_winding_mask
        bz_wind_axes = []
        for axis_idx in range(mat.shape[0]):
            for comp_idx in range(self.dim_total):
                if mat[axis_idx, comp_idx]:
                    bz_wind_axes.append((axis_idx, comp_idx))
        return bz_wind_axes
    
    @property
    def is_k_torus(self) -> bool:
        r"""Does the mesh wind around the BZ in all k-directions?

        A torus mesh has an axis for each k-dimension and winds around the BZ 
        in each k-direction. This means the mesh is

        .. math::
            T^d = S^1 \times S^1 \times ... \times S^1

        where :math:`d` is the dimension of k-space.
        
        Notes
        -----
        - This only considers the k-space axes/dimensions.
          Non-periodic lambda axes will not affect the periodicity of the k-axes.
        - If the mesh is not a grid, this will always return False.
        - If the number of k-axes is less than dim_k, this will return False.
        - If the number of k-axes is equal to dim_k but not all k-axes wind around the BZ,
          this will return False.
        """
        if not self.is_grid:
            return False

        k_axes = self.k_axes
        bz_winding_axes = self.bz_winding_axes

        if len(bz_winding_axes) != self.dim_k:
            return False

        # Check if all k_axes are in the bz_winding_axes
        for k_ax in k_axes:
            if k_ax not in bz_winding_axes:
                return False
        return True
    

    def info(self, show: bool = True) -> str:
        """Information summary about the mesh.
        
        Returns
        -------
        str
            Information summary of the mesh.
        """
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
            
        # Full grid (optional flag some versions have)
        is_k_torus = getattr(self, "is_k_torus", None)

        # Closed axes 
        closed_axes = self._get_endpt_ax_comp()
        if closed_axes:
            ca_str = ", ".join(f"(axis {a} contains endpoint of component {c})" for a, c in closed_axes)
        else:
            ca_str = "None"
        # Looped axes
        loop_axes = self._get_loop_ax_comp()
        if loop_axes:
            la_str = ", ".join(f"(axis {a} loops component {c})" for a, c in loop_axes)
        else:
            la_str = "None"
        # Winding axes
        winding_axes = self._get_bz_wind_ax_comp()
        if winding_axes:
            wa_str = ", ".join(f"(axis {a} winds component {c})" for a, c in winding_axes)
        else:
            wa_str = "None"

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
        # lines.append(f"k-shape: {_fmt_tuple(shape_k)}")
        # lines.append(f"λ-shape: {_fmt_tuple(shape_p)}")
        lines.append(f"k-axes: {_fmt_list(k_axes)}")
        lines.append(f"λ-axes: {_fmt_list(p_axes)}")
        # lines.append(f"Axis names: {_fmt_list(axis_names)}")
        # Optional full-grid flag
        if is_k_torus is not None and mesh_type != "path":
            lines.append(f"Is a torus in k-space (all k-axes wind BZ): {_yn(is_k_torus)}")
        lines.append(f"Looped axes: {la_str}")
        lines.append(f"BZ-winding axes: {wa_str}")
        lines.append(f"Endpoint axes: {ca_str}")

        if show:
            print("\n".join(lines))
        else:
            return "\n".join(lines)

    def __str__(self) -> str:
        # Pretty, multi-line view for print(mesh)
        return self.info(show=False)

    def _set_ax_info(self, tol: float = 1e-8) -> np.ndarray:
        r"""
        Determine per-axis/component topology purely from mesh points. 

        This will mark axes as looping, winding the BZ, or containing endpoints for each component
        based on the mesh points. This is done by comparing the first and last points along each axis.
        This will only check k-components for BZ winding. If a k-axis winds a k-component by 1, it
        is marked as winding the BZ for that component. If the first and last points along an axis are
        equal (within tolerance), it is marked as containing endpoints for that component. Looping axes
        are those that either wind the BZ or contain endpoints.

        Parameters
        ----------
        tol : float
            Tolerance for detecting a wrap by 1. Use ~1e-8 for double.

        Notes
        -----
        - This will overwrite any previously set topology information. 
        - If a k-axis does not contain the edges of the BZ (ki=1) then it will not be detected as
          winding the BZ. This is up to the user to mark the axis as winding (for custom meshes or paths)
          using the `wind_bz` method. When using `build_grid` this will be set automatically.
        """
        if not self.filled:
            raise ValueError("Mesh points are not initialized.")

        # k-mask to only check k-components if needed
        k_comp_mask = np.zeros(self.dim_total, dtype=bool)
        k_comp_mask[:self.dim_k] = True

        # Iterate over sampling axes; compare first vs last hyperfaces.
        for axis_idx in range(self.num_axes):
            closed_vec  = np.zeros(self.dim_total, dtype=bool)
            winding_vec = np.zeros(self.dim_total, dtype=bool)
            looped_vec  = np.zeros(self.dim_total, dtype=bool)
            for c in range(self.dim_total):
                arr = self.get_axis_range(axis_idx, c)
                arr = np.ravel(arr)

                if arr.size == 0:
                    # no points along this axis -> skip
                    continue
                if np.ptp(arr) <= tol:
                    # constant along this axis -> skip
                    continue

                delta = float(abs(arr[-1] - arr[0])) # difference between first and last point

                winds_bz = abs(delta - 1.0) < tol and k_comp_mask[c]  # delta = 1 and is k-component
                eq0 = abs(delta) < tol # delta = 0 

                # TODO: maybe check if next point would be outside BZ for k-axis?
                winding_vec[c] = bool(winds_bz)  # delta = 1 -> winding k
                closed_vec[c]  = bool(winds_bz or eq0)  # delta = 1 or 0 -> closed (includes endpoints)
                looped_vec[c]  = bool(winds_bz or eq0)  # delta = 1 or 0 -> looped 
                
                # Update axes
                ax = self.axes[axis_idx]
                if winding_vec[c]:
                    ax.add_wind_bz_component(c)
                if closed_vec[c]:
                    ax.add_endpoint_component(c)
                if looped_vec[c]:
                    ax.add_loop_component(c)

    # ---- Topology configuration (explicit) ----
    def wind_bz(self, axis_idx: int, component_idx: int):
        r"""Declare an axis winds around the Brillouin zone for a specified k-component.

        Calling this function will mark an axis as winding a given k-component of the vector 
        in :math:`(\mathbf{k}, \lambda)`-space. This means that
        the two ends of the axis are identified, and sampling along ``axis_idx``
        winds ``component_idx`` around the Brillouin zone.

        Notes
        ------
        Winding the BZ implies the axis is also looped. This does not necessarily imply
        the axis contains endpoints (first and last points equal), as the axis may wind
        around the BZ without having equal endpoints.

        If the mesh is built using :meth:`build_grid`, :meth:`build_custom`,
        or :meth:`build_custom`, this will be set automatically
        for k-axes that span the BZ (i.e., go from 0 to 1 in that k-component).

        This allows ``WFArray`` to decide whether phases apply at the edge
        of the mesh (axis is closed) or just beyond the edge of the mesh (axis is not closed). 
        This will apply when ``axis_idx`` is a k-axis and ``component_idx`` is a k-component.

        See Also
        --------
        :ref:`mesh-nb`

        Examples
        --------
        A situation where this may be useful is when creating a path
        through k-space that winds around the BZ but does not have equal endpoints.
        In this case, the user can manually mark the axis as winding the BZ for the
        relevant k-component.
        >>> from pythtb import Mesh
        >>> mesh = Mesh(dim_k=2, axis_types=['k'])
        >>> points = np.linspace([0,0], [1, 1], 10, endpoint=False)  # path from (0,0) to (1, 1)
        >>> mesh.build_custom(points)
        >>> mesh.wind_bz(axis_idx=0, component_idx=0)  # mark as winding k_x
        >>> mesh.wind_bz(axis_idx=0, component_idx=1)  # mark as winding k_y
        >>> print(mesh)
        """
        if axis_idx < 0 or axis_idx >= self.num_axes:
            raise IndexError(f"axis_idx {axis_idx} out of bounds for {self.num_axes} axes")
        if component_idx < 0 or component_idx >= self.dim_total:
            raise IndexError(f"component_idx {component_idx} out of bounds for {self.dim_total} components")
        if component_idx >= self.dim_k:
            raise ValueError(f"component_idx {component_idx} is not a k-component (dim_k={self.dim_k})")
        
        ax = self.axes[axis_idx]
        if not ax.is_k_axis:
            raise ValueError(f"axis_idx {axis_idx} is not a k-axis (type={ax.type})")
        
        if component_idx not in ax.winds_bz_components:
            ax.add_wind_bz_component(component_idx)
            ax.add_loop_component(component_idx)  # winding implies looped

    def unwind_bz(self, axis_idx: int, component_idx: int):
        r"""Declare an axis does not wind around the Brillouin zone for a specified k-component.

        Calling this function will unmark an axis as winding a given k-component of the vector 
        in :math:`(\mathbf{k}, \lambda)`-space. 

        See Also
        --------
        :meth:`wind_bz`
        """
        if axis_idx < 0 or axis_idx >= self.num_axes:
            raise IndexError(f"axis_idx {axis_idx} out of bounds for {self.num_axes} axes")
        if component_idx < 0 or component_idx >= self.dim_total:
            raise IndexError(f"component_idx {component_idx} out of bounds for {self.dim_total} components")
        if component_idx >= self.dim_k:
            raise ValueError(f"component_idx {component_idx} is not a k-component (dim_k={self.dim_k})")
        
        ax = self.axes[axis_idx]
        if not ax.is_k_axis:
            raise ValueError(f"axis_idx {axis_idx} is not a k-axis (type={ax.type})")
        
        if component_idx in ax.winds_bz_components:
            ax.remove_wind_bz_component(component_idx)
            # NOTE: do not remove from loop_components, as it may still be looped by endpoints

    def loop_axis(self, axis_idx: int, component_idx: int):
        r"""Declare an axis winds a specified component of the mesh vector.

        Calling this function will mark an axis as looping a given component of the vector 
        in :math:`(\mathbf{k}, \lambda)`-space. This means that
        the two ends of the axis are identified, and sampling along 
        ``axis_idx`` loops ``component_idx`` around a cycle that 
        brings the Hamiltonian back into itself.

        Notes
        ------
        This will not mark the axis as winding the BZ, use :meth:`wind_bz` instead. 
        This means the `WFArray` will not apply phases at the edge of the mesh unless
        the axis is also marked as winding the BZ.
        """
        if axis_idx < 0 or axis_idx >= self.num_axes:
            raise IndexError(f"axis_idx {axis_idx} out of bounds for {self.num_axes} axes")
        if component_idx < 0 or component_idx >= self.dim_total:
            raise IndexError(f"component_idx {component_idx} out of bounds for {self.dim_total} components")
        ax = self.axes[axis_idx]
        if component_idx not in ax.loop_components:
            ax.add_loop_component(component_idx)

    def unloop_axis(self, axis_idx: int, component_idx: int):
        r"""Declare an axis winds a specified component of the mesh vector.

        Calling this function will mark an axis as winding a given component of the vector 
        in :math:`(\mathbf{k}, \lambda)`-space. This means that
        the two ends of the axis are identified, and sampling along ``axis_idx`` winds ``component_idx``
        around a cycle that brings the Hamiltonian back into itself.
         
        Notes
        ------
        This allows ``WFArray`` to decide whether phases apply to k-components at the edge
        of the mesh (axis is closed) or just beyond the edge of the mesh (axis is not closed). 
        This will apply when ``axis_idx`` is a k-axis and ``component_idx`` is a k-component.
        """
        if axis_idx < 0 or axis_idx >= self.num_axes:
            raise IndexError(f"axis_idx {axis_idx} out of bounds for {self.num_axes} axes")
        if component_idx < 0 or component_idx >= self.dim_total:
            raise IndexError(f"component_idx {component_idx} out of bounds for {self.dim_total} components")
        ax = self.axes[axis_idx]
        if component_idx in ax.loop_components:
            ax.remove_loop_component(component_idx)

    def close_axis(self, axis_idx: int, component_idx: int):
        r"""Declare an axis as closed for a given component.

        Calling this function will mark an axis as being closed for a given
        component of the vector in :math:`(\mathbf{k}, \lambda)`-space. 
        This means that the two ends of the axis correspond to the same Hamiltonian, and
        the eigenstates at the two ends should be considered identical with equal phase.

        ... note::
            Closing an axis will also mark it as looped for the given component.
            k-axes are marked as closed automatically when the difference between
            the first and last points is 0 or 1 (in reduced units).

        Parameters
        ----------
        axis_idx : int
            The index of the axis to close. Must be a lambda axis.
        component_idx : int
            The component of the vector to close. Must be a lambda component.

        Notes
        -----
        ... warning::
            It is up to the user to ensure that the Hamiltonian is indeed the same
            at the two ends of the axis. This function does not check this condition.
            If this is not the case, the results may be incorrect.
          

        Examples
        ---------
        Say we have a model with a single k-axis and a single parameteric variable. 
        This variable varies from 0 to :math:`2\pi` and brings the Hamiltonian back
        to itself. Here, we include :math:`2\pi` in the mesh.

        >>> mesh = Mesh(dim_k=1, dim_lambda=1, axis_types=["k", "l"])
        >>> mesh.build_grid(lambda_endpoints=True, lambda_start=0, lambda_end=2*np.pi)
        
        To indicate that the endpoint of this adiabatic loop is included in the mesh
        we set the `1` axis (lambda) as closed for the `1` component (lambda).

        >>> mesh.close_axis(1, 1)

        There may be instances where the axis and component differ. For example,
        if we have a two-dimensional k-space, but only build a mesh in 
        """
        if axis_idx < 0 or axis_idx >= self.num_axes:
            raise IndexError(f"axis_idx {axis_idx} out of bounds for {self.num_axes} axes")
        if component_idx < 0 or component_idx >= self.dim_total:
            raise IndexError(f"component_idx {component_idx} out of bounds for {self.dim_total} components")
        
        # self._endpt_mask[axis_idx, component_idx] = True
        ax = self.axes[axis_idx]
        if component_idx not in ax.endpoint_components:
            ax.add_endpoint_component(component_idx)

    def open_axis(self, axis_idx: int, component_idx: int):
        r"""Declare an axis as open for a given component.

        Calling this function will mark an axis as being open (not closed) for a given
        component of the vector in :math:`(\mathbf{k}, \lambda)`-space. 
        This means that the two ends of the axis correspond to different Hamiltonians, and
        the eigenstates at the two ends should be considered distinct.

        Parameters
        ----------
        axis_idx : int
            The index of the axis to open. Must be a lambda axis.
        component_idx : int
            The component of the vector to open. Must be a lambda component.

        Examples
        ---------
        Say we have a model with a single k-axis and a single parameteric variable. 
        This variable varies from 0 to :math:`2\pi` and brings the Hamiltonian back
        to itself. Here, we include :math:`2\pi` in the mesh.

        >>> mesh = Mesh(dim_k=1, dim_lambda=1, axis_types=["k", "l"])
        >>> mesh.build_grid(lambda_endpoints=True, lambda_start=0, lambda_end=2*np.pi)
        
        To indicate that the endpoint of this adiabatic loop is included in the mesh
        we set the `1` axis (lambda) as closed for the `1` component (lambda).

        >>> mesh.close_axis(1, 1)

        If we later decide that we want to treat this axis as open, we can call
        `open_axis` to remove this designation.

        >>> mesh.open_axis(1, 1)

        There may be instances where the axis and component differ. For example,
        if we have a two-dimensional k-space, but only build a mesh in 
        """
        if axis_idx < 0 or axis_idx >= self.num_axes:
            raise IndexError(f"axis_idx {axis_idx} out of bounds for {self.num_axes} axes")
        if component_idx < 0 or component_idx >= self.dim_total:
            raise IndexError(f"component_idx {component_idx} out of bounds for {self.dim_total} components")
        
        ax = self.axes[axis_idx]
        if component_idx in ax.endpoint_components:
            ax.remove_endpoint_component(component_idx)

    def is_axis_closed(self, axis_idx: int, comp: int = 'any') -> bool:
        """Return True iff sampling axis *axis_idx* contains endpoint for at least one component."""
        if axis_idx < 0 or axis_idx >= self.num_axes:
            raise IndexError(f"axis_idx {axis_idx} out of bounds for {self.num_axes} axes")
        
        comp_type = type(comp)

        if comp_type not in [int, str] or comp_type is str and comp.lower() != 'any':
            raise TypeError("comp must be an integer or 'any'")
        if comp_type is int and abs(comp) >= self.dim_total:
            raise IndexError(f"component_idx {comp} out of bounds for {self.dim_total} components")
        
        if comp_type is str and comp.lower() == 'any':
            return bool(np.any(self.endpoint_mask[axis_idx, :]))
        else:
            return bool(self.endpoint_mask[axis_idx, comp])

    def is_axis_looped(self, axis_idx: int, comp: int = 'any') -> bool:
        """Return True iff sampling axis *axis_idx* wraps at least one component."""
        if axis_idx < 0 or axis_idx >= self.num_axes:
            raise IndexError(f"axis_idx {axis_idx} out of bounds for {self.num_axes} axes")
        comp_type = type(comp)

        if comp_type not in [int, str] or comp_type is str and comp.lower() != 'any':
            raise TypeError("comp must be an integer or 'any'")
        if comp_type is int and abs(comp) >= self.dim_total:
            raise IndexError(f"component_idx {comp} out of bounds for {self.dim_total} components")
        
        if comp_type is str and comp.lower() == 'any':
            return bool(np.any(self.loop_mask[axis_idx, :]))
        else:
            return bool(self.loop_mask[axis_idx, comp])
        
    def is_axis_bz_winding(self, axis_idx: int, comp: int = 'any') -> bool:
        """Return True iff sampling axis *axis_idx* winds around the BZ for at least one component."""
        if axis_idx < 0 or axis_idx >= self.num_axes:
            raise IndexError(f"axis_idx {axis_idx} out of bounds for {self.num_axes} axes")
        comp_type = type(comp)

        if comp_type not in [int, str] or comp_type is str and comp.lower() != 'any':
            raise TypeError("comp must be an integer or 'any'")
        if comp_type is int and abs(comp) >= self.dim_total:
            raise IndexError(f"component_idx {comp} out of bounds for {self.dim_total} components")
        
        if comp_type is str and comp.lower() == 'any':
            return bool(np.any(self.bz_winding_mask[axis_idx, :]))
        else:
            return bool(self.bz_winding_mask[axis_idx, comp])


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

        >>> mesh.flat.shape
        (10, 3)
        """
        if self.num_k_axes + self.num_lambda_axes != 1:
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

        self.axes[0].size = path.shape[0]

        self._set_ax_info()

    def build_grid(self,
        shape: tuple | list,
        gamma_centered: bool | list = False,
        lambda_endpoints: bool | list = True,
        lambda_start: int | float | list = 0.0,
        lambda_stop: int | float | list = 1.0
    ):
        r"""Build a regular Monkhorst-Pack k-space and lambda space grid.

        The grid is a uniform array that has an axis for each dimension 
        in the combined k-space and lambda space (Monkhorst-Pack mesh).

        The k-points (in reduced units) range from [0, 1) along the k-axes, 
        unless gamma_centered is True, in which case they range from [-0.5, 0.5)
        along the k-axes. The last points (1 or 0.5) are excluded for internal 
        consistency. 

        The lambda points range from ``lambda_start`` to ``lambda_stop`` 
        along the lambda axes. If these are not specified, they will default
        to 0 and 1 respectively. The endpoints are included if ``lambda_endpoints``
        flag is set to ``True`` (default is ``True``).

        This function populates the ``.grid`` and ``.flat`` attributes.
        After calling this function, the ``.grid`` attribute will be:

        - shape ``(*shape, dim_k+dim_lambda)``,
       
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

        Notes
        -----
        This function should be used to create a regular grid in k-space and parameter space.
        It is not suitable for creating paths or irregular meshes. An example of when not to
        use it is if you have a 2D k-space model and are using a mesh of values along 
        :math:`k_y` for a given :math:`k_x` value, or vice versa. In such cases,
        you should use the :func:`build_path` method instead. This only is used when
        the number of axes match the number of k-dimensions.

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

        Since we have a gamma-centered grid, the k-axes go from [-0.5, 0.5) non-inclusive.
        The endpoints for the lambda axis are included by default.

        >>> mesh.grid[0, 0, 0, 0, 0]
        array([-0.5, -0.5, -0.5,  0. ])
        >>> mesh.grid[-1, -1, -1, -1, -1]
        array([ 0.49,  0.49,  0.49,  1. ])
        """
        # Checks
        if not isinstance(shape, (tuple, list)):
            raise TypeError(f"Expected tuple or list for shape, got {type(shape)}")

        if len(shape) != self.num_k_axes + self.num_lambda_axes:
            raise ValueError(f"Expected {self.num_k_axes + self.num_lambda_axes} dimensions, got {len(shape)}")
        
        if isinstance(gamma_centered, bool):
            gamma_centered = [gamma_centered] * self.num_k_axes
        elif isinstance(gamma_centered, list) and len(gamma_centered) != self.num_k_axes:
            raise ValueError(f"Expected {self.num_k_axes} elements in gamma_centered, got {len(gamma_centered)}")
        else:
            raise TypeError("gamma_centered must be a bool or a list of bools.")

        if isinstance(lambda_endpoints, bool):
            lambda_endpoints = [lambda_endpoints] * self.num_lambda_axes
        elif isinstance(lambda_endpoints, list) and len(lambda_endpoints) != self.num_lambda_axes:
            raise ValueError(f"Expected {self.num_lambda_axes} elements in lambda_endpoints, got {len(lambda_endpoints)}")
        else:
            raise TypeError("lambda_endpoints must be a bool or a list of bools.")

        if isinstance(lambda_start, (int, float, complex)):
            lambda_start = [lambda_start] * self.num_lambda_axes
        elif isinstance(lambda_start, list) and len(lambda_start) != self.num_lambda_axes:
            raise ValueError(f"Expected {self.num_lambda_axes} elements in lambda_start, got {len(lambda_start)}")
        else:
            raise TypeError("lambda_start must be a complex, int, float or a list of them.")

        if isinstance(lambda_stop, (int, float, complex)):
            lambda_stop = [lambda_stop] * self.num_lambda_axes
        elif isinstance(lambda_stop, list) and len(lambda_stop) != self.num_lambda_axes:
            raise ValueError(f"Expected {self.num_lambda_axes} elements in lambda_stop, got {len(lambda_stop)}")
        else:
            raise TypeError("lambda_stop must be a complex, int, float or a list of them.")

        # convert shape to ints
        shape = tuple(int(x) for x in shape) 
        shape_k = shape[:self.num_k_axes]
        shape_lambda = shape[self.num_k_axes:]

        # set axes shape
        for i, ax in enumerate(self.axes):
            ax.size = shape[i]
        
        self._gamma_centered = gamma_centered
        k_endpoints = [False] * self.num_k_axes

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
        for ax_idx in self.k_axis_indices:
            # Default mapping: k-axis winds the same-index k-component
            self.wind_bz(ax_idx, ax_idx)

        self._set_ax_info()


    def build_custom(self, points):
        r"""Build a custom mesh from the given points.

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

        >>> custom_points = np.random.rand(10, 10, 2)  # 2D mesh in 2D k-space
        >>> mesh = Mesh(dim_k=2, dim_lambda=0, axis_types=['k', 'k'])
        >>> mesh.build_custom(custom_points)

        Suppose instead we have a custom path through k-space that is not a regular grid. 
        We would then need to initialize the ``Mesh`` with a single 'k' axis type.

        >>> path_points = np.random.rand(100, 2)  # 100 point path in 2D k-space
        >>> mesh = Mesh(dim_k=2, dim_lambda=0, axis_types=['k'])
        >>> mesh.build_custom(path_points)
        """
        self.is_custom = True

        if not isinstance(points, np.ndarray):
            raise ValueError("Mesh points must be a numpy array.")
        if points.ndim != len(self.shape_full):
            raise ValueError("Inconsistent dimensions between mesh points and axis types.")
        
        # Set axis sizes
        for i, ax in enumerate(self.axes):
            ax.size = points.shape[i]

        self._points = np.reshape(points, (-1, points.shape[-1]))

        # shape = points.shape[:-1]
        # self._shape_k = tuple(shape[i] for i in self.k_axes)
        # self._shape_lambda = tuple(shape[i] for i in self.lambda_axes)
        self._set_ax_info()

    
    def get_axis_range(self, axis_index: int, component_index: int) -> np.ndarray:
        """
        Return the 1D range along a mesh axis/component pair.

        Parameters
        ----------
        axis_index : int
            The index of the axis to extract the range from.
        component_index : int
            The index of the component to extract the range for.

        Returns
        -------
        np.ndarray
            The 1D array of values along the specified axis/component.
        """
        if not self.filled:
            raise ValueError("Mesh points are not initialized.")
        if axis_index < 0 or axis_index >= self.num_axes:
            raise IndexError(f"axis_index {axis_index} out of bounds for mesh with {self.num_axes} axes.")
        if component_index < 0 or component_index >= self.dim_total:
            raise IndexError(f"component_index {component_index} out of bounds for {self.dim_total} components.")
        
        idx = [0] * self.num_axes
        idx[axis_index] = slice(None)
        idx = tuple(idx)
        arr = self.grid[idx + (component_index,)]
        arr = np.asarray(arr)
        # arr should be 1D
        if arr.ndim != 1:
            arr = np.reshape(arr, -1)
        return arr
            
    def get_k_points(self) -> np.ndarray:
        """
        Return the k-point mesh from the full grid, with shape ``(nk1, nk2, ..., dim_k)``.

        Notes
        -----
        The k-mesh is orthogonal to the lambda mesh, so this function returns the unique
        k-points in the mesh. For example, if the full mesh has shape 
        ``(nk1, nk2, nl1, dim_k+dim_lambda)``, this function will return the k-points
        with shape ``(nk1, nk2, dim_k)``.
        """
        if not self.filled:
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
        Return the unique parameter-point mesh from the full grid, with shape 
        ``(nl1, nl2, ..., dim_lambda)``.

        Notes
        -----
        The lambda-mesh is orthogonal to the k-mesh, so this function returns the unique
        lambda-points in the mesh. For example, if the full mesh has shape 
        ``(nk1, nk2, nl1, dim_k+dim_lambda)``, this function will return the lambda-points
        with shape ``(nl1, dim_lambda)``.
        """
        if not self.filled:
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
        start: float | list[float] = 0.0, 
        stop: float | list[float] = 1.0, 
        endpoint: bool | list[bool] = False,
        flat: bool = True
    ) -> np.ndarray:
        """Generate a hypercube of points in the specified dimensions.

        A hypercube is a generalization of a cube to arbitrary dimensions.
        Each dimension is orthogonal to the others, and the points are evenly spaced
        along each dimension. This function generates a grid of points in a hypercube
        defined by the number of points along each dimension, as well as the start and stop
        values for each dimension. The points are from ``start`` to ``stop``
        along each dimension, with the option to include or exclude the endpoint.

        Parameters
        ----------
        *n_points: int
            Number of points along each dimension.
        start: float, list[float], optional
            Start value for the mesh grid. May also be a list of start values for each dimension. Defaults to 0.0.
        stop: float, list[float], optional
            Stop value for the mesh grid. May also be a list of stop values for each dimension. Defaults to 1.0.
        endpoint: bool, list[bool], optional
            If True, includes 1 (edge of BZ in reduced coordinates) in the mesh. May also be a list of endpoint values 
            for each dimension. Defaults to False.
        flat: bool, optional
            If True, returns flattened array of k-points (e.g. of shape ``(n1*n2*n3 , 3)``).
            If False, returns reshaped array with axes along each k-space dimension
            (e.g. of shape ``(n1, n2, n3, 3)``). Defaults to True.

        Notes
        -----

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
            raise TypeError("endpoint must be a bool or a list of bools.")
        else:
            endpoint = [endpoint] * len(n_points)

        vals = [
            np.linspace(start[idx], stop[idx], n, endpoint=endpoint[idx])
            for idx, n in enumerate(n_points)
        ]
        flat_mesh = np.stack(np.meshgrid(*vals, indexing="ij"), axis=-1)

        return flat_mesh if not flat else flat_mesh.reshape(-1, len(vals))