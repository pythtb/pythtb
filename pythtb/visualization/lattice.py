"""Module for visualizing lattice structures."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from itertools import product
from matplotlib import cm
from .utils import _proj


def plot_lattice(lattice, n_cells=1, proj_plane=None, orb_color="r", fig=None, ax=None):
    r"""Visualize the lattice geometry.

    Parameters
    ----------
    lattice : Lattice
        The lattice to visualize.
    n_cells : int, optional
        Number of unit cells to display in each direction. Default is 1.
    proj_plane : tuple or list of two integers, optional
        The projection plane onto which the 3D model is projected.
        This should be a 2-element array specifying the indices of the
        Cartesian coordinates to use for the x and y axes of the
        plot. For example, if ``proj_plane=(0,2)``, then the x-z plane is used.
        This parameter is only relevant for 3D lattices (i.e., when
        ``lattice.dim_r == 3``). If not specified, the default is
        to use the x-y plane.
    orb_color : str, optional
        Color to use for the orbitals. Default is "r" (red).
    fig : matplotlib.figure.Figure, optional
        Figure object to plot on. If not provided, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If not provided, new axes are created.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and axes objects for the plot.

    See Also
    --------
    :ref:`visualize-nb`
    :ref:`haldane-edge-nb`
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # to ensure proper padding, track all plotted coordinates
    all_coords = []

    # Draw the origin
    origin = [0.0, 0.0]
    ax.plot(origin[0], origin[1], "x", color="black", ms=6)
    all_coords.append(origin)

    # Draw lattice (unit cell) vectors as arrows and label them
    ends = []
    for i in lattice.periodic_dirs:
        start = origin
        end = _proj(lattice.lat_vecs[i], proj_plane=proj_plane)
        ends.append(end)
        all_coords.append(end)

        # lattice vector arrow
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=15,
            color="blue",
            lw=2,
            alpha=0.8,
            zorder=0,
        )
        ax.add_patch(arrow)

        # annotation of lattice
        ax.annotate(
            f"$\\vec{{a}}_{i}$",
            xy=end,  # (end[0], end[1])
            xytext=(4, 4),  # offset in points
            textcoords="offset points",
            color="blue",
            fontsize=12,
            va="bottom",
            ha="right",
        )

    # plot dotted bounding lines to unit cell
    ends = np.array(ends)
    all_coords += ends.tolist()

    # if 2d cell
    if ends.shape[0] > 1:
        # top shifted line
        start = ends[0]
        end = ends[0] + ends[1]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            ls="--",
            lw=1,
            color="b",
            zorder=0,
            alpha=0.5,
        )

        # right shifted line
        start = ends[1]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            ls="--",
            lw=1,
            color="b",
            zorder=0,
            alpha=0.5,
        )
        all_coords.append(end)

    # Draw orbitals: home-cell orbitals in red
    orb_coords = []
    orb_cart = lattice.get_orb_vecs(cartesian=True)
    supercell_range = range(-(n_cells - 1), n_cells)
    repeat = 2 if lattice.dim_r > 1 else 1
    for d in product(supercell_range, repeat=repeat):
        if lattice.dim_r == 1:
            dx = d[0]
            dy = 0
            translation = dx * lattice.lat_vecs[0]
        else:
            dx, dy = d
            translation = dx * lattice.lat_vecs[0] + dy * lattice.lat_vecs[1]

        for i in range(lattice.norb):
            pos = orb_cart[i] + translation
            p = _proj(pos, proj_plane=proj_plane)
            ax.scatter(
                p[0], p[1], color=orb_color, s=20, zorder=2, label=f"Orbital {i}"
            )
            orb_coords.append(p)

    # Adjust the axis so everything fits
    all_coords += orb_coords
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    # Add some padding
    pad_x = 0.1 * (max_x - min_x if max_x != min_x else 1)
    pad_y = 0.1 * (max_y - min_y if max_y != min_y else 1)
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)

    # Final plot adjustments
    ax.set_aspect("equal")
    if proj_plane is not None:
        ax.set_xlabel(rf"x_{proj_plane[0]}")
        ax.set_ylabel(f"x_{proj_plane[1]}")
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    return fig, ax


def plot_lattice_3d(
    lattice,
    n_cells=1,
    show_lattice_info=True,
    site_colors=None,
    site_names=None,
):
    """Visualize a 3D lattice using Plotly.

    This function creates an interactive 3D plot of your lattice,
    showing the unit-cell origin, lattice vectors (with arrowheads), and orbitals.

    Parameters
    ----------
    model : TBModel
        The tight-binding model to use for the calculation.
    annotate_onsite : bool, optional
        Whether to annotate orbitals with onsite energies.

    Returns
    -------
    fig : go.Figure
        A Plotly Figure object.
    """
    import plotly.graph_objects as go

    if lattice.dim_r != 3:
        raise ValueError("Lattice must be 3D to use this function.")

    # Container for all Plotly traces.
    traces = []
    all_coords = []

    # --- Draw Origin ---
    origin = np.array([0.0, 0.0, 0.0])

    all_coords.append(origin)

    # --- Draw Lattice Vectors ---
    # We assume lattice.periodic_dirs is an iterable of indices for lattice vectors.
    lattice_traces = []
    for i in lattice.periodic_dirs:
        start = origin
        end = np.array(lattice.lat_vecs[i])
        # Line for the lattice vector.
        lattice_traces.append(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                line=dict(color="blue", width=4),
                showlegend=False,
                hoverinfo="none",
            )
        )
        # Add a cone to simulate an arrowhead.
        direction = end - start
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction_unit = direction / norm
            lattice_traces.append(
                go.Cone(
                    x=[end[0]],
                    y=[end[1]],
                    z=[end[2]],
                    u=[direction_unit[0]],
                    v=[direction_unit[1]],
                    w=[direction_unit[2]],
                    anchor="tip",
                    sizemode="absolute",
                    sizeref=0.2,
                    showscale=False,
                    colorscale=[[0, "blue"], [1, "blue"]],
                    name=f"a{i}",
                )
            )
        # Add a text annotation (using a text scatter) at the end.
        lattice_traces.append(
            go.Scatter3d(
                x=[end[0]],
                y=[end[1]],
                z=[end[2]],
                mode="text",
                # text=[fr"$\vec{{a}}_{i}$"],
                text=[f"a{i}"],
                textposition="top center",
                textfont=dict(color="blue", size=12),
                showlegend=False,
                hoverinfo="none",
            )
        )
        all_coords.append(end)
    traces.extend(lattice_traces)

    # --- Draw Orbitals ---
    orb_x, orb_y, orb_z = [], [], []
    orb_text = []
    cmap_orb = cm.get_cmap("viridis", lattice.norb)
    orb_cart = lattice.get_orb_vecs(cartesian=True)
    supercell_range = range(-(n_cells - 1), n_cells)
    for i in range(lattice.norb):
        orb_pos = orb_cart[i].copy()
        color = site_colors[i] if site_colors is not None else cmap_orb(i)
        name = site_names[i] if site_names is not None else f"Orbital {i}"
        for dx, dy, dz in product(supercell_range, repeat=3):
            orb_text.append(f"Orbital {i}")
            translation = (
                dx * lattice.lat_vecs[0]
                + dy * lattice.lat_vecs[1]
                + dz * lattice.lat_vecs[2]
            )
            pos = orb_pos + translation
            orb_x.append(pos[0])
            orb_y.append(pos[1])
            orb_z.append(pos[2])
            all_coords.append(pos)

            traces.append(
                go.Scatter3d(
                    x=[pos[0]],
                    y=[pos[1]],
                    z=[pos[2]],
                    mode="markers",
                    marker=dict(color=color, size=10),
                    text=[rf"Orbital {i}"],
                    hoverinfo="text",
                    name=name,
                )
            )

    # --- Determine Axis Limits ---
    all_coords = np.array(all_coords)
    min_x, max_x = np.min(all_coords[:, 0]), np.max(all_coords[:, 0])
    min_y, max_y = np.min(all_coords[:, 1]), np.max(all_coords[:, 1])
    min_z, max_z = np.min(all_coords[:, 2]), np.max(all_coords[:, 2])
    pad_x = 0.1 * (max_x - min_x if max_x != min_x else 1)
    pad_y = 0.1 * (max_y - min_y if max_y != min_y else 1)
    pad_z = 0.1 * (max_z - min_z if max_z != min_z else 1)

    layout = go.Layout(
        scene=dict(
            xaxis=dict(range=[min_x - pad_x, max_x + pad_x], title="X"),
            yaxis=dict(range=[min_y - pad_y, max_y + pad_y], title="Y"),
            zaxis=dict(range=[min_z - pad_z, max_z + pad_z], title="Z"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    fig = go.Figure(data=traces, layout=layout)

    def get_pretty_model_info_str():
        lines = []
        lines.append("<b>Lattice Vectors:</b><br>")
        for i, vec in enumerate(lattice.lat_vecs):
            lines.append(
                f"a_{i} = {np.array2string(vec, precision=3, separator=', ')}<br>"
            )
        lines.append("<br>")
        lines.append("<b>Orbital Vectors:</b><br>")
        for i, orb in enumerate(lattice.orb_vecs):
            lines.append(
                f"Orbital {i} = {np.array2string(orb, precision=3, separator=', ')}<br>"
            )
        lines.append("<br>")
        return "".join(lines)

    report_text = get_pretty_model_info_str()

    if show_lattice_info:
        # 3) Add an annotation. We’ll place it in the upper-left corner (x=0.01, y=0.99).
        fig.add_annotation(
            text=report_text,
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.99,
            showarrow=False,
            align="left",
            font=dict(family="Courier New, monospace", size=12, color="black"),
            bordercolor="black",
            borderwidth=1,
            borderpad=5,
            bgcolor="white",
        )

    return fig
