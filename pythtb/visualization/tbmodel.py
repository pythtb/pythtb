"""Module for visualization of tight-binding models in 2D and 3D."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib import cm
from matplotlib import colors as mcolors
from .lattice import plot_lattice
from .utils import _proj
from .utils import _pauli_decompose_str
from .utils import _require_plotly


# TODO: Add hoverable hopping and onsite terms
def plot_tbmodel(
    model,
    proj_plane=None,
    eig_dr=None,
    draw_hoppings=True,
    annotate_onsite=False,
    ph_color="black",
    orb_color="red",
    show: bool | str = True,
):
    r"""Visualizes the tight-binding model geometry.

    Plots the tight-binding orbitals, hopping between tight-binding orbitals,
    and optionally the electron eigenstates.

    If eigenvector is not drawn, then orbitals in home cell are drawn
    as red circles, and those in neighboring cells are drawn with
    a lighter shade of red. Hopping term directions are drawn with
    green lines connecting two orbitals. Origin of unit cell is
    indicated with blue dot, while real space unit vectors are drawn
    with blue lines.

    If eigenvector is drawn, then electron eigenstate on each orbital
    is drawn with a circle whose size is proportional to wavefunction
    amplitude while its color depends on the phase. There are various
    coloring schemes for the phase factor; see more details under
    `ph_color` parameter. If eigenvector is drawn and coloring scheme
    is "red-blue" or "wheel", all other elements of the picture are
    drawn in gray or black.

    .. versionchanged:: 2.0.0
        Visualization appearance has been updated.

    Parameters
    ----------
    proj_plane : tuple or list of two integers
        Cartesian coordinates to be used for plotting. For example,
        if ``proj_plane=(0,1)`` then x-y projection of the model is
        drawn. This only should be specified if `dim_r` > 2.

        .. versionchanged:: 2.0.0
            Replaced previous parameters ``dir_first`` and ``dir_second``.

    eig_dr : Optional parameter specifying eigenstate to
        plot. If specified, this should be one-dimensional array of
        complex numbers specifying wavefunction at each orbital in
        the tight-binding basis. If not specified, eigenstate is not
        drawn.
    draw_hoppings : Optional parameter specifying whether to
        draw all allowed hopping terms in the tight-binding
        model. Default value is True.
    ph_color : {"black", "red-blue", "wheel"}, optional
        Determines the way the eigenvector phase factors are
        translated into color. Default value is "black".

        - "black" -- phase of eigenvectors are ignored and wavefunction
            is always colored in black.

        - "red-blue" -- zero phase is drawn red, while phases or :math:`\pi` or
            :math:`-\pi` are drawn blue. Phases in between are interpolated between
            red and blue. Some phase information is lost in this coloring
            because phase of :math:`\pm \pi` have the same color.

        - "wheel" -- each phase is given unique color. In steps of :math:`\pi/3`
            starting from 0, colors are assigned (in increasing hue) as:
            red, yellow, green, cyan, blue, magenta, red.

    Returns
    -------
        fig : matplotlib.figure.Figure
            Figure object from matplotlib.pyplot module
        ax : matplotlib.axes.Axes
            Axes object from matplotlib.pyplot module

    Notes
    -----
    - This function is intended for visualizing tight-binding models
        in two dimensions. For three-dimensional visualizations, consider using
        the :func:`visualize_3d` method.
    - Convention of the wavefunction phase is as
        in convention 1 in section 3.1 of :download:`notes on
        tight-binding formalism </_static/formalism/pythtb-formalism.pdf>`. In
        other words, these wavefunction phases are in correspondence
        with cell-periodic functions :math:`u_{n {\bf k}} ({\bf r})`
        not :math:`\Psi_{n {\bf k}} ({\bf r})`.

    Examples
    --------
    Draws x-y projection of tight-binding model
    tweaks figure and saves it as a PDF.

    >>> fig, ax = tb.visualize(0, 1)
    >>> plt.show()

    See Also
    --------
    :ref:`haldane-edge-nb`,
    :ref:`visualize-nb`.

    """

    # Draw orbitals: home-cell orbitals in red
    fig, ax = plot_lattice(model.lattice, proj_plane=proj_plane, orb_color=orb_color)
    cmap = plt.get_cmap("hsv", model.norb)

    # to ensure proper padding, track all plotted coordinates
    all_coords = []
    all_coords.append([0.0, 0.0])

    # append ends of lattice vectors to all_coords for proper padding
    ends = []
    for i in model.lattice.periodic_dirs:
        end = _proj(model.lattice.lat_vecs[i], proj_plane=proj_plane)
        ends.append(end)
        all_coords.append(end)

    ends = np.array(ends)
    all_coords += ends.tolist()

    # append dotted bounding lines to unit cell to all_coords for proper padding
    # if 2d cell
    if ends.shape[0] > 1:
        end = ends[0] + ends[1]
        all_coords.append(end)

    orb_coords = []
    orb_cart = model.get_orb_vecs(cartesian=True)
    for i in range(model.norb):
        pos = orb_cart[i].copy()
        p = _proj(pos, proj_plane=proj_plane)
        orb_coords.append(p)

        # For spinful case, annotate orbital with onsite decomposition.
        if model._nspin == 2 and annotate_onsite:
            onsite_str = _pauli_decompose_str(model._site_energies[i])
            ax.annotate(
                rf"$\Delta_{{{i}}} = {onsite_str}$",
                xy=p,
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                color="red",
                bbox=dict(
                    boxstyle="round,pad=0.2", fc="lightcoral", ec="none", alpha=0.6
                ),
                zorder=5,
            )
        elif model._nspin == 1 and annotate_onsite:
            onsite_str = rf"$\Delta_{{{i}}} = {model._site_energies[i]:.2f}$"
            ax.annotate(
                onsite_str,
                xy=p,
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                color="red",
                bbox=dict(
                    boxstyle="round,pad=0.2", fc="lightcoral", ec="none", alpha=0.6
                ),
                zorder=5,
            )

    # Draw hopping terms with curved arrows
    hopping_coords = []

    # maximum magnitudes of hopping strengths
    amps, hop_i, hop_j, hop_R = model._hoptable.components()
    n_hops = len(model._hoptable)
    if n_hops:
        if model._nspin == 2:
            mags = np.array([np.max(np.abs(amp)) for amp in amps], dtype=float)
        else:
            mags = np.abs(amps.astype(complex))
    else:
        mags = np.array([], dtype=float)

    biggest_hop = np.max(mags) if mags.size else 0.0
    arrow_alphas = mags / biggest_hop if biggest_hop else np.ones_like(mags)
    arrow_alphas = 0.3 + 0.7 * arrow_alphas**2

    if draw_hoppings:
        for h_idx in range(n_hops):
            i_orb = int(hop_i[h_idx])
            j_orb = int(hop_j[h_idx])

            r_vec = hop_R[h_idx] if model.dim_k != 0 else None
            intracell = True
            if r_vec is not None:
                intracell = np.all(r_vec == 0)

            for shift in range(2):  # draw both i->j+R and i-R->j hop
                pos_i = orb_cart[i_orb].copy()
                pos_j = orb_cart[j_orb].copy()

                # Determine starting and ending orbital positions
                if r_vec is not None:
                    # Adjust pos_j with lattice translation if provided
                    if shift == 0:
                        # i->j+R
                        pos_j += np.dot(r_vec, model.lat_vecs)
                    elif shift == 1:
                        # i-R->j
                        pos_i -= np.dot(r_vec, model.lat_vecs)

                p_i = _proj(pos_i, proj_plane=proj_plane)
                p_j = _proj(pos_j, proj_plane=proj_plane)

                # plot neighboring cell orbital
                # ensure we don't plot orbital in unit cell again (if no translation)
                if not intracell:
                    # ensure we only scatter orbitals once
                    if p_j not in hopping_coords and shift == 0:
                        ax.scatter(
                            p_j[0],
                            p_j[1],
                            color=orb_color,
                            s=20,
                            zorder=1,
                            alpha=0.5,
                        )
                    if p_i not in hopping_coords and shift == 1:
                        ax.scatter(
                            p_i[0],
                            p_i[1],
                            color=orb_color,
                            s=20,
                            zorder=1,
                            alpha=0.5,
                        )

                # Don't want to plot connecting arrows within unit cell twice
                if intracell and shift == 1:
                    # Arrow connects orbitals within cell, so shift = 1 is same
                    # conditions as shift = 2 (same arrows)
                    continue

                # First arrow: p_i -> p_j
                arrow1 = FancyArrowPatch(
                    p_i,
                    p_j,
                    connectionstyle="arc3,rad=0.08",
                    arrowstyle="->",
                    mutation_scale=15,
                    color="green",
                    lw=1.3,
                    alpha=arrow_alphas[h_idx],
                    zorder=1,
                )
                ax.add_patch(arrow1)

                # Second arrow: p_j -> p_i
                arrow2 = FancyArrowPatch(
                    p_j,
                    p_i,
                    connectionstyle="arc3,rad=0.08",
                    arrowstyle="->",
                    mutation_scale=15,
                    color="green",
                    lw=1.3,
                    alpha=arrow_alphas[h_idx],
                    zorder=1,
                )
                ax.add_patch(arrow2)

                hopping_coords.append(p_i)
                hopping_coords.append(p_j)

    # If eigenstate is provided, overlay eigenstate information on the orbitals
    if eig_dr is not None:
        # For each orbital, size the marker by amplitude and color by phase
        cmap = cm.hsv
        for i in range(model.norb):
            pos = orb_cart[i].copy()
            p = _proj(pos, proj_plane=proj_plane)
            amp = (
                np.abs(eig_dr[i]) ** 2
            )  # intensity proportional to probability density
            phase = np.angle(eig_dr[i])
            # Map phase from [-pi, pi] to [0,1]
            color = cmap((phase + np.pi) / (2 * np.pi))
            if ph_color == "black":
                color = "k"
            ax.scatter(
                p[0],
                p[1],
                s=30 * amp * 2 * model.norb,  # size proportional to amplitude
                color=color,
                edgecolor="k",
                zorder=10,
                alpha=0.7,
                label="Eigenstate" if i == 0 else None,
            )

    # Adjust the axis so everything fits
    all_coords += hopping_coords
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

    # ax.legend(loc="upper right", fontsize=10)
    # plt.tight_layout()

    return fig, ax


def plot_tbmodel_3d(
    model,
    draw_hoppings=True,
    show_model_info=True,
    site_colors=None,
    site_names=None,
    show=True,
):
    r"""Visualize a 3D tight-binding model using ``Plotly``.

    This function creates an interactive 3D plot of your tight-binding model,
    showing the unit-cell origin, lattice vectors (with arrowheads), orbitals,
    hopping lines, and (optionally) an eigenstate overlay with marker sizes
    proportional to amplitude and colors reflecting the phase.

    .. versionadded:: 2.0.0

    Parameters
    ----------
    draw_hoppings : bool, optional
        Whether to draw hopping lines between orbitals.
    show_model_info : bool, optional
        Whether to display model information as an annotation.
    site_colors : list of str, optional
        List of colors (e.g. hex strings) for each orbital. If not provided,
        a colormap is used.
    site_names : list of str, optional
        List of names for each orbital. If not provided, orbitals are named
        "Orbital 0", "Orbital 1", etc.
    show : bool or str, optional
        Whether to display the plot immediately.

    Returns
    -------
    plotly.graph_objs._figure.Figure, optional
        A Plotly Figure object. Returned only if `show` is False.
    """
    # Import Plotly here to avoid hard dependency if function is not used.
    go = _require_plotly()

    if model.dim_r != 3:
        raise ValueError("Model must be 3D to use this function.")

    # Container for all Plotly traces.
    traces = []
    all_coords = []

    # --- Draw Origin ---
    origin = np.array([0.0, 0.0, 0.0])

    all_coords.append(origin)

    # --- Draw Lattice Vectors ---
    lattice_traces = []
    for i in model.periodic_dirs:
        start = origin
        end = np.array(model.lat_vecs[i])
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
    orb_marker_colors = []
    onsite_labels = []
    cmap_orb = cm.get_cmap("viridis", model.norb)
    orb_cart = model.get_orb_vecs(cartesian=True)
    for i in range(model.norb):
        orb_text.append(f"Orbital {i}")

        if model._nspin == 2:
            onsite_str = _pauli_decompose_str(model._site_energies[i], use_unicode=True)
            onsite_label = rf"{onsite_str}"
        else:
            onsite_label = rf"{model._site_energies[i]:.2f}"
        onsite_labels.append(onsite_label)

        pos = orb_cart[i].copy()
        orb_x.append(pos[0])
        orb_y.append(pos[1])
        orb_z.append(pos[2])
        all_coords.append(pos)

        if site_colors is not None:
            # Use provided color for orbitals.
            orb_marker_colors.append(site_colors[i])

        else:
            # Convert RGBA to hex.
            orb_marker_colors.append(mcolors.to_hex(cmap_orb(i)))

        if site_names is not None:
            name = site_names[i]
        else:
            name = f"Orbital {i}"

        traces.append(
            go.Scatter3d(
                x=[pos[0]],
                y=[pos[1]],
                z=[pos[2]],
                mode="markers",
                marker=dict(color=orb_marker_colors[i], size=10),
                text=[rf"Orbital {i}, Onsite Energy = {onsite_label}"],
                hoverinfo="text",
                name=name,
            )
        )

    # Draw hopping terms ---
    if draw_hoppings:
        hopping_traces = []
        amps, hop_i, hop_j, hop_R = model._hoptable.components()
        n_hops = len(model._hoptable)
        if n_hops:
            if model._nspin == 2:
                mags = np.array([np.max(np.abs(amp)) for amp in amps], dtype=float)
            else:
                mags = np.abs(amps.astype(complex))
        else:
            mags = np.array([], dtype=float)
        biggest_hop = np.max(mags) if mags.size else 1.0
        arrow_alphas = mags / biggest_hop if biggest_hop else np.ones_like(mags)
        arrow_alphas = 0.3 + 0.7 * arrow_alphas**2  # Non-linear mapping.
        for h_idx in range(n_hops):
            amp = amps[h_idx]
            i_orb = int(hop_i[h_idx])
            j_orb = int(hop_j[h_idx])
            r_vec = hop_R[h_idx] if model.dim_k != 0 else None
            intracell = True

            if r_vec is not None:
                intracell = np.all(np.array(r_vec) == 0)

            # Draw hopping for both directions.
            for shift in range(2):
                pos_i = orb_cart[i_orb].copy()
                pos_j = orb_cart[j_orb].copy()
                if r_vec is not None:
                    if shift == 0:
                        pos_j = pos_j + np.dot(r_vec, model.lat_vecs)
                    elif shift == 1:
                        pos_i = pos_i - np.dot(r_vec, model.lat_vecs)

                if not intracell:
                    # ensure we only scatter orbitals once
                    traces.append(
                        go.Scatter3d(
                            x=[pos_i[0]],
                            y=[pos_i[1]],
                            z=[pos_i[2]],
                            mode="markers",
                            marker=dict(color=orb_marker_colors[i_orb], size=8),
                            name="",
                            showlegend=False,
                            text=[
                                f"Orbital {i_orb}, \n Onsite Energy: {onsite_labels[i_orb]}"
                            ],
                            hoverinfo="text",
                        )
                    )

                    traces.append(
                        go.Scatter3d(
                            x=[pos_j[0]],
                            y=[pos_j[1]],
                            z=[pos_j[2]],
                            mode="markers",
                            marker=dict(color=orb_marker_colors[j_orb], size=8),
                            name="",
                            showlegend=False,
                            text=[
                                rf"Orbital {j_orb}, onsite energy: {onsite_labels[j_orb]}"
                            ],
                            hoverinfo="text",
                        )
                    )

                # Don't want to plot connecting arrows within unit cell twice
                if intracell and shift == 1:
                    # Arrow connects orbitals to home cell, so shift = 1 is same
                    # conditions as shift = 2 (same arrows)
                    continue

                if model._nspin == 2:
                    amp_str = _pauli_decompose_str(amp, use_unicode=True)
                else:
                    amp_str = f"{amp:.2f}"

                if r_vec is not None:
                    r_vec_str = np.array2string(r_vec, precision=2, separator=", ")
                    r_vec_str = r_vec_str.replace("\n", "")
                    r_vec_str = r_vec_str.replace(" ", "")
                    hop_str = f"Hopping {i_orb} --> {j_orb} + {r_vec} = {amp_str}"
                else:
                    hop_str = f"Hopping {i_orb} --> {j_orb} = {amp_str}"

                hopping_traces.append(
                    go.Scatter3d(
                        x=[pos_i[0], pos_j[0]],
                        y=[pos_i[1], pos_j[1]],
                        z=[pos_i[2], pos_j[2]],
                        mode="lines",
                        line=dict(
                            color="green",
                            width=3,
                        ),
                        opacity=arrow_alphas[h_idx],
                        text=hop_str,
                        showlegend=False,
                        hoverinfo="text",
                    )
                )
                all_coords.append(pos_i)
                all_coords.append(pos_j)
        traces.extend(hopping_traces)

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
        lines.append("<b>Tight-Binding Model Information</b><br>")
        lines.append("<br>")
        lines.append("<b>Lattice Vectors:</b><br>")
        for i, vec in enumerate(model.lat_vecs):
            lines.append(
                f"a_{i} = {np.array2string(vec, precision=3, separator=', ')}<br>"
            )
        lines.append("<br>")
        lines.append("<b>Orbital Vectors:</b><br>")
        for i, orb in enumerate(model.orb_vecs):
            lines.append(
                f"Orbital {i} = {np.array2string(orb, precision=3, separator=', ')}<br>"
            )
        lines.append("<br>")
        lines.append(f"<b>Number of Spins:</b> {model._nspin}")
        return "".join(lines)

    report_text = get_pretty_model_info_str()

    if show_model_info:
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

    def _in_ipython() -> bool:
        try:
            from IPython import get_ipython  # type: ignore

            return get_ipython() is not None
        except Exception:
            return False

    def _display_plotly_html(fig) -> None:
        # Safe for myst-nb + Jupyter; keeps MathJax intact
        from IPython.display import HTML, display  # type: ignore

        html = fig.to_html(include_plotlyjs="cdn", full_html=False)
        display(HTML(html))

    if show is True:
        if _in_ipython():
            _display_plotly_html(fig)
        else:
            # Plain Python script: fall back to renderer’s show (may open a browser)
            fig.show()
    else:
        return fig


def plot_bands(
    model,
    k_path,
    nk=101,
    evals=None,
    evecs=None,
    ktick_labels=None,
    bands_label=None,
    proj_orb_idx=None,
    proj_spin=False,
    fig=None,
    ax=None,
    scat_size=3,
    lw=2,
    lc="b",
    ls="solid",
    cmap="plasma",
    cbar=True,
):
    """Plot the band structure along a specified path in k-space.

    This function allows for customization of the plot, including projection of orbitals,
    spin projection, figure and axis objects, title, scatter size, line width,
    line color, line style, colormap, and whether to show a color bar.

    .. versionadded:: 2.0.0

    Parameters
    ----------
    k_nodes : list[list[float]]
        List of high symmetry points (in reduced units) to plot the bands through.
        For example, ``[[0,0,0], [0, 1/2, 1/2]]``.
    k_node_labels : list[str], optional
        Labels of high symmetry points. Defaults to None.
    nk : int, optional
        Total number of k-points to sample along the path. Defaults to 101.
    proj_orb_idx : list[int], optional
        List of orbital indices to project onto. Defaults to None.
        This will give the bands a colorscale indicating the weight of
        the Bloch states onto the list of orbitals.
    proj_spin : bool, optional
        Whether to project the spin components. Defaults to ``False``.
        If ``True``, the bands will be colored according to their spin character.
    fig : matplotlib.figure.Figure, optional
        Figure object to plot on. Defaults to None.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. Defaults to None.
    scat_size : float, optional
        Size of the scatter points. Defaults to 3. Only relevant if
        `proj_spin` is True or `proj_orb_idx` is not None.
    lw : float, optional
        Line width of the band lines. Defaults to 2.
    lc : str, optional
        Line color of the band lines. Defaults to "b". Irrelevant
        if `proj_spin` is True or `proj_orb_idx` is not None.
    ls : str, optional
        Line style of the band lines. Defaults to "solid".
        Irrelevant if `proj_spin` is True or `proj_orb_idx` is not None.
    cmap : str, optional
        Colormap for the band plot. Defaults to "plasma". Only relevant if
        `proj_spin` is True or `proj_orb_idx` is not None.
    cbar : bool, optional
        Whether to show a color bar. Defaults to True.
        Only relevant if `proj_spin` is True or `proj_orb_idx` is not None.

    Returns:
        fig : matplotlib.figure.Figure
        ax: matplotlib.axes.Axes
    """

    if fig is None:
        fig, ax = plt.subplots()

    # generate k-path and labels
    (k_vec, k_dist, k_nodes) = model.k_path(k_path, nk, report=False)

    # scattered bands with sublattice color
    if proj_orb_idx is not None:
        if evals is None or evecs is None:
            # diagonalize model on path
            evals, evecs = model.solve_ham(k_vec, return_eigvecs=True)

        n_eigs = evals.shape[-1]
        wt = abs(evecs) ** 2

        if model._nspin == 1:
            col = np.sum([wt[..., i] for i in proj_orb_idx], axis=0)
        elif model._nspin == 2:
            wt = wt.reshape(wt.shape[0], wt.shape[1], model.nstate // 2, 2)
            col = np.sum([wt[..., i, :] for i in proj_orb_idx], axis=(0, -1))

        for n in range(n_eigs):
            label = bands_label if n == 0 else None
            scat = ax.scatter(
                k_dist,
                evals[:, n],
                c=col[:, n],
                cmap=cmap,
                marker="o",
                s=scat_size,
                vmin=0,
                vmax=1,
                zorder=2,
                label=label,
            )

        if cbar:
            cbar = fig.colorbar(scat, ticks=[1, 0], pad=0.01)
            cbar.ax.set_yticklabels([1, 0], size=12)
            cbar.ax.set_title(r"$ \sum_i |\langle \psi_{nk} | \phi_i \rangle |^2$")

    elif proj_spin:
        if evals is None or evecs is None:
            # diagonalize model on path
            evals, evecs = model.solve_ham(k_vec, return_eigvecs=True)

        n_eigs = evals.shape[-1]

        if model._nspin <= 1:
            raise ValueError("Spin needs to be greater than 1 for projecting spin.")

        wt = abs(evecs) ** 2
        col = np.sum(wt[..., 1], axis=2)

        for n in range(n_eigs):
            label = bands_label if n == 0 else None
            scat = ax.scatter(
                k_dist,
                evals[:, n],
                c=col[:, n],
                cmap=cmap,
                marker="o",
                s=scat_size,
                vmin=0,
                vmax=1,
                zorder=2,
                label=label,
            )

        cbar = fig.colorbar(scat, ticks=[1, 0])
        cbar.ax.set_yticklabels(
            [
                r"$ |\langle \psi_{nk} | \chi_{\uparrow} \rangle |^2$",
                r"$|\langle \psi_{nk} | \chi_{\downarrow} \rangle |^2$",
            ],
            size=12,
        )

    else:
        if evals is None:
            evals = model.solve_ham(k_vec, return_eigvecs=False)

        n_eigs = evals.shape[-1]

        for i, band in enumerate(evals.T):  # assuming evals shape is (nkpts, nbands)
            label = bands_label if i == 0 else None
            ax.plot(k_dist, band, c=lc, lw=lw, ls=ls, label=label)

        # continuous bands
        # ax.plot(k_dist, evals, c=lc, lw=lw, ls=ls, label=bands_label)

    if bands_label is not None:
        ax.legend(loc="upper right", fontsize=12)

    ax.set_xlim(k_nodes[0], k_nodes[-1])
    ax.set_xticks(k_nodes)
    for n in range(len(k_nodes)):
        ax.axvline(x=k_nodes[n], linewidth=0.5, color="k", zorder=1)
    if ktick_labels is not None:
        ax.set_xticklabels(ktick_labels, size=12)

    ax.set_ylabel(r"Energy $E(\mathbf{{k}})$", size=12)
    ax.yaxis.labelpad = 10

    return fig, ax
