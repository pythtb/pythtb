"""Module for visualization of Wannier functions in 2D."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_density(
    wan,
    wan_idx,
    mark_home_cell=False,
    mark_center=False,
    show_lattice=False,
    dens_size=40,
    lat_size=2,
    fig=None,
    ax=None,
    show=False,
    cbar=True,
):
    """Plot the Wannier function density on the lattice in 2D.

    Parameters
    ----------
    wan_idx : int
        Index of the Wannier function to plot.
    mark_home_cell : bool
        If True, mark the home cell in the plot.
    mark_center : bool
        If True, mark the center of the Wannier function in the plot.
    show_lattice : bool
        If True, show the lattice sites in the plot.
    dens_size : float
        Size of the density markers in the plot.
    lat_size : float
        Size of the lattice site markers in the plot.
    show : bool
        If True, display the plot immediately.
    fig : matplotlib.figure.Figure | None
        Matplotlib figure object to plot on. If None, a new figure is created.
    ax : matplotlib.axes.Axes | None
        Matplotlib axes object to plot on. If None, new axes are created.
    cbar : bool
        If True, include a colorbar in the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """

    center = wan.centers[wan_idx]
    positions = wan._get_sc_weights(wan_idx)

    # Extract arrays for plotting or further processing
    xs = positions["all"]["xs"]
    ys = positions["all"]["ys"]
    w0i_wt = positions["all"]["wt"]

    xs_home = positions["home"]["xs"]
    ys_home = positions["home"]["ys"]

    if fig is None:
        fig, ax = plt.subplots()

    # Weight plot
    dens_plot = ax.scatter(
        xs,
        ys,
        c=w0i_wt,
        s=dens_size,
        cmap="plasma",
        norm=LogNorm(vmin=2e-16, vmax=1),
        marker="h",
        zorder=0,
    )

    if show_lattice:
        ax.scatter(xs, ys, marker="o", c="k", s=lat_size, zorder=2)

    if mark_home_cell:
        ax.scatter(
            xs_home,
            ys_home,
            marker="o",
            s=lat_size,
            zorder=2,
            facecolors="none",
            edgecolors="b",
        )

    if mark_center:
        ax.scatter(
            center[0],
            center[1],
            marker="x",
            label=rf"Center $\mathbf{{r}}_c = ({center[0]: .3f}, {center[1]: .3f})$",
            c="g",
            zorder=1,
        )
        ax.legend(loc="upper right")

    if cbar:
        cbar = plt.colorbar(dens_plot, ax=ax)
        cbar.set_label(rf"$|w_{wan_idx}(\mathbf{{r}} )|^2$", rotation=270)
        cbar.ax.get_yaxis().labelpad = 20

    if show:
        plt.show()

    return fig, ax


def plot_decay(
    wan,
    wan_idx: int,
    fig=None,
    ax=None,
    show=False,
):
    """Plot the Wannier function density as a function of distance from center.

    Parameters
    ----------
    wan_idx : int
        Index of the Wannier function to plot.
    fig : matplotlib.figure.Figure | None
        Matplotlib figure object to plot on. If None, a new figure is created.
    ax : matplotlib.axes.Axes | None
        Matplotlib axes object to plot on. If None, new axes are created.
    show : bool
        If True, display the plot immediately.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """

    if fig is None:
        fig, ax = plt.subplots()

    # Extract arrays for plotting or further processing
    positions = wan._get_sc_weights(wan_idx)

    r = positions["all"]["r"]
    w0i_wt = positions["all"]["wt"]

    # binning data
    max_r = np.amax(r)
    num_bins = int(np.ceil(max_r))
    r_bins = [[i, i + 1] for i in range(num_bins)]
    r_ledge = [i for i in range(num_bins)]
    r_cntr = [0.5 + i for i in range(num_bins)]
    w0i_wt_bins = [[] for i in range(num_bins)]

    # bins of weights
    for i in range(r.shape[0]):
        for j, r_intvl in enumerate(r_bins):
            if r_intvl[0] <= r[i] < r_intvl[1]:
                w0i_wt_bins[j].append(w0i_wt[i])
                break

    # average value of bins
    avg_w0i_wt_bins = []
    for i in range(num_bins):
        if len(w0i_wt_bins[i]) != 0:
            avg_w0i_wt_bins.append(sum(w0i_wt_bins[i]) / len(w0i_wt_bins[i]))

    # numpify
    avg_w0i_wt_bins = np.array(avg_w0i_wt_bins)
    r_ledge = np.array(r_ledge)
    r_cntr = np.array(r_cntr)
    cutoff = int(0.7 * max_r)

    ax.scatter(r[r < cutoff], w0i_wt[r < cutoff], zorder=1, s=10, c="b")

    # bar of avgs
    ax.bar(
        r_ledge[r_ledge < cutoff],
        avg_w0i_wt_bins[r_ledge < cutoff],
        width=1,
        align="edge",
        ec="k",
        zorder=0,
        ls="-",
        alpha=0.3,
    )

    ax.set_xlabel(r"$|\mathbf{r} - \mathbf{{r}}_c|$", size=12)
    ax.set_ylabel(rf"$|w_{wan_idx}(\mathbf{{r}}- \mathbf{{r}}_c)|^2$", size=12)
    ax.set_ylim(0.8 * min(w0i_wt[r < cutoff]), 1.5)
    ax.set_yscale("log")

    if show:
        plt.show()

    return fig, ax


def plot_centers(
    wan,
    center_scale=15,
    section_home_cell=True,
    color_home_cell=True,
    translate_centers=False,
    show=False,
    legend=True,
    pmx=4,
    pmy=4,
    center_color="r",
    center_marker="*",
    lat_home_color="b",
    lat_color="k",
    fig=None,
    ax=None,
):
    """Plot the Wannier function centers in the supercell.

    Parameters
    ----------
    center_scale : float
        Scale factor for the size of the center markers. Scales with the spread
        of the Wannier functions, this is a multiplicative factor.
    section_home_cell : bool
        If True, delineate the home cell in the plot.
    color_home_cell : bool
        If True, color the home cell orbitals differently from other cells.
    translate_centers : bool
        If True, translate the home cell Wannier centers to neighboring cells.
    show : bool
        If True, display the plot immediately.
    legend : bool
        If True, include a legend in the plot.
    pmx : int
        Plus-minus range in x-direction for plotting supercell.
    pmy : int
        Plus-minus range in y-direction for plotting supercell.
    center_color : str
        Color for the Wannier center markers.
    center_marker : str
        Marker style for the Wannier center markers.
    lat_home_color : str
        Color for the home cell lattice sites.
    lat_color : str
        Color for the other lattice sites.
    fig : matplotlib.figure.Figure | None
        Matplotlib figure object to plot on. If None, a new figure is created.
    ax : matplotlib.axes.Axes | None
        Matplotlib axes object to plot on. If None, new axes are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """

    centers = wan.centers
    positions_wt = wan._get_sc_weights(0)
    positions_centers = wan._get_sc_centers()

    # All positions
    xs_orb = positions_wt["all"]["xs"]
    ys_orb = positions_wt["all"]["ys"]

    # Home cell site positions
    xs_orb_home = positions_wt["home"]["xs"]
    ys_orb_home = positions_wt["home"]["ys"]

    if fig is None:
        fig, ax = plt.subplots()

    # Draw lines sectioning out home supercell
    if section_home_cell:
        lat_vecs = wan.lattice.lat_vecs

        c1 = np.array([0, 0])
        c2 = c1 + lat_vecs[0]
        c3 = c1 + lat_vecs[1]
        c4 = c1 + lat_vecs[0] + lat_vecs[1]

        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], c="k", ls="--", lw=1)
        ax.plot([c1[0], c3[0]], [c1[1], c3[1]], c="k", ls="--", lw=1)
        ax.plot([c3[0], c4[0]], [c3[1], c4[1]], c="k", ls="--", lw=1)
        ax.plot([c2[0], c4[0]], [c2[1], c4[1]], c="k", ls="--", lw=1)

    if color_home_cell:
        # Zip the home cell coordinates into tuples
        home_coords = set(zip(xs_orb_home, ys_orb_home))

        # Keep (x, y) pairs that are not in home_coordinates
        out = [(x, y) for x, y in zip(xs_orb, ys_orb) if (x, y) not in home_coords]
        if out:
            xs_out, ys_out = zip(*out)
        else:
            xs_out, ys_out = [], []  # In case no points are left

        ax.scatter(
            xs_orb_home, ys_orb_home, zorder=1, s=20, marker="o", c=lat_home_color
        )
        ax.scatter(xs_out, ys_out, zorder=1, s=20, marker="o", c=lat_color)
    else:
        ax.scatter(xs_orb, ys_orb, zorder=1, s=20, marker="o", c=lat_color)

    # scatter centers
    for i in range(centers.shape[0]):
        if translate_centers:
            x = positions_centers["centers all"]["xs"][i]
            y = positions_centers["centers all"]["ys"][i]

            if i == 0:
                label = "Wannier centers"
            else:
                label = None

            ax.scatter(
                x,
                y,
                zorder=1,
                label=label,
                s=np.exp(11 * wan.spread[i]) * center_scale,
                marker=center_marker,
                c=center_color,
            )
        else:
            center = centers[i]
            if i == 0:
                label = "Wannier centers"
            else:
                label = None
            ax.scatter(
                center[0],
                center[1],
                zorder=2,
                c=center_color,
                alpha=0.5,
                s=np.exp(11 * wan.spread[i]) * center_scale,
                label=label,
                marker=center_marker,
            )

    if legend:
        ax.legend(loc="upper right")

    center_sc = (1 / 2) * (lat_vecs[0] + lat_vecs[1])
    ax.set_xlim(center_sc[0] - pmx, center_sc[0] + pmx)
    ax.set_ylim(center_sc[1] - pmy, center_sc[1] + pmy)

    if show:
        plt.show()

    return fig, ax
