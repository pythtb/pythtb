from pythtb import TBModel, Lattice


def ssh(v, w):
    r"""Su-Schrieffer-Heeger (SSH) model.

    .. versionadded:: 2.0.0

    This function constructs the SSH model with the specified hopping parameters.
    The SSH model is a one-dimensional tight-binding model that describes a chain of atoms
    with alternating hopping parameters. The tight-binding Hamiltonian for the SSH model can be
    written as:

    .. math::
       H = v \sum_{i} (c_{i, 1}^{\dagger} c_{i, 2} + \text{h.c.}) + w \sum_{i} (c_{i, 2}^{\dagger} c_{i+1, 1} + \text{h.c.})


    Parameters
    ----------
    v : float
        The intracell hopping within the unit cell.
    w : float
        The intercell hopping to neighboring unit cells.

    Returns
    -------
    TBModel
        The tight-binding model for the SSH lattice.
    """
    lat_vecs = [[1]]
    orb_vecs = [[0], [1 / 2]]
    lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0])

    my_model = TBModel(lattice=lat, spinful=False)

    my_model.set_hop(v, 0, 1, [0])
    my_model.set_hop(w, 1, 0, [1])

    return my_model
