from pythtb import TBModel, Lattice


def checkerboard(delta, t) -> TBModel:
    r"""Checkerboard tight-binding model.

    .. versionadded:: 2.0.0

    This function creates a checkerboard tight-binding model with the specified
    hopping parameters and on-site energy. The model is defined on a 2D square
    lattice with two sublattices. The lattice vectors are given by,

    .. math::

        \mathbf{a}_1 = (1, 0), \quad \mathbf{a}_2 = (0, 1)

    and the orbital positions are given by,

    .. math::

        \mathbf{\tau}_1 = \left(0, 0\right), \quad \mathbf{\tau}_2 = \left(\frac{1}{2}, \frac{1}{2}\right)

    The second-quantized Hamiltonian can be written as:

    .. math::

        H = t \sum_{\langle i,j \rangle} (c_i^\dagger c_j + \text{h.c.}) + \Delta \sum_i n_i

    Parameters
    ----------
    t : float
        Nearest neighbor hopping amplitude.

    delta : float
        On-site energy. Positive for one sublattice, negative for the other.

    Returns
    -------
    TBModel
        An instance of the model.
    """

    lat_vecs = [[1, 0], [0, 1]]
    orb_vecs = [[0, 0], [1 / 2, 1 / 2]]
    lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1])

    model = TBModel(lattice=lat, spinful=False)

    # set on-site energies
    model.set_onsite([-delta, delta], mode="set")

    model.set_hop(t, 1, 0, [0, 0])
    model.set_hop(t, 1, 0, [1, 0])
    model.set_hop(t, 1, 0, [0, 1])
    model.set_hop(t, 1, 0, [1, 1])

    return model
