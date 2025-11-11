import numpy as np
from pythtb import TBModel, Lattice


def graphene(delta: float, t: float) -> TBModel:
    r"""Graphene tight-binding model.

    .. versionadded:: 2.0.0

    This function creates a graphene tight-binding model with the specified
    hopping parameters and on-site energy. The model is defined on a 2D honeycomb
    lattice with two sublattices. The lattice vectors are given by,

    .. math::

        \mathbf{a}_1 = a(1, 0), \quad \mathbf{a}_2 = a\left(\frac{1}{2}, \frac{\sqrt{3}}{2}\right),

    and the orbital positions are given by,

    .. math::

        \mathbf{\tau}_1 = \frac{1}{3} \mathbf{a}_1 + \frac{1}{3} \mathbf{a}_2,
        \quad \mathbf{\tau}_2 = \frac{2}{3} \mathbf{a}_1 + \frac{2}{3} \mathbf{a}_2

    The second-quantized Hamiltonian can be written as:

    .. math::

        H = \Delta \sum_i n_i + t \sum_{\langle i,j \rangle} (c_i^\dagger c_j + \text{h.c.})

    Parameters
    ----------
    delta : float
        On-site energy difference between the two orbitals.
    t : float
        Hopping parameter between nearest neighbor orbitals.

    Returns
    -------
    TBModel
        An instance of the model.
    """

    lat_vecs = [[1, 0], [1 / 2, np.sqrt(3) / 2]]
    orb_vecs = [[1 / 3, 1 / 3], [2 / 3, 2 / 3]]
    lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1])

    my_model = TBModel(lattice=lat, spinful=False)

    my_model.set_onsite([-delta, delta])
    my_model.set_hop(t, 0, 1, [0, 0])
    my_model.set_hop(t, 1, 0, [1, 0])
    my_model.set_hop(t, 1, 0, [0, 1])

    return my_model
