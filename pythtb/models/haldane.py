from pythtb import TBModel
import numpy as np


def haldane(
        delta: float, 
        t1: float, 
        t2: float, 
        phi: float=np.pi/2
        ) -> TBModel:
    r"""Haldane tight-binding model.

    .. versionadded:: 2.0.0

    This function creates a Haldane tight-binding model with the specified
    hopping parameters and on-site energy. The model is defined on a 2D honeycomb
    lattice with two sublattices. The lattice vectors are given by,

    .. math::

        \mathbf{a}_1 = (1, 0), \quad \mathbf{a}_2 = \left(\frac{1}{2}, \frac{\sqrt{3}}{2}\right)

    and the orbital positions are given by,

    .. math::

        \mathbf{\tau}_1 = \frac{1}{3} \mathbf{a}_1 + \frac{1}{3} \mathbf{a}_2, 
        \quad \mathbf{\tau}_2 = \frac{2}{3} \mathbf{a}_1 + \frac{2}{3} \mathbf{a}_2

    The second-quantized Hamiltonian can be written as:

    .. math::

        H = \Delta \sum_i (-)^i c_i^\dagger c_i + t_1 \sum_{\langle i,j \rangle} (c_i^\dagger c_j 
        + \text{h.c.}) + t_2 \sum_{\langle\langle i,j \rangle\rangle} (ic_i^\dagger c_j + \text{h.c.})

    Parameters
    ----------
    delta : float
        Onsite mass term. Opposite sign for the two sublattices.
    t1 : float
        Nearest neighbor hopping amplitude.
    t2 : float
        Next-nearest neighbor hopping amplitude. Peierls phase is included.

    Returns
    -------
    TBModel
        An instance of the model.

    Notes
    -----
    The Haldane model describes a two-dimensional topological insulator with a
    non-trivial band structure. It is characterized by a finite Chern number
    and exhibits edge states that are protected by time-reversal symmetry [haldane]_. 

    References
    ----------
    .. [haldane] Haldane, F. D. M. (1988).
        O(3) Nonlinear :math:`\sigma` Model and the Quantum Hall Effect in Two Dimensions. 
        *Physical Review Letters*, 61(20), 2015–2018.
    """

    lat = [[1, 0], [1/2, np.sqrt(3) / 2]]
    orb = [[1/3, 1/3], [2/3, 2/3]]

    model = TBModel(2, 2, lat, orb)

    model.set_onsite([-delta, delta], mode="set")

    for lvec in ([0, 0], [-1, 0], [0, -1]):
        model.set_hop(t1, 0, 1, lvec, mode="set")

    for lvec in ([1, 0], [-1, 1], [0, -1]):
        model.set_hop(t2 * np.exp(1j*phi), 0, 0, lvec, mode="set")
    for lvec in ([-1, 0], [1, -1], [0, 1]):
        model.set_hop(t2 * np.exp(1j*phi), 1, 1, lvec, mode="set")

    return model
