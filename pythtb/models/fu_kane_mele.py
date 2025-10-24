import numpy as np
from pythtb import TBModel, Lattice


def fu_kane_mele(t, soc, m, beta):
    r"""Fu-Kane-Mele tight-binding model.

    .. versionadded:: 2.0.0

    This function creates a Haldane tight-binding model with the specified
    hopping parameters and on-site energy. The model is defined on a 2D honeycomb
    lattice with two sublattices. The lattice vectors are given by,

    .. math::

        \mathbf{a}_1 = (0, 1, 1), \quad \mathbf{a}_2 = (1, 0, 1),
        \quad \mathbf{a}_3 = (1, 1, 0)

    and the orbital positions are given by,

    .. math::

        \mathbf{\tau}_1 = (0, 0, 0), 
        \quad \mathbf{\tau}_2 = \frac{1}{4} \mathbf{a}_1 + \frac{1}{4} \mathbf{a}_2
        + \frac{1}{4} \mathbf{a}_3

    The second-quantized Hamiltonian can be written as:

    .. math::

        H = m \sin(\beta) \hat{e}_{111} \cdot (\sum_{i \in \text{A}} c_i^{\dagger} \vec{\sigma} c_i 
        - \sum_{i \in \text{B}} c_i^{\dagger} \vec{\sigma} c_i) 
        + \sum_{\langle i,j \rangle} t c_i^{\dagger} c_j 
        + i \lambda_{SO} \sum_{\langle\langle i,j \rangle\rangle} c_i^{\dagger} 
        \vec{\sigma} \cdot (\mathbf{d}_{ij}^{1} \times \mathbf{d}_{ij}^{2}) c_j

    where :math:`\hat{e}_{111}` is the unit vector along the [111] direction, 
    :math:`t_{ij} = 3t + m \cos(\beta)`
    for first-neighbor hopping along the [111] direction, and :math:`t_{ij} = t` otherwise. 
    The vectors 
    :math:`\mathbf{d}_{ij}^{1}` and :math:`\mathbf{d}_{ij}^{2}` are the two nearest-neighbor bond 
    vectors connecting 
    second-neighbor sites :math:`i` and :math:`j`.

    Parameters
    ----------
    t : float
        Spin-independent nearest-neighbor hopping amplitude.
    soc : float
        Spin-orbit coupling strength. Modulates next-nearest neighbor
        hopping amplitudes.
    m : float
        Magnetic field strength.
    beta : float
        Adiabatic parameter which controls the strength of the staggered magnetic field
        and the hopping amplitude along [111] direction.

    Returns
    -------
    TBModel
        An instance of the model.

    Notes
    -----
    The Fu-Kane-Mele model describes a three-dimensional topological insulator with a
    non-trivial band structure. It is characterized by a strong :math:`\mathbb{Z}_2` invariant
    and exhibits surface states that are protected by time-reversal symmetry [fu-kane-mele]_.
        
    See Also
    --------
    [fu-kane-mele]_, [Essin-Moore-Vanderbilt]_

    References
    ----------
    .. [fu-kane-mele] Fu, C. L. Kane, and E. J. Mele, Phys. Rev. Lett. 98, 106803
       (2007).
    .. [Essin-Moore-Vanderbilt] Essin, A. M. Moore, and D. Vanderbilt,
       Phys. Rev. Lett. 102, 146805 (2009).
    """

    lat_vecs = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    orb_vecs = [[0, 0, 0], [0.25, 0.25, 0.25]]
    lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1, 2])

    model = TBModel(lattice=lat, spinful=True)

    h = m * np.sin(beta) * np.array([1, 1, 1])
    dt = m * np.cos(beta)

    h0 = [0] + list(h)
    h1 = [0] + list(-h)

    model.set_onsite(h0, 0)
    model.set_onsite(h1, 1)

    # spin-independent first-neighbor hops
    for lvec in ([-1, 0, 0], [0, -1, 0], [0, 0, -1]):
        model.set_hop(t, 0, 1, lvec)

    model.set_hop(3 * t + dt, 0, 1, [0, 0, 0], mode="add")

    # spin-dependent second-neighbor hops
    lvec_list = ([1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 1, 0], [0, -1, 1], [1, 0, -1])
    dir_list = ([0, 1, -1], [-1, 0, 1], [1, -1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1])
    for j in range(6):
        spin = np.array([0.0] + dir_list[j])
        model.set_hop(1j * soc * spin, 0, 0, lvec_list[j])
        model.set_hop(-1j * soc * spin, 1, 1, lvec_list[j])

    return model
