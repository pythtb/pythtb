import numpy as np
from pythtb import TBModel, Lattice


def fu_kane_mele(t, soc, dt=[0, 0, 0, 0]):
    r"""Fu-Kane-Mele tight-binding model.

    This function creates a Fu-Kane-Mele tight-binding model on a diamond
    lattice. The lattice vectors are given by,

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

        H = t \sum_{\langle ij \rangle} c_i^{\dagger} c_j
        + i \lambda_{SO} \sum_{\langle\langle ij \rangle\rangle} c_i^{\dagger}
        \vec{\sigma} \cdot (\mathbf{d}_{ij}^{1} \times \mathbf{d}_{ij}^{2}) c_j

    where the first term is a nearest-neighbor hopping term connecting the two fcc sublattices
    of the diamond lattice, and the second term is a spin-orbit coupling term connecting
    second-neighbor sites within the same sublattice. Here, :math:`\mathbf{d}_{ij}^{1,2}`
    are the two nearest-neighbor bond vectors connecting sites :math:`i` and :math:`j`.

    Due to inversion symmetry, each band is doubly degenerate. The degeneracy is lifted by symmetry
    lowering perturbations of the four nearest-neighbor hoppings :math:`t \rightarrow t + \delta t_p`
    with :math:`p = 1, 2, 3, 4` indexing the four bonds connected to each site.

    .. versionadded:: 2.0.0

    Parameters
    ----------
    t : float
        Spin-independent nearest-neighbor hopping amplitude.
    soc : float
        Spin-orbit coupling strength. Modulates next-nearest neighbor
        hopping amplitudes.
    dt : list[float, float, float, float], optional
        Offsets added to the four nearest-neighbor hoppings along the
        bonds connected to each site. The entries are applied in the
        following order:

        - `dt[0]` : bond along ``R = [0, 0, 0]``
        - `dt[1]` : bond along ``R = [-1, 0, 0]``
        - `dt[2]` : bond along ``R = [0, -1, 0]``
        - `dt[3]` : bond along ``R = [0, 0, -1]``

        The default is ``[0, 0, 0, 0]``, which corresponds to uniform
        hopping amplitudes. This parameter allows for symmetry-lowering
        perturbations to the nearest-neighbor hoppings.

    Returns
    -------
    TBModel
        An instance of the model.

    Notes
    -----
    - The Fu-Kane-Mele model describes a three-dimensional topological insulator with a
      non-trivial band structure. It is characterized by a strong :math:`\mathbb{Z}_2` invariant
      and exhibits surface Dirac cones that are protected by time-reversal and inversion
      symmetry [1]_.

    References
    ----------
    .. [1] \ L. Fu, C. L. Kane, and E. J. Mele, *Phys. Rev. Lett.*, **98**, 106803
        (2007).
    """

    lat_vecs = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    orb_vecs = [[0, 0, 0], [0.25, 0.25, 0.25]]
    lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1, 2])

    model = TBModel(lattice=lat, spinful=True)

    # spin-independent first-neighbor hops
    for idx, lvec in enumerate([[0, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]):
        model.set_hop(t + dt[idx], 0, 1, lvec)

    # spin-dependent second-neighbor hops
    lvec_list = ([1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 1, 0], [0, -1, 1], [1, 0, -1])
    dir_list = ([0, 1, -1], [-1, 0, 1], [1, -1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1])
    for j in range(6):
        spin = np.array([0.0] + dir_list[j])
        model.set_hop(1j * soc * spin, 0, 0, lvec_list[j])
        model.set_hop(-1j * soc * spin, 1, 1, lvec_list[j])

    return model
