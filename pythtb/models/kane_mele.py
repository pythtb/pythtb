from pythtb import TBModel, Lattice
import numpy as np


def kane_mele(delta, t, soc, rashba) -> TBModel:
    r"""Kane-Mele tight-binding model.

    .. versionadded:: 2.0.0

    This function creates a Kane-Mele tight-binding model with the specified
    parameters. The model is defined on a 2D honeycomb lattice with two sublattices.
    The lattice vectors are given by:

    .. math::

        \mathbf{a}_1 = a(1, 0), \quad \mathbf{a}_2 = a\left(\frac{1}{2}, \frac{\sqrt{3}}{2}\right),

    and the orbital positions are given by:

    .. math::

        \mathbf{r}_1 = \frac{1}{3} \mathbf{a}_1 + \frac{1}{3} \mathbf{a}_2, 
        \quad \mathbf{r}_2 = \frac{2}{3} \mathbf{a}_1 + \frac{2}{3} \mathbf{a}_2

    The Hamiltonian in second-quantized form is given by:

    .. math::

        H = \Delta \sum_{i} c_i^\dagger c_i + 
        t \sum_{\langle i,j \rangle} ( c_i^\dagger c_j + h.c.) +
        \lambda_{SO} \sum_{\langle \langle i,j \rangle \rangle} ( c_i^\dagger \sigma_z c_j + \text{h.c.}) + \\
        \lambda_{R} \sum_{\langle i,j \rangle} ( c_i^\dagger \mathbf{\sigma} \times 
        \mathbf{\hat{d}}_{\langle i,j \rangle} c_j + \text{h.c.})

    Parameters
    ----------
    onsite : float
        On-site energy.
    t : float, complex
        Hopping parameter.
    soc : float, complex
        Spin-orbit coupling strength.
    rashba : float, complex
        Rashba coupling strength.

    Returns
    -------
    TBModel
        An instance of the model.

    Notes
    -----
    The Kane-Mele model describes a two-dimensional topological insulator with spin-orbit coupling.
    It is defined on a honeycomb lattice and includes both intrinsic and Rashba spin-orbit coupling [kane-mele]_.

    References
    ----------
    .. [kane-mele] Kane, C. L., & Mele, E. J. (2005). Quantum Spin Hall Effect in Graphene. *Physical Review Letters*, 95(22), 226801.
    """

    # define lattice vectors
    lat_vecs = [[1, 0], [1 / 2, np.sqrt(3) / 2]]
    # define coordinates of orbitals
    orb_vecs = [[1 / 3, 1 / 3], [2 / 3, 2 / 3]]

    lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1])

    # make two dimensional tight-binding Kane-Mele model
    ret_model = TBModel(lattice=lat, spinful=True)

    # set on-site energies
    ret_model.set_onsite([delta, -delta])

    # useful definitions
    sigma_x = np.array([0, 1, 0, 0])
    sigma_y = np.array([0, 0, 1, 0])
    sigma_z = np.array([0, 0, 0, 1])

    # set hoppings (one for each connected pair of orbitals)
    # (amplitude, i, j, [lattice vector to cell containing j])

    # spin-independent first-neighbor hoppings
    ret_model.set_hop(t, 0, 1, [0, 0])
    ret_model.set_hop(t, 0, 1, [0, -1])
    ret_model.set_hop(t, 0, 1, [-1, 0])

    # second-neighbour spin-orbit hoppings (s_z)
    nnn_hop = 1j * soc * sigma_z
    ret_model.set_hop(-nnn_hop, 0, 0, [0, 1])
    ret_model.set_hop(nnn_hop, 0, 0, [1, 0])
    ret_model.set_hop(-nnn_hop, 0, 0, [1, -1])
    ret_model.set_hop(nnn_hop, 1, 1, [0, 1])
    ret_model.set_hop(-nnn_hop, 1, 1, [1, 0])
    ret_model.set_hop(nnn_hop, 1, 1, [1, -1])

    # Rashba first-neighbor hoppings: (s_x)(dy)-(s_y)(d_x)

    # bond unit vectors are (np.sqrt(3) / 2, 1/2) then (0,-1) then (-np.sqrt(3) / 2, 1/2)
    ret_model.set_hop(
        1j * rashba * ((1 / 2) * sigma_x - (np.sqrt(3) / 2) * sigma_y),
        0,
        1,
        [0, 0],
        mode="add",
    )
    ret_model.set_hop(1j * rashba * -sigma_x, 0, 1, [0, -1], mode="add")
    ret_model.set_hop(
        1j * rashba * ((1 / 2) * sigma_x + (np.sqrt(3) / 2) * sigma_y),
        0,
        1,
        [-1, 0],
        mode="add",
    )

    return ret_model
