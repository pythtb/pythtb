import numpy as np
from pythtb import TBModel


def fu_kane_mele(t, soc, m, beta):
    """
    fu_kane_mele tight-binding model.

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
        Adiabatic parameter which controls the strength of the magnetic field
        and the hopping amplitude along [111] direction.

    Returns
    -------
    TBModel
        An instance of the model.
    """

    lat = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    orb = [[0, 0, 0], [0.25, 0.25, 0.25]]
    model = TBModel(3, 3, lat, orb, nspin=2)

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
