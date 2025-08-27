import sys

sys.path.append("../")

from pythtb import tb_model, wf_array
import numpy as np


def test_answer():

    # define lattice vectors
    lat = [[2.3, -0.2], [1.9, 2.4]]
    # define coordinates of orbitals
    orb = [[0.15, 0.34], [0.29, 0.65]]
    # make two-dimensional model
    bulk_model = tb_model(2, 2, lat, orb, per=[0, 1])

    # Add hopping terms.  Note that there are no hoppings
    # along the second periodic lattice vector.  Therefore
    # this nominally two-dimensional material is just an
    # infinite stack of one-dimensional wires.
    #
    t_first = 0.8 + 0.6j
    t_second = 2.0
    delta = -0.8
    bulk_model.set_onsite([-delta, delta])
    bulk_model.set_hop(t_second, 0, 0, [1, 0])
    bulk_model.set_hop(t_second, 1, 1, [1, 0])
    bulk_model.set_hop(t_first, 0, 1, [0, 0])
    bulk_model.set_hop(t_first, 1, 0, [1, 0])

    # sampling of Brillouin zone
    numk = 21  # should be an odd number
    # how many copies along direction 1
    num_wire = 3

    # compute berry phases for the bottom band along both directions
    bulk_array = wf_array(bulk_model, [numk, 100])
    bulk_array.solve_on_grid([0.0, 0.0])
    # (skip last kpoints to avoid double counting)
    bulk_phase_0 = np.mean(bulk_array.berry_phase([0], dir=0, contin=True)[:-1])
    bulk_phase_1 = np.mean(bulk_array.berry_phase([0], dir=1, contin=True)[:-1])
    # charge center
    bulk_location = (bulk_phase_0 / (2.0 * np.pi)) * bulk_model._lat[0] + (
        bulk_phase_1 / (2.0 * np.pi)
    ) * bulk_model._lat[1]
    # periodicity vectors for the wannier center
    bulk_location_periodicity = [bulk_model._lat[0], bulk_model._lat[1]]

    # make a better choice for the location of the charge center
    # this thing is hard-coded
    bulk_location = (
        bulk_location
        + 0 * bulk_location_periodicity[0]
        + 1 * bulk_location_periodicity[1]
    )

    # compute what would be charge center if stack three unit cells
    # on top of each other along direction 1
    bulk_location_three = (
        (bulk_location + 0 * bulk_location_periodicity[1])
        + (bulk_location + 1 * bulk_location_periodicity[1])
        + (bulk_location + 2 * bulk_location_periodicity[1])
    )
    # take the average of all three locations
    bulk_location_three = bulk_location_three / float(num_wire)

    # now enlarge model along direction 1
    sc_model = bulk_model.make_supercell(
        [[1, 0], [0, num_wire]], to_home=False, to_home_suppress_warning=True
    )
    sc_array = wf_array(sc_model, [numk, 100])
    sc_array.solve_on_grid([0.0, 0.0])
    # (skip last kpoints to avoid double counting)
    sc_phase_0 = np.mean(sc_array.berry_phase(range(num_wire), dir=0, contin=True)[:-1])
    sc_phase_1 = np.mean(sc_array.berry_phase(range(num_wire), dir=1, contin=True)[:-1])
    # charge center
    sc_location = (sc_phase_0 / (2.0 * np.pi)) * sc_model._lat[0] + (
        sc_phase_1 / (2.0 * np.pi)
    ) * sc_model._lat[1]
    # periodicity vectors for the wannier center
    sc_location_periodicity = [sc_model._lat[0], sc_model._lat[1]]
    # make a better choice for the location of the charge center
    # this thing is hard-coded
    sc_location = (
        sc_location + 1 * sc_location_periodicity[0] + 2 * sc_location_periodicity[1]
    )
    # divide with number of wires
    sc_location = sc_location / float(num_wire)

    assert np.allclose(bulk_location_three, sc_location, rtol=1.0e-5)

    # center of charge for system that is periodic along
    # direction 0 and finite along direction 1
    def get_centers_01(mod, num_bands):
        # get wavefunctions on a grid
        wfa = wf_array(mod, [numk])
        wfa.solve_on_grid([0.0])

        # compute center of charge along the periodic direction
        ph0 = wfa.berry_phase(range(num_bands), dir=0, contin=True)

        # for finite direction we simply compute average position
        pos1 = []
        for i in range(numk - 1):
            # sum over three bands
            pos1.append(np.sum(wfa.position_expectation([i], range(num_bands), dir=1)))
        # average over kpoints
        pos1 = np.mean(pos1)

        # construct center of charge
        ret_center = (ph0 / (2.0 * np.pi)) * mod._lat[0] + pos1 * mod._lat[1]
        # periodicity vectors for the center of charge
        ret_periodicity = [mod._lat[0]]

        return ret_center, ret_periodicity

    # now cut a finite piece, so that there
    # there are effectivelly only three one-dimensional "wires"
    finite_model = bulk_model.cut_piece(num=num_wire, fin_dir=1, glue_edgs=False)
    # get center of charge for this model
    finite_location, finite_location_periodicity = get_centers_01(
        finite_model, num_wire
    )
    # make a better choice for the location of the charge center
    # this thing is hard-coded
    finite_location = finite_location + 1 * finite_location_periodicity[0]
    # divide with number of wires
    finite_location = finite_location / float(num_wire)

    assert np.allclose(bulk_location_three, finite_location, rtol=1.0e-5)

    # now create a new finite model with a different non-periodic vector
    # code chooses automatically a non-periodic vector that is perpendicular to the periodic vector(s)
    finite_model_orthogonalized = finite_model.change_nonperiodic_vector(
        np_dir=1, new_latt_vec=None, to_home_suppress_warning=True
    )
    # get center of charge for model with these periodicity vectors
    finite_location_orthogonalized, finite_location_orthogonalized_periodicity = (
        get_centers_01(finite_model_orthogonalized, num_wire)
    )
    # make a better choice for the location of the charge center
    # this thing is hard-coded
    finite_location_orthogonalized = (
        finite_location_orthogonalized
        + 5 * finite_location_orthogonalized_periodicity[0]
    )
    # divide with number of wires
    finite_location_orthogonalized = finite_location_orthogonalized / float(num_wire)

    assert np.allclose(bulk_location_three, finite_location_orthogonalized, rtol=1.0e-3)

    # redo everything as above but with an arbitrary choice of a non-periodic vector
    finite_model_arb = finite_model.change_nonperiodic_vector(
        np_dir=1, new_latt_vec=[-1.3, 4.8], to_home_suppress_warning=True
    )
    finite_location_arb, finite_location_arb_periodicity = get_centers_01(
        finite_model_arb, num_wire
    )
    # make a better choice for the location of the charge center
    # this thing is hard-coded
    finite_location_arb = finite_location_arb + 6 * finite_location_arb_periodicity[0]
    # divide with number of wires
    finite_location_arb = finite_location_arb / float(num_wire)

    assert np.allclose(bulk_location_three, finite_location_arb, rtol=1.0e-3)
