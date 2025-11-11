import os
import numpy as np
from tests.utils import import_run


EXAMPLE_DIR = os.path.dirname(__file__)
run = import_run(EXAMPLE_DIR)
(
    berry_phase_circ_0,
    berry_phase_circ_1,
    berry_phase_circ_01,
    berr_flux_square_0,
    berr_flux_square_1,
    berr_flux_square_01,
    plaq,
) = run()


def test_answer():
    assert np.isclose(berr_flux_square_0, 2.17921648013)
    assert np.isclose(berr_flux_square_1, -2.17921648013)
    assert np.isclose(berr_flux_square_01, 0.0)
