import sys

sys.path.append("../../examples")
sys.path.append("../")

import numpy as np
from examples.edge import evecs, ed


def test_answer():
    print(evecs[ed, :])
    assert np.isclose(evecs[ed, 0], 2.80366220e-01 + 0.00000000e00j)
    assert np.isclose(evecs[ed, 4], 1.56548374e-01 + 3.48852597e-02j)
