import numpy as np
import pytest
from pythtb.models import haldane
from pythtb import WFArray, Mesh


@pytest.mark.parametrize(
    "k_endpoints",
    [
        (True),  # Include BZ boundary points
        (False),  # Exclude BZ boundary points
    ],
)
def test_chern_haldane(k_endpoints):
    delta = 1
    t1 = 1
    eps = 1e-1  # small number to avoid phase boundaries
    t2 = [
        -2 * delta / (3 * np.sqrt(3)),  # should be topological
        -delta / (3 * np.sqrt(3)) - eps,  # should be topological
        -delta / (3 * np.sqrt(3)) + eps,  # should be trivial
        0,  # should be trivial
        delta / (3 * np.sqrt(3)) - eps,  # should be trivial
        delta / (3 * np.sqrt(3)) + eps,  # should be topological
        2 * delta / (3 * np.sqrt(3)),  # should be topological
    ]

    model = haldane(delta=delta, t1=t1, t2=0, phi=np.pi / 2)

    mesh = Mesh(["k", "k"])
    mesh.build_grid([100, 100], k_endpoints=k_endpoints)
    wfa = WFArray(model.lattice, mesh)

    for t2_val in t2:
        my_model = haldane(delta=delta, t1=t1, t2=t2_val, phi=np.pi / 2)
        wfa.solve_model(my_model)
        chern0 = wfa.chern_number(plane=(0, 1), state_idx=[0])
        chern1 = wfa.chern_number(plane=(0, 1), state_idx=[1])

        if abs(t2_val) < delta / (3 * np.sqrt(3)):
            np.testing.assert_almost_equal(
                chern0,
                0,
                decimal=8,
                err_msg=f"Haldane model with t2={t2_val} should be trivial",
            )
        else:
            np.testing.assert_almost_equal(
                abs(chern1),
                1,
                decimal=8,
                err_msg=f"Haldane model with t2={t2_val} should be topological",
            )
            np.testing.assert_almost_equal(
                chern0,
                -chern1,
                decimal=8,
                err_msg="Chern numbers for band 0 and 1 should be opposite",
            )
