import numpy as np
from pythtb.models import haldane


# Test Berry curvature and second Chern number calculation
def test_berry_curvature_second_chern(fkm_model):
    # FKM model has C2 = 1 for beta from 0 to 2pi
    betas = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    betas, theta_beta, c2 = fkm_model.axion_angle(
        nks=(13, 13, 13),
        beta=betas,
        param_periods={"beta": 2 * np.pi},
        return_second_chern=True,
    )
    assert 0.8 < c2 < 1.2  # coarse grid tolerance
    # Check that theta_curve has same number of points as betas
    assert theta_beta.shape[0] == betas.size


def test_chern_haldane():
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

    for t2_val in t2:
        model = haldane(delta=delta, t1=t1, t2=t2_val, phi=np.pi / 2)
        chern = model.chern_number(plane=(0, 1), nks=(200, 200))
        if abs(t2_val) < delta / (3 * np.sqrt(3)):
            np.testing.assert_almost_equal(
                chern,
                0,
                decimal=8,
                err_msg=f"Haldane model with t2={t2_val} should be trivial",
            )
        else:
            np.testing.assert_almost_equal(
                abs(chern),
                1,
                decimal=8,
                err_msg=f"Haldane model with t2={t2_val} should be topological",
            )
