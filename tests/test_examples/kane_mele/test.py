import numpy as np
import pytest
from run import run_kane_mele
import os

OUTPUTDIR = "golden_outputs"

def test_kane_mele():
    evals_golden = np.load(os.path.join(OUTPUTDIR, "kane_mele_evals.npy"))
    wan_cent_golden = np.load(os.path.join(OUTPUTDIR, "kane_mele_wan_cent.npy"))

    evals_golden = np.array(evals_golden)
    wan_cent_golden = np.array(wan_cent_golden)

    #NOTE: evals in pythTB v1.8.0 have last two axes swapped (bands, k-points)
    evals_golden = np.transpose(evals_golden, (0, 2, 1))

    evals_list, wan_cent_list = run_kane_mele()

    assert np.allclose(evals_list, evals_golden, rtol=1e-8), "Eigenvalues mismatch"
    assert np.allclose(wan_cent_list, wan_cent_golden, rtol=1e-8), "Wannier centers mismatch"
