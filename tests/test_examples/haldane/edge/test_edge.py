import os
import numpy as np
from tests.utils import import_run

OUTPUTDIR = "golden_outputs"
# NOTE: Replace with your expected output file name(s). Should be in order
# of the results returned by run()
OUTPUTS = {
    "evals": "evals.npy",
    "evecs": "evecs.npy",
    "evals_half": "evals_half.npy",
    "evecs_half": "evecs_half.npy",
}


def test_example():
    example_dir = os.path.dirname(__file__)
    run = import_run(example_dir)

    # Load expected results
    expected = {}
    for label, fname in OUTPUTS.items():
        path = os.path.join(os.path.dirname(__file__), OUTPUTDIR, fname)
        expected[label] = np.load(path)

    # Get result from model
    results = run()
    if not isinstance(results, (tuple, list)):
        results = [results]
    if len(results) != len(OUTPUTS):
        raise AssertionError(f"Expected {len(OUTPUTS)} outputs, got {len(results)}")
    # Compare results with expected outputs
    # NOTE: Modify to match your expected output structure
    for i, (label, fname) in enumerate(OUTPUTS.items()):
        result = results[i]
        # There are degenerate states in the Haldane model, so we need to check
        # that the projectors are equivalent rather than the eigenvectors themselves.
        if label == "evecs_half":
            P1 = result @ result.conj().T  # Projector for result
            P2 = expected[label] @ expected[label].conj().T  # Projector for golden
            assert np.allclose(P1, P2, rtol=1e-8, atol=1e-14), (
                f"Projectors for {label} are not equivalent"
            )
        else:
            np.testing.assert_allclose(result, expected[label], rtol=1e-8, atol=1e-14)
