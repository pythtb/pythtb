import os
import numpy as np
from tests.utils import import_run

OUTPUTDIR = "golden_outputs"
#NOTE: Replace with your expected output file name(s). Should be in order
# of the results returned by run()
OUTPUTS = {
    "berry_phase": "berry_phase_orig.npy",
    "berry_phase2": "berry_phase_perp.npy",
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
    #NOTE: Modify to match your expected output structure
  
    for i, (label, fname) in enumerate(OUTPUTS.items()):
        np.testing.assert_allclose(results[i], expected[label], rtol=1e-8, atol=1e-14)