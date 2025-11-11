import os
import numpy as np
import json
import datetime
import platform
from run import run

OUTPUTDIR = "golden_outputs"
OUTPUTS = {
    "out1": "out1.npy",
    "out2": "out2.npy",
    "out3": "out3.npy"
}
#NOTE: Replace with your expected output file name(s). Should be in order
# of the results returned by run()

def test_example():
    name = os.path.basename(os.path.dirname(__file__))
    group = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    log_file = os.path.join(base_path, group, "status.json")

    # Load expected results
    expected = {}
    for label, fname in OUTPUTS.items():
        path = os.path.join(os.path.dirname(__file__), OUTPUTDIR, fname)
        expected[label] = np.load(path)
    
    # Get result from model
    results = run()
    if not isinstance(results, (tuple, list)):
        results = [results]

    # Prepare entry
    entry = {
        "last_pass": datetime.datetime.now().isoformat(),
        "pythtb_version": get_version("pythtb"),
        "python_version": platform.python_version(),
        "status": "pass"
    }
    
    if len(results) != len(OUTPUTS):
        entry["status"] = "fail"
        entry["reason"] = f"Expected {len(OUTPUTS)} outputs, got {len(results)}"
        raise AssertionError(entry["reason"])
    # Compare results with expected outputs
    #NOTE: Modify to match your expected output structure
    try:
        for i, (label, fname) in enumerate(OUTPUTS.items()):
            np.testing.assert_allclose(results[i], expected[label], rtol=1e-8, atol=1e-14)
    except AssertionError as e:
        entry["status"] = "fail"
        entry["reason"] = f"Mismatch in {label}: {str(e)}"

    # Log the pass/fail status
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    status = {}
    if os.path.exists(log_file):
        try:
            with open(log_file) as f:
                content = f.read().strip()
                if content:
                    status = json.loads(content)
        except Exception as e:
            print(f"⚠️ Warning: Couldn't parse {log_file}, starting fresh. Reason: {e}")

    status[name] = entry
    with open(log_file, "w") as f:
        json.dump(status, f, indent=4)

    if entry["status"] == "fail":
        raise AssertionError(entry["reason"])

def get_version(pkg):
    try:
        import importlib.metadata as im
        return im.version(pkg)
    except Exception:
        return "unknown"
