import os
import numpy as np
import json
import datetime
import platform
from tests.utils import import_run

OUTPUTDIR = "golden_outputs"
OUTPUTS = {
    "phi_1": "phi1.npy",
    "rib_eval": "rib_eval.npy",
    "jump_k": "jump_k.npy",
    "pos_exps": "pos_exps.npy",
    "hwfcs": "hwfcs.npy",
}
# NOTE: Replace with your expected output file name(s). Should be in order
# of the results returned by run()


def test_example():
    example_dir = os.path.dirname(__file__)
    run = import_run(example_dir)
    name = os.path.basename(os.path.dirname(__file__))
    group = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    log_file = os.path.join(base_path, group, "status.json")

    # Load expected results
    expected = {}
    for label, fname in OUTPUTS.items():
        path = os.path.join(os.path.dirname(__file__), OUTPUTDIR, fname)
        expected[label] = np.load(path, allow_pickle=True)

    # Get result from model
    results = run()
    if not isinstance(results, (tuple, list)):
        results = [results]

    # Prepare entry
    entry = {
        "last_pass": datetime.datetime.now().isoformat(),
        "pythtb_version": get_version("pythtb"),
        "python_version": platform.python_version(),
        "status": "pass",
    }

    if len(results) != len(OUTPUTS):
        entry["status"] = "fail"
        entry["reason"] = f"Expected {len(OUTPUTS)} outputs, got {len(results)}"
        raise AssertionError(entry["reason"])
    # Compare results with expected outputs
    # NOTE: Modify to match your expected output structure
    try:
        for i, (label, fname) in enumerate(OUTPUTS.items()):
            result = results[i]
            expect = expected[label]
            if label == "rib_eval":
                np.testing.assert_allclose(result, expect, rtol=1e-8, atol=1e-14)
            elif label == "jump_k":
                np.testing.assert_array_equal(result, expect)
            elif label == "pos_exps":
                print(result.shape)
                result = np.array(result, dtype=complex)
                expect = np.array(expect, dtype=complex)
                np.testing.assert_allclose(result, expect, rtol=1e-8, atol=1e-14)
            elif label == "hwfcs":
                for j in range(result.shape[0]):
                    result[j] = np.array(result[j], dtype=complex)
                    expect[j] = np.array(expect[j], dtype=complex)
                    np.testing.assert_allclose(
                        result[j], expect[j], rtol=1e-8, atol=1e-14
                    )
            elif label == "phi_1":
                if isinstance(result, np.ndarray) and result.dtype == "object":
                    # Convert object arrays to numpy arrays for comparison
                    result = np.array(
                        [np.array(item) for item in result], dtype=complex
                    )
                if isinstance(expect, np.ndarray) and expect.dtype == "object":
                    expect = np.array(
                        [np.array(item) for item in expect], dtype=complex
                    )
                np.testing.assert_allclose(result, expect, rtol=1e-8, atol=1e-14)

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
