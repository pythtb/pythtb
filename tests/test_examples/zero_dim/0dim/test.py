import os
import numpy as np
import json
import datetime
import platform
from run import run

OUTPUTDIR = "golden_outputs"
OUTPUTNAME = "evals.npy" #NOTE: Replace with your expected output file name

def test_example():
    name = os.path.basename(os.path.dirname(__file__))
    group = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    log_file = os.path.join(base_path, group, "status.json")

    golden_path = os.path.join(os.path.dirname(__file__), OUTPUTDIR, OUTPUTNAME)

    entry = {
        "last_pass": datetime.datetime.now().isoformat(),
        "pythtb_version": get_version("pythtb"),
        "python_version": platform.python_version(),
        "status": "pass"
    }

    #NOTE: Modify as needed to handle golden data and generated data 
    expected = np.load(golden_path)
    result = run()
    try:
        assert np.allclose(result, expected, rtol=1e-8)
    except AssertionError as e:
        entry["status"] = "fail"
        entry["reason"] = str(e)

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
