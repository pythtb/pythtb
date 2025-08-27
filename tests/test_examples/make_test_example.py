import os
import argparse
import datetime
import json
import platform
from pathlib import Path
import importlib.metadata

BASE_DIR = Path(".")

def get_version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"

def write_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

def create_example(group, name):
    example_dir = BASE_DIR / group / name
    golden_dir = example_dir / "golden_outputs"
    golden_dir.mkdir(parents=True, exist_ok=True)

    # run.py
    write_file(example_dir / "run.py", '''\
import numpy as np

def run():
    # Replace this with your model + computation
    return np.array([0.0])  # Dummy output
''')

    # test.py
    write_file(example_dir / "test.py", '''\
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
''')

    # regen_golden_data.py
    write_file(example_dir / "regen_golden_data.py", '''\
# regen_golden_data.py
import os
import numpy as np
import json
import datetime
import platform
import importlib.metadata
from .run import run

OUTPUTDIR = "golden_outputs"
FILENAMES = ["OUT1.npy", "OUT2.npy", "OUT3.npy"]
LOGFILE = os.path.join(os.path.dirname(__file__), "golden_log.json")

def get_version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"

def regenerate():
    os.makedirs(OUTPUTDIR, exist_ok=True)
    results = run()
    if not isinstance(results, (tuple, list)):
        results = [results]

    for result, fname in zip(results, FILENAMES):
        path = os.path.join(OUTPUTDIR, fname)
        np.save(path, result)

    metadata = {
        "generated_at": datetime.datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "pythtb_version": get_version("pythtb")
    }
    with open(LOGFILE, "w") as f:
        json.dump(metadata, f, indent=4)

    print("✅ Golden data regenerated:", FILENAMES)

if __name__ == "__main__":
    regenerate()
''')

    # Ensure empty or missing status.json is handled gracefully
    status_path = BASE_DIR / group / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)

    status = {}
    if status_path.exists():
        try:
            with open(status_path) as f:
                content = f.read().strip()
                if content:
                    status = json.loads(content)
        except Exception as e:
            print(f"⚠️ Warning: Couldn't parse existing {status_path.name}, starting fresh. Reason: {e}")

    status[name] = {
        "status": "unknown",
        "created": datetime.datetime.now().isoformat(),
        "pythtb_version": get_version("pythtb"),
        "python_version": platform.python_version()
    }

    with open(status_path, "w") as f:
        json.dump(status, f, indent=4)

    print(f"✅ Created example: {group}/{name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, help="Model group name (e.g. haldane)")
    parser.add_argument("--name", required=True, help="Example name (e.g. ex1)")
    args = parser.parse_args()
    create_example(args.group, args.name)

if __name__ == "__main__":
    main()