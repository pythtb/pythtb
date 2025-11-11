import argparse
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
    write_file(
        example_dir / "run.py",
        """\
import numpy as np

def run():
    # Replace this with your model + computation
    return np.array([0.0])  # Dummy output
""",
    )

    # test.py
    write_file(
        example_dir / f"test_{name}.py",
        """\
import os
import numpy as np
from tests.utils import import_run

OUTPUTDIR = "golden_outputs"
#NOTE: Replace with your expected output file name(s). Should be in order
# of the results returned by run()
OUTPUTS = {
    "out1": "out1.npy",
    "out2": "out2.npy",
    "out3": "out3.npy"
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

""",
    )

    # regen_golden_data.py
    write_file(
        example_dir / "regen_golden_data.py",
        """\
# regen_golden_data.py
import os
import numpy as np
import json
import datetime
import platform
import importlib.metadata
from run import run

OUTPUTDIR = "golden_outputs"
FILENAMES = ["OUT1.npy", "OUT2.npy", "OUT3.npy"]
LOGFILE = os.path.join(os.path.dirname(__file__), OUTPUTDIR, "golden_log.json")

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
        "group": os.path.basename(os.path.dirname(os.path.dirname(__file__))),
        "name": os.path.basename(os.path.dirname(__file__)),
        "filenames": FILENAMES,
        "generated_at": datetime.datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "pythtb_version": get_version("pythtb")
    }
    with open(LOGFILE, "w") as f:
        json.dump(metadata, f, indent=4)

    print("✅ Golden data regenerated:", FILENAMES)

if __name__ == "__main__":
    regenerate()
""",
    )

    print(f"✅ Created example: {group}/{name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group", required=True, help="Model group name (e.g. haldane)"
    )
    parser.add_argument("--name", required=True, help="Example name (e.g. ex1)")
    args = parser.parse_args()
    create_example(args.group, args.name)


if __name__ == "__main__":
    main()
