# regen_golden_data.py
import os
import numpy as np
import json
import datetime
import platform
import importlib.metadata
from run import run

OUTPUTDIR = "golden_outputs"
FILENAMES = [
    "bphase_circ0.npy", "bphase_circ1.npy", "bphase_circ01.npy",
    "bflux_square_0.npy", "bflux_square_1.npy", "bflux_square_01.npy",
    "plaq.npy"]
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
