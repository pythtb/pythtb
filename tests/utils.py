import importlib.util
import os


# This avoids namespace conflicts in the test environment
def import_run(example_dir):
    spec = importlib.util.spec_from_file_location(
        "run", os.path.join(example_dir, "run.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run
