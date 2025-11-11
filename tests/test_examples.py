# # tests/test_examples.py
# import subprocess
# from pathlib import Path

# # 1. Locate the directory this file lives in:
# HERE = Path(__file__).resolve().parent

# # 2. Go up one level to the repo root, then into `examples/`
# REPO_ROOT   = HERE.parent
# EXAMPLES_DIR = REPO_ROOT / "examples"

# # 3. Collect your scripts
# SCRIPTS = list(EXAMPLES_DIR.glob("*.py"))
# OUTPUT_DIR = HERE / "test_outputs"

# def run_scripts():
#     OUTPUT_DIR.mkdir(exist_ok=True)
#     for script in SCRIPTS:
#         print(f"Running {script.name}")
#         subprocess.run(
#             ["python", str(script)],
#             cwd=EXAMPLES_DIR,      # ensure any relative paths inside the examples work
#             check=True
#         )

# if __name__ == "__main__":
#     run_scripts()