import os
import subprocess
from pathlib import Path

def find_tests():
    test_files = []
    base = Path("tests/test_examples")
    for group in base.iterdir():
        if not group.is_dir():
            continue
        for example in group.iterdir():
            test_file = example / "test.py"
            if test_file.exists():
                test_files.append(test_file)
    return test_files

def run_tests():
    test_files = find_tests()
    print(f"Running {len(test_files)} tests...\n")
    failed = 0
    for test_file in test_files:
        print(f"▶ {test_file}")
        try:
            subprocess.run(["python", str(test_file)], check=True)
        except subprocess.CalledProcessError:
            print("❌ FAILED\n")
            failed += 1
        else:
            print("✅ PASSED\n")
    print(f"=== {len(test_files) - failed} passed, {failed} failed ===")

if __name__ == "__main__":
    run_tests()