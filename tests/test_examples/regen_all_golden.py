import os
import shutil
import subprocess
from pathlib import Path
import datetime

base_dir = Path("tests/test_examples")
backup_dir = Path("golden_backups") / datetime.datetime.now().strftime("backup_%Y%m%d_%H%M%S")

def backup_golden_outputs():
    print(f"\n📦 Backing up all golden_outputs/ to {backup_dir}")
    for group in base_dir.iterdir():
        if not group.is_dir():
            continue
        for example in group.iterdir():
            golden = example / "golden_outputs"
            if golden.exists():
                rel_path = group.name + "/" + example.name
                dest = backup_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(golden, dest, dirs_exist_ok=True)
    print("✅ Backup complete.\n")

def regenerate_all():
    print("🔁 Regenerating golden data for all examples...\n")
    count = 0
    for group in base_dir.iterdir():
        if not group.is_dir():
            continue
        for example in group.iterdir():
            regen_script = example / "regen_golden_data.py"
            if regen_script.exists():
                print(f"▶ Running {group.name}/{example.name}/regen_golden_data.py")
                try:
                    subprocess.run(["python", str(regen_script)], check=True)
                    count += 1
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed: {e}")
    print(f"\n✅ Regenerated golden data for {count} examples.\n")

if __name__ == "__main__":
    backup_golden_outputs()
    regenerate_all()