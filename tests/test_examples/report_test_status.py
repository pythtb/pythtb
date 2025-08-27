import os
import json
from pathlib import Path

base_dir = Path(".")
md_lines = ["# ✅ Test Status Checklist", ""]

def print_report():
    print(f"{'Group':<15} | {'Example':<25} | {'Status':<8} | {'Last Pass'}")
    print("-" * 80)

    for group in sorted(base_dir.iterdir()):
        if not group.is_dir():
            continue
        status_file = group / "status.json"
        if not status_file.exists():
            continue

        with open(status_file) as f:
            try:
                status = json.load(f)
            except Exception:
                print(f"⚠️ Skipping corrupt {status_file}")
                continue

        group_all_passed = True
        group_lines = []

        for example, entry in sorted(status.items()):
            status_str = entry.get("status", "unknown")
            last_pass = entry.get("last_pass", "")
            is_pass = status_str == "pass"
            checkmark = "[x]" if is_pass else "[ ]"
            group_all_passed &= is_pass
            group_lines.append(f"    - {checkmark} {example}")

            print(f"{group.name:<15} | {example:<25} | {status_str:<8} | {last_pass}")

        group_checkmark = "[x]" if group_all_passed and group_lines else "[ ]"
        md_lines.append(f"- {group_checkmark} {group.name}")
        md_lines.extend(group_lines)

    with open("PASSING.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print("\n✅ Markdown checklist written to `PASSING.md`")

if __name__ == "__main__":
    print_report()