# tests/conftest.py
import pytest
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from copy import deepcopy
import re

results = defaultdict(list)

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()

    if rep.when == "call":
        test_path = Path(item.fspath).relative_to(item.config.rootdir)
        test_name = item.name
        passed = rep.passed
        results[str(test_path.with_suffix(""))].append((test_name, passed))

def build_nested_tree(results):
    tree = {}
    for path, tests in results.items():
        parts = Path(path).parts
        node = tree
        for part in parts:
            node = node.setdefault(part, {})
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        node["_tests"] = sorted((name, passed, now if passed else None) for name, passed in tests)
    return tree

def format_tree(node, indent=0):
    lines = []
    for key, value in sorted(node.items()):
        if key == "_tests":
            for test_name, passed, timestamp in value:
                check = "✅" if passed else "❌"
                time_str = f" — *{timestamp}*" if timestamp else ""
                lines.append("  " * indent + f"- {check} `{test_name}`{time_str}")
        else:
            lines.append("  " * indent + f"- **{key}/**")
            lines.extend(format_tree(value, indent + 1))
    return lines

def load_existing_tree(filepath):
    if not filepath.exists():
        return {}

    tree = {}
    stack = [tree]
    current_indent = 0

    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            if stripped.startswith("- **") and stripped.endswith("/**"):
                key = stripped[4:-3]
                level = indent // 2
                while len(stack) > level + 1:
                    stack.pop()
                new_node = {}
                stack[-1][key] = new_node
                stack.append(new_node)
            elif stripped.startswith("- ✅") or stripped.startswith("- ❌"):
                match = re.match(r"- (✅|❌) `(.+?)`(?: — \*(.+?)\*)?", stripped)
                if match:
                    status, name, time = match.groups()
                    stack[-1].setdefault("_tests", []).append((name, status == "✅", time))
    return tree

def merge_trees(existing, new):
    for key, value in new.items():
        if key == "_tests":
            existing_tests = {name: (status, ts) for name, status, ts in existing.get("_tests", [])}
            for name, status, timestamp in value:
                if name not in existing_tests or not existing_tests[name][0]:  # Only update on pass or new
                    existing_tests[name] = (status, timestamp)
            existing["_tests"] = sorted([(k, v[0], v[1]) for k, v in existing_tests.items()])
        else:
            existing.setdefault(key, {})
            merge_trees(existing[key], value)
            

def format_tree_with_time(node, indent=0):
    lines = []
    for key, value in sorted(node.items()):
        if key == "_tests":
            for test_name, passed, timestamp in value:
                check = "✅" if passed else "❌"
                time_str = f" — *{timestamp}*" if timestamp else ""
                lines.append("  " * indent + f"- {check} `{test_name}`{time_str}")
        else:
            lines.append("  " * indent + f"- **{key}/**")
            lines.extend(format_tree_with_time(value, indent + 1))
    return lines


def render_markdown(tree, updated_at):
    output = [
        "# 📋 Test Status Report",
        "",
        f"Last updated on **{updated_at}**",
        "",
        "---",
        ""
    ]
    output.extend(format_tree_with_time(tree))
    return "\n".join(output)

@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    out_path = Path(session.config.rootdir) / "tests" / "README.md"
    existing_tree = load_existing_tree(out_path)
    new_tree = build_nested_tree(results)

    # Merge into a copy so we can detect real changes
    merged_tree = deepcopy(existing_tree)
    merge_trees(merged_tree, new_tree)

    # If nothing changed, don't touch the file (prevents spurious diffs)
    if merged_tree == existing_tree:
        print(f"\nℹ️ No test result changes; leaving {out_path} untouched.")
        return

    ts = datetime.now().strftime('%Y-%m-%d at %H:%M:%S')
    out_path.write_text(render_markdown(merged_tree, ts), encoding="utf-8")
    print(f"\n✅ Updated test results in {out_path}")