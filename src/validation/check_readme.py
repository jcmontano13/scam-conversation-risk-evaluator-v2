"""
check_readme.py

Description:
    Checks that README.md still matches the repository, so that anyone can
    rerun the project from the README alone. A README goes out of date
    quietly: a script is renamed, an output is added, a command is never
    written down. This script finds those gaps.

    Checks:
    1. Every "python <script>.py" command in the README points to a script
       that exists.
    2. Every Python script under src/ is mentioned in the README.
    3. Every test file under tests/ is mentioned in the README.
    4. Every output file named in config.py (data/processed, outputs, docs)
       is mentioned in the README by file name.
    5. The README has the sections a new user needs: setup, dataset,
       reproducibility settings, execution order and results.
    6. The Quick Start runs the scripts in the same order as the
       Execution Order section.

    The script only reads; it never edits the README.

Inputs:
    - README.md
    - config.py
    - src/ and tests/ folders

Outputs:
    - outputs/reports/readme_check.txt

Run:
    - python src/validation/check_readme.py

Exit code:
    - 0 if every check passes, 1 otherwise.
"""

import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import config  # noqa: E402

README_PATH = PROJECT_ROOT / "README.md"
OUTPUT_PATH = Path(config.REPORTS_DIR) / "readme_check.txt"

# Folders whose files a user must know about. Raw data is covered by the Dataset section.
DOCUMENTED_FOLDERS = ["data/processed", "outputs", "docs"]

# Python files that are imported, not run, so they need a mention but no command.
MODULES = {"__init__.py"}

REQUIRED_SECTIONS = [
    "## Quick Start",
    "## Setup",
    "### Dataset",
    "## Reproducibility Settings",
    "## Execution Order",
    "## Results Summary",
]


def mentioned(name, text):
    """True if a file name appears in the text as a whole name (cv_results.csv is not found inside repeated_cv_results.csv)."""
    return re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text) is not None


def read_readme():
    if not README_PATH.exists():
        raise FileNotFoundError("README.md not found in the project root.")
    return README_PATH.read_text(encoding="utf-8")


def commands_in(text):
    """Returns every script path used in a 'python <path>.py' command, in order."""
    return [match.replace("\\", "/") for match in re.findall(r"python\s+([\w./\\-]+\.py)", text)]


def check_commands_exist(readme):
    missing = sorted({script for script in commands_in(readme) if not (PROJECT_ROOT / script).exists()})
    return not missing, missing


def check_scripts_documented(readme):
    scripts = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "src").rglob("*.py")
        if path.name not in MODULES and "__pycache__" not in path.parts
    )
    missing = [script for script in scripts if not mentioned(script, readme)]
    return not missing, missing, len(scripts)


def check_tests_documented(readme):
    tests = sorted(path.name for path in (PROJECT_ROOT / "tests").glob("test_*.py"))
    missing = [test for test in tests if not mentioned(test, readme)]
    return not missing, missing, len(tests)


def config_output_files():
    """Returns {config name: relative path} for every file path in config.py under the documented folders."""
    files = {}
    for name in dir(config):
        if not name.isupper():
            continue
        value = getattr(config, name)
        if not isinstance(value, Path) or not value.suffix:
            continue
        try:
            relative = value.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            continue
        if any(relative.startswith(folder) for folder in DOCUMENTED_FOLDERS):
            files[name] = relative
    return files


def check_outputs_documented(readme):
    files = config_output_files()
    missing = sorted(f"{name} -> {path}" for name, path in files.items() if not mentioned(Path(path).name, readme))
    return not missing, missing, len(files)


def check_sections(readme):
    missing = [section for section in REQUIRED_SECTIONS if not re.search(rf"^{re.escape(section)}\b", readme, re.M)]
    return not missing, missing


def check_order(readme):
    """The Quick Start and the Execution Order list the scripts in the same order."""
    try:
        quick = readme.split("## Quick Start", 1)[1].split("\n## ", 1)[0]
        execution = readme.split("## Execution Order", 1)[1].split("\n## ", 1)[0]
    except IndexError:
        return False, ["Quick Start or Execution Order section missing"]

    quick_order = list(dict.fromkeys(commands_in(quick)))
    execution_order = list(dict.fromkeys(commands_in(execution)))
    if quick_order == execution_order:
        return True, []

    only_quick = [s for s in quick_order if s not in execution_order]
    only_execution = [s for s in execution_order if s not in quick_order]
    problems = [f"Only in Quick Start: {s}" for s in only_quick] + [f"Only in Execution Order: {s}" for s in only_execution]
    if not problems:
        problems = ["Same scripts, different order"]
    return False, problems


def main():
    readme = read_readme()
    results = []

    passed, missing = check_commands_exist(readme)
    results.append(("Every README command points to an existing script", passed, missing))

    passed, missing, total = check_scripts_documented(readme)
    results.append((f"Every script in src/ is documented ({total} scripts)", passed, missing))

    passed, missing, total = check_tests_documented(readme)
    results.append((f"Every test file is documented ({total} files)", passed, missing))

    passed, missing, total = check_outputs_documented(readme)
    results.append((f"Every output file in config.py is listed ({total} files)", passed, missing))

    passed, missing = check_sections(readme)
    results.append(("README has all required sections", passed, missing))

    passed, missing = check_order(readme)
    results.append(("Quick Start order matches Execution Order", passed, missing))

    lines = ["README Check", f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for check, passed, missing in results:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {check}")
        lines.append(f"[{status}] {check}")
        for item in missing:
            print(f"        - {item}")
            lines.append(f"        - {item}")

    failed = sum(1 for _, passed, _ in results if not passed)
    verdict = "README is complete" if failed == 0 else f"{failed} check(s) failed - update README.md and run again"
    lines += ["", verdict]
    print(f"\n{verdict}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {OUTPUT_PATH}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()