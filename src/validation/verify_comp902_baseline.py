"""
verify_comp902_baseline.py

Description:
    Confirms that the COMP903 repository reproduces the frozen COMP902
    pipeline outputs exactly. The parsed turns, thread-level features and
    train/test split regenerated in this repository are compared with the
    files in the COMP902 repository.

    Each file is compared using a SHA-256 hash after line endings are
    normalised, so the check gives the same result on Windows and Linux.

Inputs:
    - data/processed/scam_dialogue_parsed.csv
    - data/processed/scam_dialogue_thread_features.csv
    - data/processed/split_indices.json
    - The same three files in the COMP902 repository (--reference-dir)

Output:
    - outputs/reports/comp902_baseline_verification.md

Run:
    - python src/validation/verify_comp902_baseline.py --reference-dir "C:\\path\\to\\scam-conversation-risk-evaluator"
"""

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import BASELINE_VERIFICATION_REPORT, PARSED_DATA, PROCESSED_DATA, SPLIT_INDICES

COMP902_SOURCE_COMMIT = "a282ecf"

FILES_TO_COMPARE = {
    "Parsed turns": PARSED_DATA,
    "Thread-level features": PROCESSED_DATA,
    "Train/test split indices": SPLIT_INDICES,
}


def normalised_hash(file_path):
    """
    Description:
        Calculates a SHA-256 hash of a file after converting Windows line
        endings (CRLF) to Unix line endings (LF).

    Input:
        file_path: Path to the file.

    Output:
        Returns the hash as a hexadecimal string.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    content = Path(file_path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def count_lines(file_path):
    """
    Description:
        Counts the lines in a text file.

    Input:
        file_path: Path to the file.

    Output:
        Returns the number of lines.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def compare_files(reference_dir):
    """
    Description:
        Compares each COMP903 file with the matching COMP902 file.

    Input:
        reference_dir: Path to the COMP902 repository folder.

    Output:
        Returns a list of result dictionaries, one per file.
    """
    results = []

    for label, comp903_path in FILES_TO_COMPARE.items():
        relative_path = comp903_path.relative_to(PROJECT_ROOT)
        comp902_path = Path(reference_dir) / relative_path

        comp903_hash = normalised_hash(comp903_path)
        comp902_hash = normalised_hash(comp902_path)

        results.append(
            {
                "label": label,
                "file": relative_path.as_posix(),
                "comp903_lines": count_lines(comp903_path),
                "comp902_lines": count_lines(comp902_path),
                "comp903_hash": comp903_hash,
                "comp902_hash": comp902_hash,
                "match": comp903_hash == comp902_hash,
            }
        )

    return results


def write_report(results, reference_dir):
    """
    Description:
        Writes the verification results as a markdown report.

    Input:
        results: List of result dictionaries from compare_files().
        reference_dir: Path to the COMP902 repository folder.

    Output:
        Saves the report and returns True if every file matched.
    """
    all_match = all(row["match"] for row in results)

    lines = [
        "# COMP902 Baseline Verification",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"COMP902 source commit: {COMP902_SOURCE_COMMIT}",
        f"Reference folder: {Path(reference_dir).name}",
        "",
        f"Overall result: {'PASS - all files identical' if all_match else 'FAIL - see differences below'}",
        "",
        "| File | COMP903 lines | COMP902 lines | Match |",
        "|---|---|---|---|",
    ]

    for row in results:
        lines.append(
            f"| {row['file']} | {row['comp903_lines']} | {row['comp902_lines']} | "
            f"{'Yes' if row['match'] else 'No'} |"
        )

    lines += ["", "## SHA-256 Hashes (line endings normalised)", ""]
    for row in results:
        lines += [
            f"- {row['label']}",
            f"  - COMP903: `{row['comp903_hash']}`",
            f"  - COMP902: `{row['comp902_hash']}`",
        ]

    BASELINE_VERIFICATION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_VERIFICATION_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return all_match


def main():
    parser = argparse.ArgumentParser(description="Verify COMP903 outputs against COMP902.")
    parser.add_argument(
        "--reference-dir",
        required=True,
        help="Path to the COMP902 repository folder",
    )
    args = parser.parse_args()

    results = compare_files(args.reference_dir)
    all_match = write_report(results, args.reference_dir)

    for row in results:
        print(f"{row['file']}: {'identical' if row['match'] else 'DIFFERENT'}")

    print(f"\nReport saved to: {BASELINE_VERIFICATION_REPORT}")

    if not all_match:
        sys.exit(1)


if __name__ == "__main__":
    main()