#!/usr/bin/env python3
"""
Test MaxSAT wcnf files for errors using compare.py and write detailed and summary results to CSV.
"""
import glob
import subprocess
import re
import csv
import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Scan .wcnf files for solver errors via compare.py and save to CSV"
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=".",
        help="Directory containing .wcnf files (default: current directory)",
    )
    parser.add_argument(
        "-o", "--output",
        default="results.csv",
        help="Output detailed CSV filename (default: results.csv)",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.input_dir, "*_*_*.wcnf")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files matching pattern '*_*_*.wcnf' in {args.input_dir}")
        return

    records = []
    for filepath in files:
        filename = os.path.basename(filepath)
        parts = filename.split("_")
        solver = parts[0]
        num = parts[1]

        # notify which file is being processed
        print(f"Processing {filename}...")

        # run compare.py and capture output
        try:
            proc = subprocess.run(
                ["/usr/local/scratch/paxiant/MaxSAT-Fuzzer/compare.py", "--timeout", "10", filepath],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            output = proc.stdout
        except Exception as e:
            output = str(e)

        # search for error line and code
        pattern_error = rf"c {re.escape(solver)} with ERROR CODE\s+([0-9]+)\s+and"
        match = re.search(pattern_error, output)
        if match:
            error_code = match.group(1)
            line = next(
                (l for l in output.splitlines() if f"c {solver} with ERROR CODE" in l),
                ""
            )
        else:
            error_code = "XXX"
            line = ""

        records.append({
            "solver": solver,
            "num": num,
            "error_code": error_code,
            "message": line
        })

    # write detailed results to CSV
    with open(args.output, "w", newline="") as csvfile:
        fieldnames = ["solver", "num", "error_code", "message"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    print(f"Wrote {len(records)} detailed records to {args.output}")

    # build summary per unique solver/num
    summary = {}
    for rec in records:
        key = (rec["solver"], rec["num"])
        solver, num = key
        code = rec["error_code"]
        # initialize counts
        if key not in summary:
            summary[key] = {"total": 0, "repeat_fault": 0, "repeat_with_big": 0}
        summary[key]["total"] += 1
        # try parse code
        try:
            code_int = int(code)
        except ValueError:
            continue
        # exact repeat
        if code_int == int(num):
            summary[key]["repeat_fault"] += 1
        # repeat with big numbers (num + 1000)
        if code_int == int(num) + 1000:
            summary[key]["repeat_with_big"] += 1

    # write summary CSV
    summary_file = os.path.splitext(args.output)[0] + "_summary.csv"
    with open(summary_file, "w", newline="") as csvfile:
        fieldnames = ["solver", "num", "total_instances", "repeat_fault_count", "repeat_with_big_count"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for (solver, num), counts in sorted(summary.items()):
            writer.writerow({
                "solver": solver,
                "num": num,
                "total_instances": counts["total"],
                "repeat_fault_count": counts["repeat_fault"],
                "repeat_with_big_count": counts["repeat_with_big"]
            })
    print(f"Wrote {len(summary)} summary records to {summary_file}")

if __name__ == "__main__":
    main()

