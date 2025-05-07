#!/usr/bin/env python3

import lzma
import os
import sys
import csv
import re

# Your search lines
bigger = "c SumOfWeightsi: > UINT32"
smaller = "c SumOfWeightsi: < UINT32"
sum_of_weights_pattern = re.compile(r"c SumOfWeights\.: (\d+)")  # Regex to extract the number

# Check for required folder argument
if len(sys.argv) < 2:
    print("Usage: ./check_lines.py <folder-with-xz-files>")
    sys.exit(1)

folder = os.path.abspath(sys.argv[1])
parent_folder = os.path.dirname(folder)
output_csv = os.path.join(parent_folder, "biggerSmaller.csv")

# Find all .xz files in the folder (recursive)
xz_files = []
for root, _, files in os.walk(folder):
    for name in files:
        if name.endswith('.xz'):
            xz_files.append(os.path.join(root, name))

results = []

for file in xz_files:
    found_bigger = False
    found_smaller = False
    always_bigger_2_62 = True  # Assume true until proven otherwise
    try:
        with lzma.open(file, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Check for "bigger" and "smaller" conditions
                if not found_bigger and bigger in line:
                    found_bigger = True
                if not found_smaller and smaller in line:
                    found_smaller = True

                # Check for "bigger 2^62" condition
                match = sum_of_weights_pattern.search(line)
                if match:
                    number = int(match.group(1))
                    if number <= 2**62:
                        always_bigger_2_62 = False  # If any number is <= 2^62, set to False

                if found_bigger and found_smaller and not always_bigger_2_62:
                    break
    except Exception as e:
        print(f"Error reading {file}: {e}")
        continue

    # Extract Solver and Bug from the filename
    match = re.search(r'/([^/_]+(?:-[^/_]+)?)_(\d+)\.log\.xz$', file)
    if match:
        solver, bug = match.group(1), match.group(2)
        results.append({
            "Solver": solver,
            "Bug": bug,
            "bigger": found_bigger,
            "smaller": found_smaller,
            "bigger 2^62": always_bigger_2_62
        })
    else:
        print(f"Could not extract Solver and Bug from filename: {file}")

# Write to CSV
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ["Solver", "Bug", "bigger", "smaller", "bigger 2^62"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

# Also print to terminal
for row in results:
    print(f"{row['Solver']}_{row['Bug']}: bigger={row['bigger']}, smaller={row['smaller']}, bigger 2^62={row['bigger 2^62']}")

print(f"\nResults written to: {output_csv}")

