#!/usr/bin/env python3
import re
import csv
import os
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

# 40	to	1.6	Invalid Return Code of MaxSAT solver == 40
# 50	to	1.6	Invalid Return Code of MaxSAT solver == 50
# 134	to	1.1	Invalid Return Code of MaxSAT solver == 134
# 135	to	1.2	Invalid Return Code of MaxSAT solver == 135
# 136	to	1.3	Invalid Return Code of MaxSAT solver == 136
## 137    to	1.4	Invalid Return Code of MaxSAT solver == 137 -- did not occur
# 139	to	1.5	Invalid Return Code of MaxSAT solver == 139
# 501	to	3.2	POTENTIAL ERROR: TIMEOUT and MEMPEAK  Timeout and Memory peak (740748) is 100 times bigger than the median memory peak.
# 502	to	3.1	POTENTIAL ERROR: TIMEOUT is 100 times bigger than median time of all other solvers.
# 511	to	4.6	return value is 20 but s-status is not UNSATISFIABLE!
# 602	to	2.5	Hard clauses are SATISFIABLE, but solver states s UNSATISFIABLE.
# 603	to	2.4	Verifier returned, that hard clauses are UNSATISFIABLE but solver states otherwise.
# 605	to	4.1	s status line NOT in solver output.
# 607	to	2.6	Verifier returned, that given model is too small.
# 608	to	2.6	Verifier returned that given model is UNSATISAFIABLE.
# 609	to	4.1	s string in o-value (example): o 0c All SoftClauses are Satisfiable!
# 611	to	4.1	s OPTIMUM FOUND - but no o value given
# 613	to	2.3	The given o value is negative, probably because of an overflow.
# 650	to	2.2	MaxSAT solver o-values given by solver, model and the minimal o value are three different values.
# 651	to	2.3	MaxSAT o-value equals minimal o-value BUT given model has a bigger o-value.
# 652	to	2.3	MaxSAT solver o value is bigger than the o-value of its model which equals the minimal o-value.
# 653	to	2.3	MaxSAT solver o value is smaller than the o-value of its model, but the o-value of it's model equals the minimal o-value.
# 655	to	2.1	o-value of MaxSAT solver model equals o value of the solver but it is bigger than the minimal o-value.
# 656	to	2.3	The o-values are otherwise inconsistent.
# 701	to	4.2	No fault but the length of the model is at least 10x longer than the actual number of variables.
# 702	to	4.3	MaxSAT solver returned something in stderr.
# 703	to	4.3	MaxSAT solver had ERROR written in some form in stdout

# --- Embedded error-to-fault mapping ---
ERROR_MAP = {
    40: "1.6", 50: "1.6", 134: "1.1", 135: "1.2", 136: "1.3", 139: "1.5", 137: "1.4",
    501: "3.2", 502: "3.1", 511: "4.6", 602: "2.5", 603: "2.4",
    605: "4.1", 607: "2.6", 608: "2.6", 609: "4.1", 611: "4.1",
    613: "2.3", 650: "2.2", 651: "2.3", 652: "2.3", 653: "2.3",
    655: "2.1", 656: "2.3", 701: "4.2", 702: "4.3", 703: "4.3",
}

def generate_general_stats(log_text, output_dir):
    """Extract and write general stats to general_stats.csv."""
    stats_pattern = re.compile(r"={3,} (.+?) Stats ={3,}\n(.+?)(?=\n={3,}|\Z)", re.DOTALL)
    stats_matches = stats_pattern.findall(log_text)
    all_data = {}
    for name, content in stats_matches:
        data = {}
        for line in content.split('\n'):
            if match := re.match(r"Bugs: Exit Codes/Solvers\s+:\s*(\d+)/(\d+)", line):
                data['Exit Codes'] = match.group(1)
                data['Solvers'] = match.group(2)
            elif match := re.match(r"Total Executions(?: \(Script\))?\s+:\s*(\d+)", line):
                data['Executions'] = match.group(1)
            elif match := re.match(r"Time Avgerage/Total\s+:\s*([\d.]+)/([\d.]+)", line):
                data['Average Time'] = match.group(1)
                data['Total Time'] = match.group(2)
            elif match := re.match(r"Execution Time\s+:\s*([\d.]+)", line):
                data['Total Time'] = match.group(1)
            elif match := re.match(r"Number Threads\s+:\s*(\d+)", line):
                data['Threads'] = match.group(1)
            elif match := re.match(r"Errors Found\s+:\s*(\d+)", line):
                data['Errors Found'] = match.group(1)
            elif match := re.match(r"Bugs: Unique/All\s+:\s*(\d+)/(\d+)", line):
                data['Unique Bugs'] = match.group(1)
                data['Errors Found'] = match.group(2)
            elif line.startswith("Invalid return codes"):
                invalid_text = re.search(r"\{(.+?)\}", line)
                if invalid_text:
                    formatted = re.sub(r";\s*", ", ", invalid_text.group(1))
                    data['Invalid return codes'] = formatted.strip()
        all_data[name.strip()] = data
    if 'Average Time' not in all_data.get("Overall", {}):
        overall = all_data.get("Overall", {})
        if overall.get('Total Time') and overall.get('Executions'):
            overall['Average Time'] = str(round(float(overall['Total Time']) / int(overall['Executions']), 2))
    out_path = os.path.join(output_dir, 'general_stats.csv')
    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        headers = [''] + list(all_data.keys())
        writer.writerow(headers)
        keys = ['Exit Codes', 'Solvers', 'Executions', 'Average Time', 'Total Time', 'Threads', 'Errors Found', 'Unique Bugs', 'Invalid return codes']
        for key in keys:
            row = [key] + [all_data.get(name, {}).get(key, '') for name in all_data]
            writer.writerow(row)
    print("CSV file generated:", out_path)
    print("\tGeneral statistics (e.g. execution times, thread count) are saved in general_stats.csv.")
    return all_data, stats_matches

def generate_bug_stats(log_text, stats_matches, output_dir, apply_map=False):
    """Extract bug details and write bugs_stats(.csv), applying mapping if requested."""
    bugs_pattern = re.compile(r"Solver\(Bug\):Count/1\.Time\s+:\s*(.+)")
    unique_bugs_pattern = re.compile(r"Unique Solver\(Bug\):Count/1T:\s*(.+)")
    bug_descriptions_pattern = re.compile(
        r"==== Bug Descriptions =============================\n(.+?)(?=\n={3,}|\Z)",
        re.DOTALL
    )
    bug_details = {}
    has_unique = {}

    # Parse each stats block
    for name, content in stats_matches:
        bug_details[name] = []
        unique_bugs = set()
        has_unique[name] = False
        # Unique bugs block
        if ubm := unique_bugs_pattern.search(content):
            has_unique[name] = True
            for ub in ubm.group(1).split(','):
                try:
                    solver_bug, count_time = ub.strip().split(':')
                    solver, bug = re.match(r"(\w+)\((\d+)\)", solver_bug.strip()).groups()
                    count, first_time = count_time.split('/')
                    # Determine mapped bug for consistency
                    mapped = ERROR_MAP.get(int(bug), bug) if apply_map else bug
                    bug_details[name].append({
                        'Solver': solver,
                        'Bug': mapped,
                        'Count': count,
                        'First Occurence': first_time,
                        'Unique': 'yes'
                    })
                    unique_bugs.add((solver, mapped))
                except Exception:
                    continue
        # All bugs block
        if bm := bugs_pattern.search(content):
            for entry in bm.group(1).split(','):
                try:
                    solver_bug, count_time = entry.strip().split(':')
                    solver, bug = re.match(r"(\w+)\((\d+)\)", solver_bug.strip()).groups()
                    # Determine mapped bug consistently
                    mapped = ERROR_MAP.get(int(bug), bug) if apply_map else bug
                    # Skip if this solver/bug combo was already marked unique
                    if (solver, mapped) in unique_bugs:
                        continue
                    count, first_time = count_time.split('/')
                    unique_mark = 'no' if has_unique[name] else ''
                    bug_details[name].append({
                        'Solver': solver,
                        'Bug': mapped,
                        'Count': count,
                        'First Occurence': first_time,
                        'Unique': unique_mark
                    })
                except Exception:
                    continue

    # Aggregate duplicates when mapping is active
    if apply_map:
        for name_key, entries in bug_details.items():
            agg = {}
            for e in entries:
                key = (e['Solver'], e['Bug'])
                cnt = int(e['Count']) if e['Count'] else 0
                ft = float(e['First Occurence']) if e['First Occurence'] else float('inf')
                uq = (e['Unique'].lower() == 'yes')
                if key not in agg:
                    agg[key] = {'Count': cnt, 'First Occurence': ft, 'Unique': uq}
                else:
                    agg[key]['Count'] += cnt
                    if ft < agg[key]['First Occurence']:
                        agg[key]['First Occurence'] = ft
                    agg[key]['Unique'] = agg[key]['Unique'] and uq
            # rebuild entries
            new_list = []
            for (sol, bg), vals in agg.items():
                new_list.append({
                    'Solver': sol,
                    'Bug': bg,
                    'Count': str(vals['Count']),
                    'First Occurence': str(vals['First Occurence']),
                    'Unique': 'yes' if vals['Unique'] else 'no'
                })
            bug_details[name_key] = new_list

    # Compute virtual best occurrences
    virtual_best = {}
    for name, bugs in bug_details.items():
        if name == 'Overall':
            continue
        for bug in bugs:
            try:
                key = (bug['Solver'], bug['Bug'])
                t = float(bug['First Occurence'])
                if key not in virtual_best or t < virtual_best[key]:
                    virtual_best[key] = t
            except Exception:
                continue
    # Add VirtualBest row set
    bug_details['VirtualBest'] = []
    for (solver, bug), first_time in sorted(virtual_best.items()):
        bug_details['VirtualBest'].append({
            'Solver': solver,
            'Bug': bug,
            'Count': '',
            'First Occurence': str(first_time),
            'Unique': ''
        })
    has_unique['VirtualBest'] = False

    # Determine output filename with mapping suffix
    suffix = '_mapped' if apply_map else ''
    out_path = os.path.join(output_dir, f'bugs_stats{suffix}.csv')
    # Write CSV with multi-row header
    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        headers = []
        subheaders = []
        columns_count = {}
        for name in bug_details:
            if has_unique.get(name, False):
                columns_count[name] = 5
                headers.extend([name] * 5)
                subheaders.extend(['Solver', 'Bug', 'Count', 'First Occurence', 'Unique'])
            else:
                columns_count[name] = 4
                headers.extend([name] * 4)
                subheaders.extend(['Solver', 'Bug', 'Count', 'First Occurence'])
        writer.writerow(headers)
        writer.writerow(subheaders)

        max_rows = max(len(bug_details[n]) for n in bug_details)
        for i in range(max_rows):
            row = []
            for name in bug_details:
                cols = columns_count[name]
                if i < len(bug_details[name]):
                    entry = bug_details[name][i]
                    cells = [entry['Solver'], entry['Bug'], entry['Count'], entry['First Occurence']]
                    if cols == 5:
                        cells.append(entry['Unique'])
                else:
                    cells = [''] * cols
                row.extend(cells)
            writer.writerow(row)

        # Append bug descriptions if present
        if bdm := bug_descriptions_pattern.search(log_text):
            writer.writerow([])
            writer.writerow(['Bug Descriptions'])
            for line in bdm.group(1).strip().split('\n'):
                if m := re.match(r"Error (\d+): (.+)", line):
                    writer.writerow([m.group(1), m.group(2)])

    print("CSV file generated:", out_path)
    print(f"\tDetailed bug statistics saved in bugs_stats{suffix}.csv.")
    return bug_details, virtual_best

def generate_fuzzer_stats(log_text, output_dir):
    """Extract per-fuzzer stats (ignoring DeltaDebugger) and write to fuzzer_stats.csv.
    Ensures for rows like '32BitNumber' and 'Satisfiable', the percentage is in one column."""
    fuzzer_specific_pattern = re.compile(
        r"^=+\s*(?P<fuzzer>(?!DeltaDebugger)\S+)\s+Stats\s*=+\s*\n"
        r"(?P<block>.*?)(?=^=+\s*(?:DeltaDebugger|\S+\s+Stats)\s*=+|\Z)",
        re.DOTALL | re.MULTILINE
    )
    table_pattern = re.compile(
        r"^Stat\s+Min\s+Max\s+Avg\s+Percentage\s*\n(?P<stats>(?:^.+\n)+?)"
        r"(?=^\s*$|^(?:=+|\S))",
        re.DOTALL | re.MULTILINE
    )
    fuzzer_matches = list(fuzzer_specific_pattern.finditer(log_text))
    out_path = os.path.join(output_dir, 'fuzzer_stats.csv')
    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['Fuzzer', 'Stat', 'Min', 'Max', 'Avg', 'Percentage', 'Additional Info'])
        for fuzzer_match in fuzzer_matches:
            fuzzer_name = fuzzer_match.group('fuzzer').strip()
            block = fuzzer_match.group('block')
            table_match = table_pattern.search(block)
            if table_match:
                stats_table = table_match.group('stats').strip().split('\n')
                for line in stats_table:
                    tokens = re.split(r"\s+", line.strip())
                    # Ensure tokens has at least 6 elements; pad if not.
                    while len(tokens) < 6:
                        tokens.append('')
                    stat = tokens[0]
                    if stat == "32BitNumber":
                        # For 32BitNumber, remove unwanted "are"
                        min_val = ""
                        max_val = ""
                        avg_val = ""
                        percentage = tokens[1]
                        additional_info = tokens[2] + " " + " ".join(tokens[4:])
                    elif stat == "Satisfiable":
                        # For Satisfiable, clear numeric columns, take the real percentage from token[1]
                        # and set additional info as all tokens from index 2 joined (yielding "% of the hard clauses are satisfiable")
                        min_val = ""
                        max_val = ""
                        avg_val = ""
                        percentage = tokens[1]
                        additional_info = " ".join(tokens[2:])
                    else:
                        min_val = tokens[1]
                        max_val = tokens[2]
                        avg_val = tokens[3]
                        percentage = tokens[4]
                        additional_info = " ".join(tokens[5:])
                        if max_val == "%" or percentage == "%":
                            percentage = min_val
                            min_val = ""
                            max_val = ""
                            additional_info = " ".join(tokens[2:])
                    writer.writerow([
                        fuzzer_name,
                        stat,
                        min_val,
                        max_val,
                        avg_val,
                        percentage,
                        additional_info
                    ])
    print("CSV file generated:", out_path)
    print("\tFuzzer-specific statistics (excluding DeltaDebugger) are saved in fuzzer_stats.csv.")

    
    
def generate_bug_first_occurrence(bug_details, virtual_best, output_dir, apply_map=False):
    """Collect bug first occurrence times per solver/bug and write bug_first_occurrence.csv."""
    all_solver_bug = set()
    fuzzer_times = defaultdict(dict)
    for fuzzer_name, bugs in bug_details.items():
        if fuzzer_name == 'Overall': continue
        for bug in bugs:
            solver_bug = (bug['Solver'], bug['Bug'])
            all_solver_bug.add(solver_bug)
            fuzzer_times[solver_bug][fuzzer_name] = bug['First Occurence']
    for solver_bug, time in virtual_best.items():
        fuzzer_times[solver_bug]['VirtualBest'] = str(time)
    sorted_solver_bug = sorted(all_solver_bug, key=lambda x: (x[0], x[1]))
    fuzzer_names = sorted([name for name in bug_details.keys() if name not in ('Overall', 'VirtualBest')])
    fuzzer_names = ['VirtualBest'] + fuzzer_names
    # Determine output filename with mapping suffix
    suffix = '_mapped' if apply_map else ''
    out_path = os.path.join(output_dir, f'bug_first_occurrence{suffix}.csv')
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        header = ['Solver', 'Bug'] + fuzzer_names
        writer.writerow(header)
        for solver, bug in sorted_solver_bug:
            row = [solver, bug]
            for fuzzer in fuzzer_names:
                row.append(fuzzer_times.get((solver, bug), {}).get(fuzzer, ''))
            writer.writerow(row)
    print("CSV file generated:", out_path)
    print(f"\tAggregated bug first occurrence times by solver and fuzzer are saved in bug_first_occurrence{suffix}.csv")

def generate_fuzzer_comparison(bug_details, output_dir):
    """Build a comparison matrix between fuzzers and write fuzzer_comparison.csv."""
    all_fuzzers = [name for name in bug_details.keys() if name != 'Overall']
    bugs_found = {fuzzer: set((bug['Solver'], bug['Bug']) for bug in bug_details[fuzzer]) for fuzzer in all_fuzzers}
    fuzzer_totals = {fuzzer: len(bugs_found[fuzzer]) for fuzzer in all_fuzzers}
    ordered_fuzzers = [f for f in all_fuzzers if f not in ('VirtualBest', 'DeltaDebugger')]
    ordered_fuzzers = sorted(ordered_fuzzers, key=lambda f: fuzzer_totals[f], reverse=True)
    final_fuzzers = ['VirtualBest', 'DeltaDebugger'] + [f for f in ordered_fuzzers if f not in ('VirtualBest', 'DeltaDebugger')]
    comparison_matrix = {}
    for fuzzer_row in final_fuzzers:
        comparison_matrix[fuzzer_row] = {}
        row_set = bugs_found.get(fuzzer_row, set())
        for fuzzer_col in final_fuzzers:
            if fuzzer_row == fuzzer_col:
                comparison_matrix[fuzzer_row][fuzzer_col] = ''
            else:
                col_set = bugs_found.get(fuzzer_col, set())
                only_in_row = row_set - col_set
                comparison_matrix[fuzzer_row][fuzzer_col] = len(only_in_row)
    out_path = os.path.join(output_dir, 'fuzzer_comparison.csv')
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow([''] + ['Total Bugs'] + final_fuzzers)
        for fuzzer_row in final_fuzzers:
            total = fuzzer_totals.get(fuzzer_row, 0)
            row = [fuzzer_row, total] + [comparison_matrix[fuzzer_row][fuzzer_col] for fuzzer_col in final_fuzzers]
            writer.writerow(row)
    print("CSV file generated:", out_path)
    print("\tA matrix comparing bug detection differences between fuzzers is saved in fuzzer_comparison.csv.")

def generate_cdf_plot(output_dir):
    """Create both linear and logarithmic CDF plots from the bugs_stats.csv file.
    Always ignore overall statistics and save the plots as fuzzer_cdf_plot_linear.png and fuzzer_cdf_plot_log.png in output_dir."""
    in_path = os.path.join(output_dir, 'bugs_stats.csv')
    out_png_linear = os.path.join(output_dir, 'fuzzer_cdf_plot_linear.png')
    out_png_log = os.path.join(output_dir, 'fuzzer_cdf_plot_log.png')

    # Predefined color mapping for fuzzers
    FUZZER_COLORS = {
        "VirtualBest": "blue",
        "DeltaDebugger": "orange",
        "PaxianPy": "green",
        "PaxianPyTiny": "red",
        "PaxianPySmall": "purple",
        "Paxian": "brown",
        "Manthey": "magenta",
        "Pollitt": "teal",
        "Soos": "olive"
    }

    # Load CSV with two header rows
    try:
        data = pd.read_csv(in_path, delimiter=';', header=[0, 1])
    except Exception as e:
        print("Error loading", in_path, ":", e)
        return

    # Flatten multi-index columns
    data.columns = ['{}_{}'.format(col[0], col[1]).strip('_') for col in data.columns.values]

    # Extract columns related to "First Occurence" and ignore overall stats
    columns_time = [col for col in data.columns if 'First Occurence' in col and not col.startswith('Overall_')]

    fuzzer_data = []
    for col in columns_time:
        times = pd.to_numeric(data[col], errors='coerce').dropna().sort_values()
        if times.empty:
            continue
        cdf = np.arange(1, len(times) + 1)
        label = col.replace('_First Occurence', '')
        fuzzer_data.append((label, times, cdf))

    # Sort by number of bugs (descending)
    fuzzer_data.sort(key=lambda x: len(x[1]), reverse=True)

    # Generate Linear Plot
    plt.figure(figsize=(10, 6))
    for label, times, cdf in fuzzer_data:
        color = FUZZER_COLORS.get(label, None)  # Use predefined color or default
        plt.step(times, cdf, label=label, color=color)
    plt.xscale('linear')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Bugs Detected')
    plt.title('CDF of First Bug Occurrences Over Time per Fuzzer (Linear Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(out_png_linear)
    plt.close()
    print("Linear CDF plot generated:", out_png_linear)

    # Generate Logarithmic Plot
    plt.figure(figsize=(10, 6))
    for label, times, cdf in fuzzer_data:
        color = FUZZER_COLORS.get(label, None)  # Use predefined color or default
        plt.step(times, cdf, label=label, color=color)
    plt.xscale('log', base=2)
    plt.xlabel('Time (seconds, log scale base 2)')
    plt.ylabel('Bugs Detected')
    plt.title('CDF of First Bug Occurrences Over Time per Fuzzer (Logarithmic Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(out_png_log)
    # plt.show()
    plt.close()
    print("Logarithmic CDF plot generated:", out_png_log)

# === NEW FUNCTION: Generate comprehensive fuzzer summary table ===
def generate_fuzzer_summary_table(output_dir):
    import pandas as pd
    import os

    df = pd.read_csv(os.path.join(output_dir, 'fuzzer_stats.csv'), delimiter=';')
    
    summary_rows = []
    fuzzers = df['Fuzzer'].unique()
    
    for fuzzer in fuzzers:
        row = {}
        row['Fuzzer'] = fuzzer
        df_fuzzer = df[df['Fuzzer'] == fuzzer]
        
        # Has Hard Clauses: computed as 100 - HardClauses percentage
        df_hard = df_fuzzer[df_fuzzer['Stat'] == "HardClauses"]
        if not df_hard.empty:
            try:
                hard_pct = float(str(df_hard.iloc[0]['Percentage']).replace('%', '').strip())
                computed = 100 - hard_pct
                row['Has Hard Clauses'] = f"{computed:.2f}%"
            except Exception:
                row['Has Hard Clauses'] = ''
            # Also detail numbers for Hard Clauses
            try:
                min_val = int(round(float(df_hard.iloc[0]['Min'])))
                max_val = int(round(float(df_hard.iloc[0]['Max'])))
                avg_val = int(round(float(df_hard.iloc[0]['Avg'])))
                row['Hard Clauses (min–max, avg)'] = f"{min_val}-{max_val}, {avg_val}"
            except Exception:
                row['Hard Clauses (min–max, avg)'] = ''
        else:
            row['Has Hard Clauses'] = ''
            row['Hard Clauses (min–max, avg)'] = ''
            
        # Has Soft Clauses: computed as 100 - SoftClauses percentage
        df_soft = df_fuzzer[df_fuzzer['Stat'] == "SoftClauses"]
        if not df_soft.empty:
            try:
                soft_pct = float(str(df_soft.iloc[0]['Percentage']).replace('%', '').strip())
                computed = 100 - soft_pct
                row['Has Soft Clauses'] = f"{computed:.2f}%"
            except Exception:
                row['Has Soft Clauses'] = ''
            try:
                min_val = int(round(float(df_soft.iloc[0]['Min'])))
                max_val = int(round(float(df_soft.iloc[0]['Max'])))
                avg_val = int(round(float(df_soft.iloc[0]['Avg'])))
                row['Soft Clauses (min–max, avg)'] = f"{min_val}-{max_val}, {avg_val}"
            except Exception:
                row['Soft Clauses (min–max, avg)'] = ''
        else:
            row['Has Soft Clauses'] = ''
            row['Soft Clauses (min–max, avg)'] = ''
            
        # Objective 0: from BestOValue row
        df_obj = df_fuzzer[df_fuzzer['Stat'] == "BestOValue"]
        if not df_obj.empty:
            try:
                obj_pct = float(str(df_obj.iloc[0]['Percentage']).replace('%', '').strip())
                row['Objective 0'] = f"{obj_pct:.2f}%"
            except Exception:
                row['Objective 0'] = ''
        else:
            row['Objective 0'] = ''
        
        # Satisfiable Hard Clauses: directly from Satisfiable row (percentage)
        df_sat = df_fuzzer[df_fuzzer['Stat'] == "Satisfiable"]
        if not df_sat.empty:
            perc = str(df_sat.iloc[0]['Percentage']).strip()
            if not perc.endswith('%'):
                perc += '%'
            row['Satisfiable Hard Clauses'] = perc
        else:
            row['Satisfiable Hard Clauses'] = ''
            
        # 32 Bit Numbers: directly from 32BitNumber row (percentage)
        df_32 = df_fuzzer[df_fuzzer['Stat'] == "32BitNumber"]
        if not df_32.empty:
            perc = str(df_32.iloc[0]['Percentage']).strip()
            if not perc.endswith('%'):
                perc += '%'
            row['32 Bit Numbers'] = perc
        else:
            row['32 Bit Numbers'] = ''
            
        # Variables (min–max, avg):
        df_var = df_fuzzer[df_fuzzer['Stat'] == "Variables"]
        if not df_var.empty:
            try:
                min_val = int(round(float(df_var.iloc[0]['Min'])))
                max_val = int(round(float(df_var.iloc[0]['Max'])))
                avg_val = int(round(float(df_var.iloc[0]['Avg'])))
                row['Variables (min–max, avg)'] = f"{min_val}-{max_val}, {avg_val}"
            except Exception:
                row['Variables (min–max, avg)'] = f"{df_var.iloc[0]['Min']}-{df_var.iloc[0]['Max']}, {df_var.iloc[0]['Avg']}"
        else:
            row['Variables (min–max, avg)'] = ''
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows, columns=[
         'Fuzzer',
         'Has Hard Clauses',
         'Has Soft Clauses',
         'Objective 0',
         'Satisfiable Hard Clauses',
         '32 Bit Numbers',
         'Hard Clauses (min–max, avg)',
         'Soft Clauses (min–max, avg)',
         'Variables (min–max, avg)'
    ])
    
    summary_path = os.path.join(output_dir, 'fuzzer_summary.csv')
    summary_df.to_csv(summary_path, sep=';', index=False)
    print("Fuzzer summary CSV generated:", summary_path)

def extract_data(log_text, output_dir):
    all_data, stats_matches = generate_general_stats(log_text, output_dir)
    bug_details, virtual_best = generate_bug_stats(log_text, stats_matches, output_dir)
    generate_fuzzer_stats(log_text, output_dir)
    generate_bug_first_occurrence(bug_details, virtual_best, output_dir)
    generate_fuzzer_comparison(bug_details, output_dir)
    generate_fuzzer_summary_table(output_dir)
    generate_cdf_plot(output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract data from a log file and optionally map error codes to fault numbers.'
    )
    parser.add_argument(
        'input_log_file',
        help='Path to the input log file'
    )
    parser.add_argument(
        '--mapping',
        dest='apply_map',
        action='store_true',
        help='Apply embedded error-to-fault mapping when generating bug stats'
    )
    args = parser.parse_args()

    log_file_path = args.input_log_file
    output_dir = os.path.dirname(os.path.abspath(log_file_path))
    with open(log_file_path, 'r') as f:
        log_text = f.read()

    all_data, stats_matches = generate_general_stats(log_text, output_dir)
    bug_details, virtual_best = generate_bug_stats(
        log_text, stats_matches, output_dir, apply_map=args.apply_map
    )
    generate_fuzzer_stats(log_text, output_dir)
    generate_bug_first_occurrence(bug_details, virtual_best, output_dir, apply_map=args.apply_map)
    generate_fuzzer_comparison(bug_details, output_dir)
    generate_fuzzer_summary_table(output_dir)
    generate_cdf_plot(output_dir)
    print(f"CSV files and plots generated in: {output_dir}")