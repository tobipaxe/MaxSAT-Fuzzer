#!/usr/bin/env python3
import re
import csv
import sys
import os
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def generate_bug_stats(log_text, stats_matches, output_dir):
    """Extract bug details and write bugs_stats.csv."""
    bugs_pattern = re.compile(r"Solver\(Bug\):Count/1\.Time\s+:\s*(.+)")
    unique_bugs_pattern = re.compile(r"Unique Solver\(Bug\):Count/1T:\s*(.+)")
    bug_descriptions_pattern = re.compile(r"==== Bug Descriptions =============================\n(.+?)(?=\n={3,}|\Z)", re.DOTALL)
    
    bug_details = {}
    has_unique = {}
    for name, content in stats_matches:
        bug_details[name] = []
        unique_bugs = set()
        has_unique[name] = False
        if unique_match := unique_bugs_pattern.search(content):
            has_unique[name] = True
            for ub in unique_match.group(1).split(","):
                try:
                    solver_bug, count_time = ub.strip().split(":")
                    solver, bug = re.match(r"(\w+)\((\d+)\)", solver_bug.strip()).groups()
                    count, first_time = count_time.split('/')
                    bug_details[name].append({
                        'Solver': solver,
                        'Bug': bug,
                        'Count': count,
                        'First Occurence': first_time,
                        'Unique': 'yes'
                    })
                    unique_bugs.add((solver, bug))
                except Exception:
                    continue
        if bug_match := bugs_pattern.search(content):
            for entry in bug_match.group(1).split(","):
                try:
                    solver_bug, count_time = entry.strip().split(":")
                    solver, bug = re.match(r"(\w+)\((\d+)\)", solver_bug.strip()).groups()
                    if (solver, bug) not in unique_bugs:
                        count, first_time = count_time.split('/')
                        unique_mark = 'no' if has_unique[name] else ''
                        bug_details[name].append({
                            'Solver': solver,
                            'Bug': bug,
                            'Count': count,
                            'First Occurence': first_time,
                            'Unique': unique_mark
                        })
                except Exception:
                    continue
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
    out_path = os.path.join(output_dir, 'bugs_stats.csv')
    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        headers, subheaders, columns_count = [], [], {}
        for name in bug_details:
            if has_unique.get(name, False):
                columns_count[name] = 5
                headers.extend([name]*5)
                subheaders.extend(['Solver', 'Bug', 'Count', 'First Occurence', 'Unique'])
            else:
                columns_count[name] = 4
                headers.extend([name]*4)
                subheaders.extend(['Solver', 'Bug', 'Count', 'First Occurence'])
        writer.writerow(headers)
        writer.writerow(subheaders)
        max_len = max(len(bug_details[n]) for n in bug_details)
        for i in range(max_len):
            row = []
            for name in bug_details:
                cols = columns_count[name]
                if i < len(bug_details[name]):
                    bug = bug_details[name][i]
                    row.extend([bug['Solver'], bug['Bug'], bug['Count'], bug['First Occurence']])
                    if cols == 5:
                        row.append(bug['Unique'])
                else:
                    row.extend(['']*cols)
            writer.writerow(row)
        bug_descriptions_match = bug_descriptions_pattern.search(log_text)
        if bug_descriptions_match:
            bug_descriptions = bug_descriptions_match.group(1).strip().split('\n')
            writer.writerow([])
            writer.writerow(['Bug Descriptions'])
            for description in bug_descriptions:
                if m := re.match(r"Error (\d+): (.+)", description):
                    writer.writerow([m.group(1), m.group(2)])
    print("CSV file generated:", out_path)
    print("\tDetailed bug statistics with solver, bug counts, and first occurrence times are saved in bugs_stats.csv.")
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

    
    
def generate_bug_first_occurrence(bug_details, virtual_best, output_dir):
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
    sorted_solver_bug = sorted(all_solver_bug, key=lambda x: (x[0], int(x[1])))
    fuzzer_names = sorted([name for name in bug_details.keys() if name not in ('Overall', 'VirtualBest')])
    fuzzer_names = ['VirtualBest'] + fuzzer_names
    out_path = os.path.join(output_dir, 'bug_first_occurrence.csv')
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
    print("\tAggregated bug first occurrence times by solver and fuzzer are saved in bug_first_occurrence.csv.")

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
    """Create a CDF plot from the bugs_stats.csv file.
    Always ignore overall statistics and save the plot as fuzzer_cdf_plot.png in output_dir."""
    in_path = os.path.join(output_dir, 'bugs_stats.csv')
    out_png = os.path.join(output_dir, 'fuzzer_cdf_plot.png')
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
    plt.figure(figsize=(10, 6))
    fuzzer_data = []
    for col in columns_time:
        times = pd.to_numeric(data[col], errors='coerce').dropna().sort_values()
        if times.empty:
            continue
        cdf = np.arange(1, len(times)+1)
        label = col.replace('_First Occurence', '')
        fuzzer_data.append((label, times, cdf))
    # Sort by number of bugs (descending)
    fuzzer_data.sort(key=lambda x: len(x[1]), reverse=True)
    for label, times, cdf in fuzzer_data:
        plt.step(times, cdf, label=label)
    # plt.xscale('log', base=2)
    # plt.xlabel('Time (seconds, log scale base 2)')
    # plt.xscale('log', base=10)
    # plt.xlabel('Time (seconds, log scale base 10)')
    plt.xscale('linear')
    plt.xlabel('Time (seconds)')

    plt.ylabel('Bugs Detected')
    plt.title('CDF of first Bug Occurences Over Time per Fuzzer')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.show()
    print("CDF plot generated:", out_png)
    print("\tA CDF plot showing the cumulative distribution of first bug occurrences over time per fuzzer is saved as fuzzer_cdf_plot.png.")


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
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_log_file>")
        sys.exit(1)
    log_file_path = sys.argv[1]
    output_dir = os.path.dirname(os.path.abspath(log_file_path))
    with open(log_file_path, 'r') as f:
        log_text = f.read()
    extract_data(log_text, output_dir)
    print("CSV files and plot generated in:", output_dir)