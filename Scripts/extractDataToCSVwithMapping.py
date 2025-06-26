#!/usr/bin/env python3
import re
import csv
import os
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob, subprocess

TAG = ""   # will be set from args

# 40	to	1.6	Invalid Return Code of MaxSAT solver == 40
# 50	to	1.6	Invalid Return Code of MaxSAT solver == 50
# 134	to	1.1	Invalid Return Code of MaxSAT solver == 134
# 135	to	1.2	Invalid Return Code of MaxSAT solver == 135
# 136	to	1.3	Invalid Return Code of MaxSAT solver == 136
# 1  :  to  1.6 Invalid Return Code of MaxSAT solver == 1
# 8  :  to  1.6 Invalid Return Code of MaxSAT solver == 8
# 13 :  to  1.6 Invalid Return Code of MaxSAT solver == 13
# 38 :  to  1.6 Invalid Return Code of MaxSAT solver == 38
# 43 :  to  1.6 Invalid Return Code of MaxSAT solver == 43
# 84 :  to  1.6 Invalid Return Code of MaxSAT solver == 84
# 160:  to  1.6 Invalid Return Code of MaxSAT solver == 160
# 174:  to  1.6 Invalid Return Code of MaxSAT solver == 174
# 179:  to  1.6 Invalid Return Code of MaxSAT solver == 179
# 130:  to  1.6 Invalid Return Code of MaxSAT solver == 130
# 133:  to  1.6 Invalid Return Code of MaxSAT solver == 133
# 137    to	1.4	Invalid Return Code of MaxSAT solver == 137 -- did not occur
# 139	to	1.5	Invalid Return Code of MaxSAT solver == 139
# 501	to	3.2	POTENTIAL ERROR: TIMEOUT and MEMPEAK  Timeout and Memory peak (740748) is 100 times bigger than the median memory peak.
# 502	to	3.1	POTENTIAL ERROR: TIMEOUT is 100 times bigger than median time of all other solvers.
# 511	to	4.6	return value is 20 but s-status is not UNSATISFIABLE!
# 602	to	2.5	Hard clauses are SATISFIABLE, but solver states s UNSATISFIABLE.
# 603	to	2.4	Verifier returned, that hard clauses are UNSATISFIABLE but solver states otherwise.
# 605	to	4.1	s status line NOT in solver output.
# 606	to	4.4	Solver status = UNKNOWN Unexpected result in the status line.
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
    1: "1.6", 8: "1.6", 13: "1.6", 38: "1.6", 40: "1.6", 43: "1.6", 
    50: "1.6", 84: "1.6", 160: "1.6", 174: "1.6", 179: "1.6",
    130: "1.6", 133: "1.6",
    134: "1.1", 135: "1.2", 136: "1.3", 139: "1.5", 137: "1.4",
    501: "3.2", 502: "3.1", 511: "4.6", 602: "2.5", 603: "2.4",
    605: "4.1", 606: "4.4", 607: "2.6", 608: "2.6", 609: "4.1", 611: "4.1",
    613: "2.3", 650: "2.2", 651: "2.3", 652: "2.3", 653: "2.3",
    655: "2.1", 656: "2.3", 701: "4.2", 702: "4.3", 703: "4.3",
}

SOLVER_NAME_MAP = {
    #Anytime23
    "NS-MS": "NoSAT-MaxSAT",
    "NSMS": "NoSAT-MaxSAT",
    "LOAND": "LOANDRA",
    "towiG": "tt-owi-Glucose41",
    #"towiG": "tt-open-wbo-inc-Glucose4_1",
    "towiI": "tt-owi-IntelSATSolver",
    #"towiI": "tt-open-wbo-inc-IntelSATSolver",
    "NuWcB": "NuWLS-c-Band",
    # "NuWcB": "NuWLS-c_band",
    "NuWcF": "NuWLS-c-FPS",
    "NuWcs": "NuWLS-c-static",
    #"NuWcs": "NuWLS-c_static",
    #Exact22
    "CMSCP": "Cashwmaxsat-CP",
    "CMSP": "Cashwmaxsat-Plus",
    "UMSS": "UWrMaxSat-SCIP",
    "WMCDCL": "WMaxCDCL",
    "WMCDCLBA": "WMaxCDCL-BandAll",
    "UWM": "UWrMaxSat",
    "EMS": "EvalMaxSAT",
    "MHS22": "MSE22-MaxHS",
    "CGSS": "MSE22-CGSS",
    "Exact": "MSE22-Exact",
    #Exact23
    # "WMCDCL": "WMaxCDCL", # already in Exact22, but equal
    "WMC61": "WMaxCDCL-S6-HS12",
    "WMC99": "WMaxCDCL-S9-HS9",
    "EMSSC": "EvalMaxSAT-SCIP",
    "EMS23": "EvalMaxSAT",
    "CGSS2": "CGSS2",
    "CGS2S": "CGSS2-SCIP",
    "CHMCP": "CASHWMAXSAT-CorePlus",
    "CHCPm": "CASHWMAXSAT-CorePlus-m",
    "PacMP": "Pacose-MaxPre2",
    "Pacos": "Pacose",
    #Exact24
    "CASDI6S": "CASHWMaxSAT-DisjCad-S6",
    "CASDI9S": "CASHWMaxSAT-DisjCad-S9",
    "CASDI01": "CASHWMaxSAT-DisjCom-S6",
    "CASDI02": "CASHWMaxSAT-DisjCom-S9",
    "EVA": "EvalMaxSAT",
    "EVASB": "EvalMaxSAT-SBVA",
    #"EVASBSA": "EvalMaxSAT-SBVA-saveCores",
    "EVASBSA": "EvalMaxSAT-SBVA-sC",
    "EXA": "Exact",
    "PAC": "Pacose",
    "PAC01": "PacoseMP2",
    "UWRSC": "UWrMaxSat-SCIP",
    "CGSABCG": "cgss_abst_cg",
    "CGSDE": "cgss_default",
    "WMA": "wmaxcdcl",
    "WMA1200": "wmaxcdcl-owbo1200",
    #"WMA1200": "wmaxcdcl-openwbo1200",
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
    out_path = os.path.join(output_dir, f'general_stats_{TAG}.csv')
    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        headers = [''] + list(all_data.keys())
        writer.writerow(headers)
        keys = ['Exit Codes', 'Solvers', 'Executions', 'Average Time', 'Total Time', 'Threads', 'Errors Found', 'Unique Bugs', 'Invalid return codes']
        for key in keys:
            row = [key] + [all_data.get(name, {}).get(key, '') for name in all_data]
            writer.writerow(row)
    print("CSV file generated:", out_path)
    print(f"\tGeneral statistics (e.g. execution times, thread count) are saved in general_stats_{TAG}.csv.")
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
                    # Replace solver name if it exists in the map
                    solver = SOLVER_NAME_MAP.get(solver, solver)
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
                    # Replace solver name if it exists in the map
                    solver = SOLVER_NAME_MAP.get(solver, solver)
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
    out_path = os.path.join(output_dir, f'bugs_stats{suffix}_{TAG}.csv')
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
    print(f"\tDetailed bug statistics saved in bugs_stats{suffix}_{TAG}.csv.")

    # === NEW: Load minimization summary if present ===
    minim_suffix = f'minimized_tested_again_summary{suffix}_{TAG}.csv'
    minim_path = os.path.join(output_dir, minim_suffix)
    minim_map = {}  # (solver,bug) -> flags string
    if os.path.exists(minim_path):
        import pandas as _pd
        df_min = _pd.read_csv(minim_path, sep=';')
        for _, r in df_min.iterrows():
            sol = r['solver']
            bug = str(r['mapped_num'])
            small = int(r['repeat_fault_count']) > 0
            big = int(r['repeat_with_big_count']) > 0
            flag = ''
            if small:
                flag += 'm'
            if big and not small:
                flag += 'M'
            minim_map[(sol, bug)] = flag
    


    # === NEW: Load bigger/smaller overview if present ===
    bigger_suffix = f'bigger_smaller{suffix}_{TAG}.csv'
    bigger_path = os.path.join(output_dir, bigger_suffix)
    bigger_map = {}  # (solver,bug) -> dict of flags
    if os.path.exists(bigger_path):
        import pandas as _pd2
        df_bs = _pd2.read_csv(bigger_path, sep=';')
        for _, r in df_bs.iterrows():
            sol = r['Solver'].lower().replace('-', '').replace('_', '')
            bug = str(r['Bug'])
            bigger_map[(sol, bug)] = {
                'smaller': str(r['smaller']).lower() == 'true',
                'bigger':  str(r['bigger']).lower()  == 'true',
                'bigger2': str(r['bigger 2^62']).lower() == 'true'
            }
    #print(bigger_map)

    # === Solver-Bug Matrix ===
    solver_bug_matrix_path = os.path.join(output_dir, f'solver_bug_matrix{suffix}_{TAG}.csv')
    overall_bugs = bug_details.get('Overall', [])
    delta_unique = {
        (b['Solver'], b['Bug'])
        for b in bug_details.get('DeltaDebugger', [])
        if b.get('Unique','').lower()=='yes'
    }
    solvers = sorted({b['Solver'] for b in overall_bugs})
    bugs    = sorted({b['Bug']    for b in overall_bugs})

    with open(solver_bug_matrix_path, 'w', newline='') as mf:
        writer = csv.writer(mf, delimiter=';')
        writer.writerow(['Solver \\ Bug'] + bugs)
        for sol in solvers:
            row = [sol]
            for bug in bugs:
                cell = ''
                if any(b['Solver'] == sol and b['Bug'] == bug for b in overall_bugs):
                    # pick letter by bigger/smaller/h flags
                    bs = bigger_map.get(((sol.replace('-', '').replace('_', '')).lower(), bug), {})
                    #bs = bigger_map.get((sol, bug), {})
                    if bs.get('bigger2', False):
                        #assert bs.get('bigger', False) and not bs.get('smaller', False)
                        letter = 'H'
                    elif bs.get('smaller', False):
                        #assert not bs.get('bigger', False) and not bs.get('bigger2', False)
                        letter = 'S'
                    elif bs.get('bigger', False):
                        #assert not bs.get('bigger2', False) and not bs.get('smaller', False)
                        letter = 'B'
                    else:
                        #assert not bs.get('bigger', False) and not bs.get('bigger2', False) and bs.get('smaller', True)
                        letter = 'Q'
                    print(f"Letter: {letter} for {sol} {bug}, bigger: {bs.get('bigger', False)}, smaller: {bs.get('smaller', False)}, bigger2: {bs.get('bigger2', False)}")
                    # delta‐unique marker
                    if (sol, bug) in delta_unique:
                        letter += 'D'
                    letter += minim_map.get((sol, bug), '')
                    # minimization marker
                    if 'm' in letter or 'M' in letter:
                        letter = letter.replace('m', '')
                        letter = letter.replace('M', '')
                    else:
                        letter += 'x'
                    
                    cell = letter
                row.append(cell)
            writer.writerow(row)

    print("CSV file generated:", solver_bug_matrix_path)
    print(f"\tSolver-Bug matrix saved in solver_bug_matrix{suffix}_{TAG}.csv.")

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
    out_path = os.path.join(output_dir, f'fuzzer_stats_{TAG}.csv')
    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(['Fuzzer', 'Stat', 'Min', 'Max', 'Avg', 'Percentage', 'Additional Info'])
        for fuzzer_match in fuzzer_matches:
            fuzzer_name = fuzzer_match.group('fuzzer').strip()
            block = fuzzer_match.group('block')

            # Ignore lines starting with "Solver" and the following lines
            filtered_block = []
            skip_lines = False
            for line in block.splitlines():
                if line.startswith("Solver"):
                    skip_lines = True
                elif skip_lines and line.strip() == "":
                    skip_lines = False
                elif not skip_lines:
                    filtered_block.append(line)
            block = "\n".join(filtered_block)

            table_match = table_pattern.search(block)
            if table_match:
                stats_table = table_match.group('stats').strip().split('\n')
                for line in stats_table:
                    tokens = re.split(r"\s+", line.strip())
                    # Ensure tokens have at least 6 elements; pad if not.
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
                        # and set additional info as all tokens from index 2 joined
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
    print(f"\tFuzzer-specific statistics (excluding DeltaDebugger) are saved in fuzzer_stats_{TAG}.csv.")

      
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
    out_path = os.path.join(output_dir, f'bug_first_occurrence{suffix}_{TAG}.csv')
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
    print(f"\tAggregated bug first occurrence times by solver and fuzzer are saved in bug_first_occurrence{suffix}_{TAG}.csv")

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
    out_path = os.path.join(output_dir, f'fuzzer_comparison_{TAG}.csv')
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow([''] + ['Total Bugs'] + final_fuzzers)
        for fuzzer_row in final_fuzzers:
            total = fuzzer_totals.get(fuzzer_row, 0)
            row = [fuzzer_row, total] + [comparison_matrix[fuzzer_row][fuzzer_col] for fuzzer_col in final_fuzzers]
            writer.writerow(row)
    print("CSV file generated:", out_path)
    print(f"\tA matrix comparing bug detection differences between fuzzers is saved in fuzzer_comparison_{TAG}.csv.")

def generate_cdf_plot(output_dir, apply_map=False):
    """Create both linear and logarithmic CDF plots from the bugs_stats.csv file.
    Always ignore overall statistics and save the plots as fuzzer_cdf_plot_linear and fuzzer_cdf_plot_log in multiple formats (e.g., PNG, PDF)."""
    
    if apply_map:
        mapping_suffix = '_mapped'
    else:
        mapping_suffix = ''
    in_path = os.path.join(output_dir, f'bugs_stats{mapping_suffix}_{TAG}.csv')
    # out_png_linear = os.path.join(output_dir, f'fuzzer_cdf_plot_linear{mapping_suffix}_{TAG}.png')
    out_pdf_linear = os.path.join(output_dir, f'fuzzer_cdf_plot_linear{mapping_suffix}_{TAG}.pdf')
    # out_png_log = os.path.join(output_dir, f'fuzzer_cdf_plot_log{mapping_suffix}_{TAG}.png')
    out_pdf_log = os.path.join(output_dir, f'fuzzer_cdf_plot_log{mapping_suffix}_{TAG}.pdf')

    # Predefined color mapping for fuzzers
    FUZZER_COLORS = {
        "VirtualBest": "orange",
        "DeltaDebugger": "blue",
        "PaxianPy": "lime",
        "PaxianPyTiny": "red",
        "PaxianPySmall": "gold",
        "Paxian": "deepskyblue",
        "Manthey": "chocolate",
        "Pollitt": "green",
        "Soos": "darkorchid"
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

    # Optional: assign priority to specific fuzzers
    fuzzer_priority = {
        'VirtualBest': 1,
        'DeltaDebugger': 2,
        # others get default priority = 999
    }

    # Assign default priority if not listed
    def get_priority(label):
        return fuzzer_priority.get(label, 999)

    # Sort by number of bugs (descending)
#    fuzzer_data.sort(key=lambda x: (len(x[1]), get_priority(x[0])), reverse=True)
    fuzzer_data.sort(key=lambda x: (-len(x[1]), get_priority(x[0])))


    # Adjust TAG for plot titles if it contains underscores
    plot_tag = TAG.replace('_', ' ') if '_' in TAG else TAG

    # Generate Linear Plot
    # plt.rcParams.update({'text.usetex': True, 'font.family': 'cmr10'})  # Enable LaTeX font rendering with Computer Modern
    # plt.rcParams.update({'text.usetex': True, 'font.family': 'serif'})
    # plt.rcParams.update({'text.usetex': True, 'font.family': 'Lucida'})  # Enable LaTeX font rendering with Lucida
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'text.latex.preamble': r'\usepackage{times}'
    })
    plt.figure(figsize=(10, 6))
    for label, times, cdf in fuzzer_data:
        color = FUZZER_COLORS.get(label, None)  # Use predefined color or default
        bug_count = len(times)  # Count the number of bugs found by the fuzzer
        plt.step(times, cdf, label=f"({bug_count} faults) {label}", color=color)
    plt.xscale('linear')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Failures Detected')
    plt.title(f'{TAG} CDF of First Fault Occurrences Over Time per Fuzzer (Linear Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    # plt.savefig(out_png_linear)
    plt.savefig(out_pdf_linear)
    plt.close()
    print("Linear CDF plot generated:", out_pdf_linear)


     # Custom ticks in seconds
    custom_ticks = [
        1e-1,           # 100 ms
        1,              # 1 second
        60,             # 1 minute
        3600,           # 1 hour
        86400,          # 1 day
        360000          # 100 hours
    ]
#    custom_labels = [
#        "1 ms",
#        "1 s",
#        "1 min",
#        "1 h",
#        "1 day",
#        "100 h"
#    ]
    custom_labels = [
        r"$100\,\mathrm{ms}$",
        r"$1\,\mathrm{s}$",
        r"$1\,\mathrm{min}$",
        r"$1\,\mathrm{h}$",
        r"$1\,\mathrm{day}$",
        r"$100\,\mathrm{h}$"
    ]


    # Generate Logarithmic Plot
    # plt.rcParams.update({'text.usetex': True, 'font.family': 'cmr10'})  # Enable LaTeX font rendering with Computer Modern
    # plt.rcParams.update({'text.usetex': True, 'font.family': 'cmr10'})  # Enable LaTeX font rendering with Computer Modern
    # plt.rcParams.update({'text.usetex': True, 'font.family': 'Lucida'})  # Enable LaTeX font rendering with Lucida
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'text.latex.preamble': r'\usepackage{times}',
        'font.size': 14
    })
    plt.figure(figsize=(10, 6))
    for label, times, cdf in fuzzer_data:
        color = FUZZER_COLORS.get(label, None)  # Use predefined color or default
        bug_count = len(times)  # Count the number of bugs found by the fuzzer
        plt.step(times, cdf, label=f"({bug_count} failures) {label}", color=color)
    plt.xscale('log', base=2)
    plt.xticks(custom_ticks, custom_labels)
    plt.xlim(2**-4, 2**19)  # Set x-axis range from 2^-3 to 2^19
    if apply_map:
        plt.ylim(0, 75)  # Set y-axis range from 0 to 75 for mapped data
    else:
        plt.ylim(0, 95)
    plt.xlabel('Time (log scale)')
    plt.ylabel('Failures Detected')
    plt.title(f'{plot_tag} CDF of First Failure Occurrences Over Time per Fuzzer (Logarithmic)')
    plt.legend(loc='upper left')  # Move legend to the top left corner
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    # plt.savefig(out_png_log)
    plt.savefig(out_pdf_log)
    plt.close()
    print("Logarithmic CDF plot generated:", out_pdf_log)

def generate_solver_timings(log_text, output_dir):
    """Extract solver timing/memory info *per fuzzer* and save to solver_timings_{TAG}.csv."""
    # Reuse the fuzzer-specific regex
    fuzzer_specific_pattern = re.compile(
        r"^=+\s*(?P<fuzzer>(?!DeltaDebugger)\S+)\s+Stats\s*=+\s*\n"
        r"(?P<block>.*?)(?=^=+\s*(?:DeltaDebugger|\S+\s+Stats)\s*=+|\Z)",
        re.DOTALL | re.MULTILINE
    )
    # Within each block look for the Solver… table
    solver_timings_pattern = re.compile(
        r"^\s*Solver\s+Min\s+Max\s+Average\s+Timeouts\s+Min\s+Max\s+Average\s*$\n"
        r"(?P<timings>(?:^\s*\S+\s+.+\n)+)",
        re.MULTILINE
    )

    out_path = os.path.join(output_dir, f'solver_timings_{TAG}.csv')
    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        # Write header with Fuzzer first
        writer.writerow([
            "Fuzzer", "Solver", "Min Time", "Max Time", "Avg Time", "Timeouts",
            "Min Memory", "Max Memory", "Avg Memory"
        ])

        for fm in fuzzer_specific_pattern.finditer(log_text):
            fuzzer = fm.group('fuzzer')
            block = fm.group('block')
            m = solver_timings_pattern.search(block)
            if not m:
                continue

            for line in m.group('timings').strip().splitlines():
                tokens = re.split(r"\s+", line.strip())
                if len(tokens) < 8:
                    continue
                solver_short = tokens[0]
                solver_long = SOLVER_NAME_MAP.get(solver_short, solver_short)
                min_t, max_t, avg_t, to = tokens[1:5]
                min_m, max_m, avg_m = tokens[5:8]
                writer.writerow([
                    fuzzer, solver_long,
                    min_t, max_t, avg_t, to,
                    min_m, max_m, avg_m
                ])

    print("CSV file generated:", out_path)
    print(f"\tPer-fuzzer solver timing/memory saved in solver_timings_{TAG}.csv.")

# === NEW FUNCTION: Generate comprehensive fuzzer summary table ===
def generate_fuzzer_summary_table(output_dir):
    import pandas as pd
    import os

    df = pd.read_csv(os.path.join(output_dir, f'fuzzer_stats_{TAG}.csv'), delimiter=';')
    
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
    
    summary_path = os.path.join(output_dir, f'fuzzer_summary_{TAG}.csv')
    summary_df.to_csv(summary_path, sep=';', index=False)
    print("CSV file generated:", summary_path)

def generate_fuzzer_overview_table(output_dir, apply_map=False):
    """Combine summary, general_stats, solver_timings and fuzzer_comparison into one CSV."""
    # choose mapped vs. unmapped where applicable
    suffix = '_mapped' if apply_map else ''

    # paths (apply same suffix)
    gen_path   = os.path.join(output_dir, f'general_stats_{TAG}.csv')
    sum_path   = os.path.join(output_dir, f'fuzzer_summary_{TAG}.csv')
    time_path  = os.path.join(output_dir, f'solver_timings_{TAG}.csv')
    comp_path  = os.path.join(output_dir, f'fuzzer_comparison_{TAG}.csv')

    # load
    general_df = pd.read_csv(gen_path, sep=';', index_col=0)
    summary_df = pd.read_csv(sum_path, sep=';')
    timings_df = pd.read_csv(time_path, sep=';')
    comp_df    = pd.read_csv(comp_path, sep=';', index_col=0)

    rows = []
    for _, s in summary_df.iterrows():
        f = s['Fuzzer']
        # from summary
        obj0 = s.get('Objective 0', '')
        sat = s.get('Satisfiable Hard Clauses', '')

        # from general_stats
        execs = general_df.at['Executions', f] if f in general_df.columns else ''
        avg_t = general_df.at['Average Time', f] if f in general_df.columns else ''

        # from solver_timings: max & count Timeouts%
        df_t = timings_df[timings_df['Fuzzer'] == f]
        if not df_t.empty:
            to_vals = df_t['Timeouts'].str.rstrip('%').astype(float)
            max_to = to_vals.max()
            max_to = f"{max_to:.3f}%"
            count_to = int((to_vals > 0.01).sum())
            count_to = str(count_to)
        else:
            max_to = ''
            count_to = ''

        # from fuzzer_comparison: Total Bugs
        total = comp_df.at[f, 'Total Bugs'] if f in comp_df.index else ''

        rows.append({
            'Fuzzer': f,
            'Total\nBugs': total,
            'Executions': execs,
            'Average Time\n(xx Solver)': avg_t,
            'max\nTimouts': max_to,
            'count\nTimouts': count_to,
            'Objective 0': obj0,
            'Satisfiable\nHardClauses': sat
        })

    overview_df = pd.DataFrame(rows, columns=[
        'Fuzzer',
        'Total\nBugs',
        'Executions',
        'Average Time\n(xx Solver)',
        'max\nTimouts',
        'count\nTimouts',
        'Objective 0',
        'Satisfiable\nHardClauses'
    ])
    overview_fname = f'fuzzer_overview{suffix}_{TAG}.csv'
    out_path = os.path.join(output_dir, overview_fname)
    overview_df.to_csv(out_path, sep=';', index=False)
    print("CSV file generated:", out_path)
    print(f"\tComprehensive fuzzer overview table combining general, summary, timing and comparison stats saved in {overview_fname}.")

def generate_fault_tests(input_dir, output_dir, apply_map=False):
    """Run compare.py on all .wcnf in input_dir, write detailed + summary CSVs."""
    pattern = os.path.join(input_dir, "*_*_*.wcnf")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No minimized faults in {input_dir}")
        return
    print(f"Found {len(files)} minimized faults in {input_dir}. Running compare.py for every file with timeout of 10 seconds for each solver, this may take a while...")

    suffix = '_mapped' if apply_map else ''
    detailed_csv = os.path.join(output_dir, f'minimized_tested_again_detailed{suffix}_{TAG}.csv')
    summary_csv = os.path.join(output_dir, f'minimized_tested_again_summary{suffix}_{TAG}.csv')

    records = []
    # --- detailed ---
    with open(detailed_csv, 'w', newline='') as detf:
        cols = ["solver", "num", "error_code", "message"]
        writer = csv.DictWriter(detf, fieldnames=cols)
        writer.writeheader()
        for path in files:
            fn = os.path.basename(path)
            solver_short, num, _ = fn.split("_", 2)
            solver = SOLVER_NAME_MAP.get(solver_short, solver_short)
            print(f"Testing {fn} ...")
            try:
                proc = subprocess.run(
                    ["../compare.py", "--timeout", "10", path],
                    cwd=os.path.dirname(__file__),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                out = proc.stdout
            except Exception as e:
                print(f"Error running compare.py on {fn}: {e}")
                out = str(e)

            m = re.search(rf"c {re.escape(solver_short)} with ERROR CODE\s+(\d+)", out)
            raw_code = m.group(1) if m else "XXX"
            # capture full line if present
            line = next((L for L in out.splitlines() if f"c {solver} with ERROR CODE" in L), "")
            # write raw_code here; mapping deferred to summary
            rec = {"solver": solver, "num": num, "error_code": raw_code, "message": line}
            records.append(rec)
            writer.writerow(rec)

    print("CSV file generated:", detailed_csv)
    print(f"\tDetailed fault-test results saved in {os.path.basename(detailed_csv)}")

    # --- summary ---
    summary = {}
    for r in records:
        key = (r["solver"], r["num"])
        if key not in summary:
            summary[key] = {"total": 0, "repeat_fault": 0, "repeat_with_big": 0}
        summary[key]["total"] += 1
        if r["error_code"] == r["num"]:
            summary[key]["repeat_fault"] += 1
        try:
            if int(r["error_code"]) == int(r["num"]) + 1000:
                summary[key]["repeat_with_big"] += 1
        except:
            pass

    # Build grouped summary rows, merging duplicate mapped_num if mapping is active
    summary_rows = []
    if apply_map:
        grouped = {}
        for (solver, num), cnt in summary.items():
            if num.isdigit():
                mapped = ERROR_MAP.get(int(num), num)
            else:
                mapped = num
            key = (solver, mapped)
            grp = grouped.setdefault(key, {"total": 0, "repeat_fault": 0, "repeat_with_big": 0})
            grp["total"] += cnt["total"]
            grp["repeat_fault"] += cnt["repeat_fault"]
            grp["repeat_with_big"] += cnt["repeat_with_big"]
        for (solver, mapped), cnt in sorted(grouped.items()):
            summary_rows.append((solver, mapped, cnt["total"], cnt["repeat_fault"], cnt["repeat_with_big"]))
    else:
        for (solver, num), cnt in sorted(summary.items()):
            summary_rows.append((solver, num, cnt["total"], cnt["repeat_fault"], cnt["repeat_with_big"]))

    # Write summary CSV
    with open(summary_csv, 'w', newline='') as sf:
        cols = ["solver", "mapped_num", "total_instances", "repeat_fault_count", "repeat_with_big_count"]
        w = csv.DictWriter(sf, fieldnames=cols, delimiter=';')
        w.writeheader()
        for solver, mapped_num, total, rf, rb in summary_rows:
            w.writerow({
                "solver": solver,
                "mapped_num": mapped_num,
                "total_instances": total,
                "repeat_fault_count": rf,
                "repeat_with_big_count": rb
            })
    print("CSV file generated:", summary_csv)
    print(f"\tSummary of fault-tests saved in {os.path.basename(summary_csv)}")

def generate_fault_overview_table(output_dir, apply_map=False):
    """Scan FaultOverview/*.log.xz for bigger/smaller/bigger2^62, 
    aggregate by (solver, mapped‐bug), and write bigger_smaller{suffix}_{TAG}.csv."""
    import glob, lzma, re, csv, os

    print("Generating fault overview table (bigger_smaller*)... This may take a while, as all *.xz files are checked for occurences of small big huge errors!")

    folder = os.path.join(output_dir, 'FaultOverview')
    pattern = os.path.join(folder, '**', '*.log.xz')
    files = glob.glob(pattern, recursive=True)
    if not files:
        print(f"No .log.xz files in {folder}")
        return

    bigger_txt = "c SumOfWeightsi: > UINT32"
    smaller_txt = "c SumOfWeightsi: < UINT32"
    sum_pat = re.compile(r"c SumOfWeights\.: (\d+)")
    suffix = '_mapped' if apply_map else ''
    agg = {}  # (solver,bug) -> {'bigger':bool,'smaller':bool,'bigger2':bool}

    for path in files:
        solver, bug = None, None
        m = re.search(r'/([^/_]+(?:-[^/_]+)?)_(\d+)\.log\.xz$', path)
        if not m:
            continue
        solver, bug_orig = m.group(1), m.group(2)
        # apply mapping if requested
        bug = ERROR_MAP.get(int(bug_orig), bug_orig) if apply_map and bug_orig.isdigit() else bug_orig

        found_bigger = False
        found_smaller = False
        always_big2 = True
        try:
            with lzma.open(path, 'rt', errors='ignore') as f:
                for line in f:
                    if not found_bigger and bigger_txt in line:
                        found_bigger = True
                    if not found_smaller and smaller_txt in line:
                        found_smaller = True
                    if sum_m := sum_pat.search(line):
                        if int(sum_m.group(1)) <= 2**62:
                            always_big2 = False
                    if found_bigger and found_smaller and not always_big2:
                        break
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue

        key = (solver, bug)
        if key not in agg:
            agg[key] = {'bigger': False, 'smaller': False, 'bigger2': True}
        agg[key]['bigger']   = agg[key]['bigger']   or found_bigger
        agg[key]['smaller']  = agg[key]['smaller']  or found_smaller
        agg[key]['bigger2']  = agg[key]['bigger2'] and always_big2

    out_csv = os.path.join(output_dir, f'bigger_smaller{suffix}_{TAG}.csv')
    with open(out_csv, 'w', newline='') as csvf:
        writer = csv.writer(csvf, delimiter=';')
        writer.writerow(['Solver', 'Bug', 'bigger', 'smaller', 'bigger 2^62'])
        for (solver, bug), v in sorted(agg.items()):
            writer.writerow([solver, bug, str(v['bigger']), str(v['smaller']), str(v['bigger2'])])

    print("CSV file generated:", out_csv)
    print(f"\tFault‐overview (bigger/smaller/bigger 2^62) saved in bigger_smaller{suffix}_{TAG}.csv")

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
    parser.add_argument(
        '--tag', 
        dest='tag', 
        default='', 
        help='If set, insert this tag into all filenames and plot titles')
    parser.add_argument(
        '--test-minimized',
        dest='test_minimized',
        action='store_true',
        help='Run compare.py on output_dir/FaultsMinimized and produce fault CSVs'
    )
    args = parser.parse_args()

    # build TAG once
    TAG = args.tag or ""

    log_file_path = args.input_log_file
    output_dir = os.path.dirname(os.path.abspath(log_file_path))
    fault_dir = os.path.join(output_dir, "FaultsMinimized")
    with open(log_file_path, 'r') as f:
        log_text = f.read()
    
    if not os.path.exists(fault_dir):
        os.makedirs(fault_dir)
    if args.test_minimized and os.path.exists(fault_dir):
        while True:
            resp = input("Have you checked that the right solvers are imported into compare.py? (y/n): ").strip().lower()
            if resp in ('y', 'yes'):
                break
            if resp in ('n', 'no'):
                print("Please update compare.py with the correct solvers before running. Exiting.")
                exit(1)
            print("Please answer 'y' or 'n'.")
        generate_fault_tests(fault_dir, output_dir, apply_map=args.apply_map)
    generate_fault_overview_table(output_dir, apply_map=args.apply_map)
    all_data, stats_matches = generate_general_stats(log_text, output_dir)
    bug_details, virtual_best = generate_bug_stats(
        log_text, stats_matches, output_dir, apply_map=args.apply_map
    )
    generate_fuzzer_stats(log_text, output_dir)
    generate_bug_first_occurrence(bug_details, virtual_best, output_dir, apply_map=args.apply_map)
    generate_fuzzer_comparison(bug_details, output_dir)
    generate_fuzzer_summary_table(output_dir)
    generate_solver_timings(log_text, output_dir)
    generate_fuzzer_overview_table(output_dir, apply_map=args.apply_map)
    generate_cdf_plot(output_dir, apply_map=args.apply_map)
    
    print(f"CSV files and plots generated in: {output_dir}")
