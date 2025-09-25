#!/usr/bin/env python3
"""
Reproduce original (unreduced) WCNF instances from minimized filenames and
measure reduction statistics.

Filename pattern (minimized file):
  Solver_Bug_red-bug-SEED-FUZZERNAME-<counter>.wcnf
Example:
  WMCDCLBA_703_red-bug-5990798035778532301-PaxianPySmall-2333935592.wcnf
Extract:
  SEED = 5990798035778532301
  FUZZER = PaxianPySmall

Steps:
1. Scan minimized directory for *.wcnf files matching pattern.
2. For each file:
   - Parse seed & fuzzer.
   - Re-run the fuzzer with that seed to regenerate (unreduced) instance.
     (Command derived from the fuzzers dict in the given config module.)
   - Parse both files with wcnfTool.py (supports old/new format automatically).
   - Compute statistics: Vars, HardClauses, SoftClauses, MaxWeight,
     SumOfWeights, BestOValue (not available without compare -> left blank),
     Average clause length (overall + soft + hard), File size.
3. Compute reduction percentages (Original vs Reduced).
4. Write detailed CSV + summary CSV (min/avg/median/max of reduction % incl. count).
5. Produce plots (if matplotlib installed):
   - reductions bar with error bars (min/max)
   - scatter plot original vs reduced (variables vs soft clauses)

Note:
- We import the config module dynamically (default: configPrivateWeighted24.py).
- Fuzzer invocation logic mimics runwcnfuzz.py:
    * base command from fuzzers[fuzzer]["command"]
    * if "seed" key exists: append "<seed_arg> <SEED>"
      else append " <SEED>"
    * upper_bound ignored here (reproduction of original shape not required)
- Reproduced WCNF is stored under <output_dir>/<reprod_subdir>/<fuzzer>_<seed>.wcnf
- Existing reproduced files are reused if --skip-existing.

"""

import argparse
import csv
import importlib.util
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to ensure repository root is on path (script assumed in Scripts/)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import wcnfTool (global-state parser)
try:
    sys.path.insert(0, str(REPO_ROOT))
    import wcnfTool
except ImportError:
    # fallback attempt (if located elsewhere)
    raise SystemExit("ERROR: Cannot import wcnfTool.py. Ensure it is in repository root.")


FILE_RX = re.compile(
    r'^(?P<solver>[A-Za-z0-9-]+)_(?P<bug>\d+)_red-bug-(?P<seed>\d+)-(?P<fuzzer>[A-Za-z0-9]+)-(?P<count>\d+)\.wcnf$'
)


def load_config_module(config_path: str):
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    spec = importlib.util.spec_from_file_location("fuzz_cfg", str(config_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "fuzzers"):
        raise AttributeError("Config module does not define 'fuzzers' dict.")
    return mod


# ADDED: helper to mimic runwcnfuzz.py argument logic exactly
def build_fuzzer_command(fuzzer_name: str, seed: str, fuzzers_cfg: Dict[str, Dict[str, Any]], upper_bound: int = -1) -> Optional[str]:
    """
    Reproduce the logic in runwcnfuzz.py:create_command()
      fuzz_command = command
      if upper_bound != -1 and fuzzer has 'upper_bound' key -> append its switch + value
      if fuzzer has 'seed' key -> append that token
      finally append the seed value
    """
    info = fuzzers_cfg.get(fuzzer_name)
    if not info:
        return None
    cmd = info.get("command", "").strip()
    if not cmd:
        return None
    if upper_bound != -1 and info.get("upper_bound", False):
        # in runwcnfuzz upper_bound is stored as a flag name (e.g. "-u")
        cmd = f"{cmd} {info['upper_bound']} {upper_bound}"
    if info.get("seed", ""):
        cmd = f"{cmd} {info['seed']}"
    cmd = f"{cmd} {seed}"
    return cmd


# PATCH: extend run_fuzzer with debug + proper working dir + identical arg building
def run_fuzzer(fuzzer: str, seed: str, fuzzers_cfg: Dict[str, Dict[str, Any]],
               out_path: Path, timeout: int, upper_bound: int, debug: bool):
    if out_path.exists() and out_path.stat().st_size > 0:
        if debug:
            print(f"[DEBUG] Reuse existing reproduction: {out_path} size={out_path.stat().st_size}")
        return True
    cmd = build_fuzzer_command(fuzzer, seed, fuzzers_cfg, upper_bound)
    if not cmd:
        print(f"[WARN] No command template for fuzzer '{fuzzer}'.")
        return False
    # Run from repo root (some fuzzers expect relative assets)
    full_cmd = f"(cd '{REPO_ROOT}' && {cmd}) > '{out_path}' 2> '{out_path}.stderr'"
    if debug:
        print(f"[DEBUG] Reproduction command: {full_cmd}")
    try:
        res = subprocess.run(full_cmd, shell=True, timeout=timeout)
        if debug:
            print(f"[DEBUG] Return code={res.returncode}")
    except subprocess.TimeoutExpired:
        print(f"[WARN] Timeout reproducing {fuzzer} seed={seed}")
        return False
    if (not out_path.exists()) or out_path.stat().st_size == 0:
        if debug:
            stderr_path = f"{out_path}.stderr"
            if os.path.exists(stderr_path):
                print(f"[DEBUG] STDERR ({stderr_path}):")
                try:
                    print(Path(stderr_path).read_text()[:2000])
                except Exception:
                    pass
        print(f"[WARN] Empty reproduction for {fuzzer} seed={seed}")
        return False
    return True


# PATCH: safer parsing (fall back if wcnfTool missing API)
def parse_with_wcnf_tool(path: Path, debug: bool) -> Dict[str, Any]:
    stats = {
        "Vars": 0, "HardClauses": 0, "SoftClauses": 0,
        "MaxWeight": 0, "SumOfWeights": 0,
        "AvgClauseLenAll": 0.0, "AvgClauseLenSoft": 0.0, "AvgClauseLenHard": 0.0,
        "Format": "", "FileSize": path.stat().st_size if path.exists() else 0
    }
    if not path.exists():
        return stats
    try:
        if debug:
            print(f"[DEBUG] Parsing file {path.name} size={path.stat().st_size}")
        if not hasattr(wcnfTool, "parse_wcnf"):
            if debug:
                print("[DEBUG] wcnfTool.parse_wcnf missing. Skipping.")
            return stats
        if hasattr(wcnfTool, "reset_values"):
            wcnfTool.reset_values()
        wcnfTool.parse_wcnf(str(path))
        # Required attributes (guard)
        for attr in ["vars", "nbHard", "nbSoft", "maxWeight", "sumOfWeights", "wcnfInputFormat", "clauses"]:
            if not hasattr(wcnfTool, attr):
                if debug:
                    print(f"[DEBUG] wcnfTool missing attr {attr}")
                return stats
        stats["Vars"] = wcnfTool.vars
        stats["HardClauses"] = wcnfTool.nbHard
        stats["SoftClauses"] = wcnfTool.nbSoft
        stats["MaxWeight"] = getattr(wcnfTool, "maxWeight", 0)
        stats["SumOfWeights"] = getattr(wcnfTool, "sumOfWeights", 0)
        stats["Format"] = getattr(wcnfTool, "wcnfInputFormat", "")
        total_len = soft_len = hard_len = 0
        for weight, clause in getattr(wcnfTool, "clauses", []):
            clen = len(clause)
            total_len += clen
            if weight == -1:
                hard_len += clen
            else:
                soft_len += clen
        total_cls = stats["HardClauses"] + stats["SoftClauses"]
        if total_cls:
            stats["AvgClauseLenAll"] = total_len / total_cls
        if stats["SoftClauses"]:
            stats["AvgClauseLenSoft"] = soft_len / stats["SoftClauses"]
        if stats["HardClauses"]:
            stats["AvgClauseLenHard"] = hard_len / stats["HardClauses"]
    except Exception as e:
        print(f"[WARN] Parse failed for {path}: {e}")
    if debug:
        print(f"[DEBUG] Parsed stats: {stats}")
    return stats


# PATCH: add debug + ensure output directory one level up (parent of minimized)
def process_directory(min_dir: Path,
                      output_dir: Path,
                      reprod_subdir: str,
                      fuzzers_cfg: Dict[str, Dict[str, Any]],
                      skip_existing: bool,
                      timeout: int,
                      upper_bound: int,
                      debug: bool) -> List[Dict[str, Any]]:
    rows = []
    reprod_dir = output_dir / reprod_subdir
    reprod_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(min_dir.glob("*.wcnf"))
    if debug:
        print(f"[DEBUG] Minimized dir: {min_dir}")
        print(f"[DEBUG] Output dir:    {output_dir}")
        print(f"[DEBUG] Reprod dir:    {reprod_dir}")
        print(f"[DEBUG] Found {len(files)} candidate minimized files.")
    for idx, file in enumerate(files, 1):
        if debug:
            print(f"\n[DEBUG] === {idx}/{len(files)} FILE {file.name} ===")
        m = FILE_RX.match(file.name)
        if not m:
            if debug:
                print("[DEBUG] Filename pattern mismatch, skip.")
            continue
        seed = m.group("seed")
        fuzzer = m.group("fuzzer")
        solver = m.group("solver")
        bug = m.group("bug")
        reproduced_path = reprod_dir / f"{fuzzer}_{seed}.wcnf"
        if debug:
            print(f"[DEBUG] Parsed seed={seed} fuzzer={fuzzer} solver={solver} bug={bug}")
            print(f"[DEBUG] Reproduction target: {reproduced_path}")
        need_run = (not skip_existing) or (not reproduced_path.exists()) or reproduced_path.stat().st_size == 0
        if debug:
            print(f"[DEBUG] need_run={need_run} skip_existing={skip_existing}")
        if need_run:
            ok = run_fuzzer(fuzzer, seed, fuzzers_cfg, reproduced_path, timeout, upper_bound, debug)
            if not ok and (not reproduced_path.exists() or reproduced_path.stat().st_size == 0):
                if debug:
                    print("[DEBUG] Reproduction failed & no file -> skip row.")
                continue
        orig_stats = parse_with_wcnf_tool(reproduced_path, debug)
        red_stats = parse_with_wcnf_tool(file, debug)
        row = {
            "Fuzzer": fuzzer,
            "Seed": seed,
            "Solver": solver,
            "Bug": bug,
            "OrigFile": reproduced_path.name,
            "ReducedFile": file.name,
            "OrigFormat": orig_stats["Format"],
            "RedFormat": red_stats["Format"],
            "OrigVars": orig_stats["Vars"],
            "RedVars": red_stats["Vars"],
            "OrigHardClauses": orig_stats["HardClauses"],
            "RedHardClauses": red_stats["HardClauses"],
            "OrigSoftClauses": orig_stats["SoftClauses"],
            "RedSoftClauses": red_stats["SoftClauses"],
            "OrigMaxWeight": orig_stats["MaxWeight"],
            "RedMaxWeight": red_stats["MaxWeight"],
            "OrigSumWeights": orig_stats["SumOfWeights"],
            "RedSumWeights": red_stats["SumOfWeights"],
            "OrigAvgClauseLenAll": f"{orig_stats['AvgClauseLenAll']:.3f}",
            "RedAvgClauseLenAll": f"{red_stats['AvgClauseLenAll']:.3f}",
            "OrigAvgClauseLenSoft": f"{orig_stats['AvgClauseLenSoft']:.3f}",
            "RedAvgClauseLenSoft": f"{red_stats['AvgClauseLenSoft']:.3f}",
            "OrigAvgClauseLenHard": f"{orig_stats['AvgClauseLenHard']:.3f}",
            "RedAvgClauseLenHard": f"{red_stats['AvgClauseLenHard']:.3f}",
            "OrigFileSize": orig_stats["FileSize"],
            "RedFileSize": red_stats["FileSize"],
        }
        # --- NEW: aggregate total clauses
        if all(isinstance(row[k], int) for k in ["OrigHardClauses","OrigSoftClauses","RedHardClauses","RedSoftClauses"]):
            row["OrigTotalClauses"] = row["OrigHardClauses"] + row["OrigSoftClauses"]
            row["RedTotalClauses"]  = row["RedHardClauses"] + row["RedSoftClauses"]
        else:
            row["OrigTotalClauses"] = ""
            row["RedTotalClauses"]  = ""
        # Reductions
        def pct(o, r):
            if isinstance(o, (int, float)) and isinstance(r, (int, float)) and o > 0:
                return 100.0 * (o - r) / o
            return ""
        reductions = [
            ("OrigVars", "RedVars", "VarReduction%"),
            ("OrigHardClauses", "RedHardClauses", "HardClauseReduction%"),
            ("OrigSoftClauses", "RedSoftClauses", "SoftClauseReduction%"),
            ("OrigTotalClauses", "RedTotalClauses", "TotalClausesReduction%"),
            ("OrigSumWeights", "RedSumWeights", "SumWeightsReduction%"),
            ("OrigFileSize", "RedFileSize", "FileSizeReduction%"),
            ("OrigAvgClauseLenAll", "RedAvgClauseLenAll", "AvgClauseLenAllReduction%")
        ]
        for o_key, r_key, lbl in reductions:
            try:
                o_val = float(row[o_key]) if "AvgClauseLen" in o_key else row[o_key]
                r_val = float(row[r_key]) if "AvgClauseLen" in r_key else row[r_key]
            except Exception:
                o_val = row[o_key]; r_val = row[r_key]
            pr = pct(o_val, r_val)
            row[lbl] = f"{pr:.2f}" if pr != "" else ""
            if debug:
                print(f"[DEBUG] {lbl}: orig={o_val} red={r_val} -> {row[lbl]}")
        rows.append(row)
        if debug:
            print(f"[DEBUG] Row columns={len(row)}")
    if debug:
        print(f"\n[DEBUG] Finished. Total rows={len(rows)}")
        if rows:
            print(f"[DEBUG] Columns in first row ({len(rows[0])}): {list(rows[0].keys())}")
    return rows


def write_csv(rows: List[Dict[str, Any]], out_csv: Path):
    if not rows:
        print("[INFO] No rows to write.")
        return
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"CSV file generated: {out_csv}")
    print(f"\tReproduction reduction detailed stats saved in {out_csv.name}.")


def summarize(rows: List[Dict[str, Any]], out_csv: Path):
    """
    Overall summary (no per-fuzzer here). Filenames & metric labels stripped of 'Reduction'/'%'.
    """
    if not rows:
        return
    reduction_keys = [k for k in rows[0].keys() if k.endswith("Reduction%")]
    summary = []
    for key in reduction_keys:
        vals = []
        for r in rows:
            v = r.get(key, "")
            if v not in ("", None):
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
        if not vals:
            continue
        base = key.replace("Reduction%", "").replace("%", "")
        summary.append({
            "Metric": base,
            "Min": f"{min(vals):.2f}",
            "Avg": f"{sum(vals)/len(vals):.2f}",
            "Median": f"{statistics.median(vals):.2f}",
            "Max": f"{max(vals):.2f}",
            "Count": len(vals),
        })
    if not summary:
        return
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Metric", "Min", "Avg", "Median", "Max", "Count"], delimiter=';')
        w.writeheader()
        for s in summary:
            w.writerow(s)
    print(f"CSV file generated: {out_csv}")
    print(f"\tOverall summary saved in {out_csv.name}.")


def summarize_per_fuzzer(rows: List[Dict[str, Any]], out_csv: Path):
    """
    Per-fuzzer summary: one line per (fuzzer, metric).
    Metric names stripped of 'Reduction%'.
    """
    if not rows:
        return
    reduction_keys = [k for k in rows[0].keys() if k.endswith("Reduction%")]
    by_fuzzer: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        fuz = r["Fuzzer"]
        by_fuzzer.setdefault(fuz, {})
        for key in reduction_keys:
            v = r.get(key, "")
            if v not in ("", None):
                try:
                    by_fuzzer[fuz].setdefault(key, []).append(float(v))
                except ValueError:
                    pass
    fieldnames = ["Fuzzer", "Metric", "Min", "Avg", "Median", "Max", "Count"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        w.writeheader()
        for fuz, metric_map in sorted(by_fuzzer.items()):
            for key, vals in metric_map.items():
                base = key.replace("Reduction%", "").replace("%", "")
                w.writerow({
                    "Fuzzer": fuz,
                    "Metric": base,
                    "Min": f"{min(vals):.2f}",
                    "Avg": f"{sum(vals)/len(vals):.2f}",
                    "Median": f"{statistics.median(vals):.2f}",
                    "Max": f"{max(vals):.2f}",
                    "Count": len(vals)
                })
    print(f"CSV file generated: {out_csv}")
    print(f"\tPer-fuzzer summary saved in {out_csv.name}.")


def make_plots(rows: List[Dict[str, Any]], detailed_csv: Path, img_format: str = "pdf"):
    """
    Generate all plots. Pure save_plot() is used everywhere to avoid NameError on path vars.
    Identity/Diff legacy block removed (source of 'VarOriguction%' KeyError).
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import statistics
    except Exception as e:
        print(f"[INFO] Plotting skipped: {e}")
        return

    if not rows:
        return

    # Helper to save in requested formats
    def save_plot(base_name: str):
        formats = ("pdf",) if img_format == "pdf" else ("png",) if img_format == "png" else ("png", "pdf")
        out_dir = detailed_csv.parent
        for ext in formats:
            path = out_dir / f"{base_name}.{ext}"
            kwargs = {"dpi": 150} if ext == "png" else {}
            plt.savefig(path, **kwargs)
        print(f"Plot generated: {base_name} ({', '.join(formats)})")

    # Ensure TotalClauses (orig/reduced) columns exist for plotting
    for r in rows:
        if "OrigTotalClauses" not in r or r.get("OrigTotalClauses", "") in ("", None):
            oh, osf = r.get("OrigHardClauses"), r.get("OrigSoftClauses")
            r["OrigTotalClauses"] = (oh + osf) if isinstance(oh, int) and isinstance(osf, int) else None
        if "RedTotalClauses" not in r or r.get("RedTotalClauses", "") in ("", None):
            rh, rsf = r.get("RedHardClauses"), r.get("RedSoftClauses")
            r["RedTotalClauses"] = (rh + rsf) if isinstance(rh, int) and isinstance(rsf, int) else None

    # Collect reduction% vectors
    reduction_keys = [k for k in rows[0].keys() if k.endswith("Reduction%")]
    data: Dict[str, List[float]] = {}
    for key in reduction_keys:
        vals = []
        for r in rows:
            v = r.get(key, "")
            try:
                vals.append(float(v))
            except Exception:
                pass
        if vals:
            data[key] = vals

    # 1) Bars (Avg with Min/Max; median triangle)
    if data:
        metrics = list(data.keys())
        avgs   = [sum(v)/len(v) for v in data.values()]
        meds   = [statistics.median(v) for v in data.values()]
        mins   = [min(v) for v in data.values()]
        maxs   = [max(v) for v in data.values()]
        err_low  = [a - mn for a, mn in zip(avgs, mins)]
        err_high = [mx - a for a, mx in zip(avgs, maxs)]
        clean_labels = [m.replace("Reduction%","").replace("%","") for m in metrics]

        plt.figure(figsize=(12, 4.2))
        x = range(len(metrics))
        bars = plt.bar(x, avgs, yerr=[err_low, err_high], capsize=5,
                       color='slateblue', alpha=0.85, edgecolor='black', linewidth=0.6,
                       label="Average (bar) & Min/Max (error bars)")
        plt.scatter(x, meds, marker='^', color='crimson', s=70, zorder=5, label="Median (triangle)")
        for i,(rect,key) in enumerate(zip(bars, metrics)):
            count = len(data[key])
            plt.text(rect.get_x()+rect.get_width()/2.0, rect.get_height()+2,
                     f"n={count}", ha='center', va='bottom', fontsize=8)
        plt.xticks(x, clean_labels, rotation=25, ha='right')
        plt.ylabel("Reduction (%)")
        plt.title("Instance Size & Weight Reduction Statistics")
        plt.legend(frameon=False)
        plt.tight_layout()
        save_plot("reproduction_stats")
        plt.close()

    # 2) Boxplots
    if data:
        plt.figure(figsize=(12, 4.2))
        metrics = list(data.keys())
        clean_labels = [m.replace("Reduction%","").replace("%","") for m in metrics]
        plt.boxplot([data[k] for k in metrics], labels=clean_labels, showmeans=True,
                    meanprops=dict(marker='o', markerfacecolor='black', markersize=5))
        plt.ylabel("Reduction (%)")
        plt.title("Reduction Distribution (Boxplots with Mean Marker)")
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        save_plot("reproduction_boxplots")
        plt.close()

    # 3) CDF (Reduction% vs cumulative fraction)
    if data:
        plt.figure(figsize=(7,5))
        for key, vals in data.items():
            sv = sorted(vals)
            y = np.linspace(0,1,len(sv))
            label = key.replace("Reduction%","").replace("%","")
            plt.plot(sv, y, label=label)
        plt.xlabel("Reduction (%)")
        plt.ylabel("Cumulative Fraction")
        plt.title("CDF of Reductions")
        plt.legend(frameon=False)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        save_plot("reproduction_cdf")
        plt.close()

        # 3b) Alternative: index on x, reduction on y (TotalClauses removed; square figure)
        plt.figure(figsize=(7,7))  # make the image quadratic
        for key, vals in data.items():
            if key == "TotalClausesReduction%":
                continue  # drop TotalClauses from this plot
            sv = sorted(vals)
            xs = range(1, len(sv)+1)
            label = key.replace("Reduction%","").replace("%","")
            plt.plot(xs, sv, label=label)
        plt.xlabel("Instances (sorted by reduction)")
        plt.ylabel("Reduction (%)")
        plt.title("Instance Reduction CDF")
        plt.legend(frameon=False)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        save_plot("reproduction_cdf_instances")
        plt.close()

    # 4) Original vs Reduced (Vars / TotalClauses / FileSize) with identity line
    try:
        def collect(o_key, r_key):
            o_vals, r_vals = [], []
            for r in rows:
                o = r.get(o_key); rd = r.get(r_key)
                if isinstance(o,(int,float)) and isinstance(rd,(int,float)) and o>=0 and rd>=0:
                    o_vals.append(o); r_vals.append(rd)
            return np.array(o_vals), np.array(r_vals)

        o_vars, r_vars = collect("OrigVars","RedVars")
        o_totc, r_totc = collect("OrigTotalClauses","RedTotalClauses")
        o_fz,   r_fz   = collect("OrigFileSize","RedFileSize")

        pairs = []
        if len(o_vars): pairs.append(("Variables","tab:blue","o", o_vars, r_vars))
        if len(o_totc): pairs.append(("TotalClauses","#006400","P", o_totc, r_totc))  # dark green
        if len(o_fz):   pairs.append(("FileSize","tab:red","d", o_fz, r_fz))

        if pairs:
            all_o = np.concatenate([p[3] for p in pairs])
            all_r = np.concatenate([p[4] for p in pairs])
            max_lim = max(all_o.max(), all_r.max())*1.05

            plt.figure(figsize=(6.2,6.2))
            for label, color, marker, o_vals, r_vals in pairs:
                plt.scatter(r_vals, o_vals, label=label, c=color, marker=marker, alpha=0.7, s=36)
            plt.plot([0,max_lim],[0,max_lim], linestyle='--', color='black', linewidth=1, label='y = x')
            plt.xlabel("Reduced")
            plt.ylabel("Original")
            plt.title("Original vs Reduced (Vars / TotalClauses / FileSize)")
            plt.legend(frameon=False, fontsize=9)
            plt.xlim(0, max_lim); plt.ylim(0, max_lim)
            plt.tight_layout()
            save_plot("reproduction_orig_vs_red")
            plt.close()

            # log-scale
            pos_mask = (all_o>0) & (all_r>0)
            if pos_mask.any():
                plt.figure(figsize=(6.2,6.2))
                for label, color, marker, o_vals, r_vals in pairs:
                    m = (o_vals>0) & (r_vals>0)
                    if m.any():
                        plt.scatter(r_vals[m], o_vals[m], label=label, c=color, marker=marker, alpha=0.7, s=36)
                maxi = max(all_o.max(), all_r.max())*1.1
                plt.plot([1,maxi],[1,maxi], linestyle='--', color='black', linewidth=1, label='y = x')
                plt.xscale('log'); plt.yscale('log')
                plt.xlabel("Reduced (log)"); plt.ylabel("Original (log)")
                plt.title("Per-Instance Reduction Scatter")
                plt.legend(frameon=False, fontsize=9)
                plt.tight_layout()
                save_plot("reproduction_orig_vs_red_log")
                plt.close()
    except Exception as e:
        print(f"[INFO] Simplified multi-metric scatter skipped: {e}")

    # 5) Reduction% pair plots
    try:
        def get_pct(key):  # returns float or None
            try: return float(key)
            except: return None

        var_pct = []; tot_pct=[]; size_pct=[]
        for r in rows:
            try:
                v = float(r.get("VarReduction%","")); t = float(r.get("TotalClausesReduction%","")); s = float(r.get("FileSizeReduction%",""))
            except: 
                v=t=s=None
            if v is not None and t is not None: var_pct.append(v); tot_pct.append(t)
            if v is not None and s is not None: size_pct.append((v,s))

        if var_pct and tot_pct:
            plt.figure(figsize=(6,5))
            plt.scatter(var_pct, tot_pct, alpha=0.65, s=32, c='tab:blue')
            lim = max(max(var_pct), max(tot_pct), 1)
            plt.plot([0,lim],[0,lim], linestyle='--', color='black', linewidth=1)
            plt.xlabel("Variable Reduction (%)")
            plt.ylabel("Total Clauses Reduction (%)")
            plt.title("Reduction %: Variables vs Total Clauses")
            plt.tight_layout()
            save_plot("reduction_pct_vars_vs_totalclauses")
            plt.close()

        if size_pct:
            x = [a for a,b in size_pct]; y = [b for a,b in size_pct]
            plt.figure(figsize=(6,5))
            plt.scatter(x, y, alpha=0.65, s=32, c='tab:red')
            lim = max(max(x), max(y), 1)
            plt.plot([0,lim],[0,lim], linestyle='--', color='black', linewidth=1)
            plt.xlabel("Variable Reduction (%)")
            plt.ylabel("File Size Reduction (%)")
            plt.title("Reduction %: Variables vs File Size")
            plt.tight_layout()
            save_plot("reduction_pct_vars_vs_filesize")
            plt.close()
    except Exception as e:
        print(f"[INFO] Reduction % pair plot skipped: {e}")

    # 6) Joint reduction% Variables vs TotalClauses colored by FileSize%
    try:
        points = []
        for r in rows:
            try:
                v = float(r.get("VarReduction%","")); t = float(r.get("TotalClausesReduction%","")); fz = float(r.get("FileSizeReduction%",""))
                if 0 <= v <= 200 and 0 <= t <= 200 and 0 <= fz <= 200:
                    points.append((v,t,fz))
            except: pass
        if points:
            arr = np.array(points)
            v,t,fz = arr[:,0], arr[:,1], arr[:,2]
            plt.figure(figsize=(6.2,5.4))
            sc = plt.scatter(v, t, c=fz, cmap='plasma', s=30, alpha=0.75)
            lim = max(v.max(), t.max(), 1)
            plt.plot([0,lim],[0,lim], linestyle='--', color='black', linewidth=1)
            cbar = plt.colorbar(sc); cbar.set_label("FileSize Reduction (%)")
            plt.xlabel("Variable Reduction (%)"); plt.ylabel("Total Clauses Reduction (%)")
            plt.title("Joint Reductions (Color = FileSize Reduction)")
            plt.xlim(0,lim); plt.ylim(0,lim)
            plt.tight_layout()
            save_plot("reduction_pct_var_total_filesize_color_filesize")
            plt.close()
    except Exception as e:
        print(f"[INFO] Joint reduction percentage plot skipped: {e}")

    # 7) Absolute and percent difference distributions (core metrics)
    try:
        diff_pairs = [
            ("OrigVars","RedVars","Variables"),
            ("OrigTotalClauses","RedTotalClauses","TotalClauses"),
            ("OrigFileSize","RedFileSize","FileSize"),
            ("OrigSumWeights","RedSumWeights","SumWeights"),
        ]
        abs_diff: Dict[str, List[float]] = {}
        pct_diff: Dict[str, List[float]] = {}
        for o_key, r_key, label in diff_pairs:
            a = []; p = []
            for r in rows:
                o = r.get(o_key); rd = r.get(r_key)
                if isinstance(o,(int,float)) and isinstance(rd,(int,float)) and o>0 and rd>=0:
                    a.append(o-rd); p.append(100.0*(o-rd)/o)
            if a: abs_diff[label]=a
            if p: pct_diff[label]=p

        if abs_diff:
            plt.figure(figsize=(8,4))
            labels = list(abs_diff.keys())
            plt.boxplot([abs_diff[k] for k in labels], labels=labels, showmeans=True,
                        meanprops=dict(marker='o', markerfacecolor='black', markersize=5))
            plt.ylabel("Absolute Difference (Original - Reduced)")
            plt.title("Absolute Reduction (Boxplot)")
            plt.xticks(rotation=20, ha='right')
            plt.tight_layout()
            save_plot("reduction_diff_abs_box")
            plt.close()

        if pct_diff:
            plt.figure(figsize=(7,5))
            for label, vals in pct_diff.items():
                sv = sorted(vals); y = np.linspace(0,1,len(sv))
                plt.plot(sv, y, label=label)
            plt.xlabel("Reduction (%)"); plt.ylabel("Cumulative Fraction")
            plt.title("CDF of Reductions (Core Metrics)")
            plt.legend(frameon=False); plt.grid(alpha=0.3, linestyle='--')
            plt.tight_layout()
            save_plot("reduction_diff_core_cdf")
            plt.close()

            # Alternative index-x version
            plt.figure(figsize=(7,5))
            for label, vals in pct_diff.items():
                sv = sorted(vals); xs = range(1, len(sv)+1)
                plt.plot(xs, sv, label=label)
            plt.xlabel("Instances (sorted by reduction)"); plt.ylabel("Reduction (%)")
            plt.title("Core Metrics Reduction vs Instances")
            plt.legend(frameon=False); plt.grid(alpha=0.3, linestyle='--')
            plt.tight_layout()
            save_plot("reduction_diff_core_cdf_instances")
            plt.close()
    except Exception as e:
        print(f"[INFO] Core diff distribution skipped: {e}")

    # 8) Optional: regenerate CDFs excluding TotalClauses to declutter
    try:
        filtered_items = [(k,v) for k,v in data.items() if k != "TotalClausesReduction%"]
        if filtered_items:
            plt.figure(figsize=(7,5))
            for key, vals in filtered_items:
                sv = sorted(vals); y = np.linspace(0,1,len(sv))
                label = key.replace("Reduction%","").replace("%","")
                plt.plot(sv, y, label=label)
            plt.xlabel("Reduction (%)"); plt.ylabel("Cumulative Fraction")
            plt.title("CDF of Reductions (excluding TotalClauses)")
            plt.legend(frameon=False); plt.grid(alpha=0.3, linestyle='--')
            plt.tight_layout()
            save_plot("reproduction_cdf_filtered")
            plt.close()

            plt.figure(figsize=(7,5))
            for key, vals in filtered_items:
                sv = sorted(vals); xs = range(1, len(sv)+1)
                label = key.replace("Reduction%","").replace("%","")
                plt.plot(xs, sv, label=label)
            plt.xlabel("Instances (sorted by reduction)"); plt.ylabel("Reduction (%)")
            plt.title("Reduction vs Instances (excluding TotalClauses)")
            plt.legend(frameon=False); plt.grid(alpha=0.3, linestyle='--')
            plt.tight_layout()
            save_plot("reproduction_cdf_instances_filtered")
            plt.close()
    except Exception as e:
        print(f"[INFO] Could not regenerate filtered CDFs: {e}")
def main():
    ap = argparse.ArgumentParser(description="Reproduce original WCNF instances and compute reduction statistics.")
    ap.add_argument("minimized_dir", help="Directory with minimized *.wcnf files.")
    ap.add_argument("-c", "--config", default="configPrivateWeighted24.py",
                    help="Config file (contains fuzzers dict).")
    ap.add_argument("-o", "--output-dir", default=None,
                    help="Output directory for CSV/plots (default: parent of minimized_dir).")
    ap.add_argument("--reprod-subdir", default="ReproducedOriginal",
                    help="Subdirectory (inside output-dir) for reproduced originals.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Reuse existing reproduced files.")
    ap.add_argument("--timeout", type=int, default=120,
                    help="Timeout per reproduction command.")
    ap.add_argument("--upper-bound", type=int, default=-1,
                    help="If > -1 pass upper bound argument when fuzzer supports it.")
    ap.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    ap.add_argument(
        "--img-format",
        choices=["pdf", "png", "both"],
        default="pdf",
        help="Image output format (default: pdf for scalable vector graphics)."
    )
    ap.add_argument("--debug", action="store_true",
                    help="Verbose debug output.")
    args = ap.parse_args()

    min_dir = Path(args.minimized_dir).resolve()
    if not min_dir.exists():
        print(f"Minimized directory does not exist: {min_dir}")
        sys.exit(1)

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = min_dir.parent  # ONE LEVEL UP as requested
    output_dir.mkdir(parents=True, exist_ok=True)

    # resolve config
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        if not cfg_path.exists():
            alt = REPO_ROOT / cfg_path.name
            if alt.exists():
                cfg_path = alt
    if not cfg_path.exists():
        print(f"ERROR: Config file not found: {cfg_path}")
        sys.exit(1)
    cfg = load_config_module(str(cfg_path))
    fuzzers_cfg = cfg.fuzzers  # type: ignore

    if args.debug:
        print(f"[DEBUG] Using config: {cfg_path}")
        print(f"[DEBUG] Output dir: {output_dir}")
        print(f"[DEBUG] Reproduction subdir: {args.reprod_subdir}")

    start = time.time()
    rows = process_directory(
        min_dir=min_dir,
        output_dir=output_dir,
        reprod_subdir=args.reprod_subdir,
        fuzzers_cfg=fuzzers_cfg,
        skip_existing=args.skip_existing,
        timeout=args.timeout,
        upper_bound=args.upper_bound,
        debug=args.debug
    )

    detailed_csv = output_dir / "reproduction_detailed.csv"
    write_csv(rows, detailed_csv)

    summary_csv = output_dir / "reproduction_summary.csv"
    summarize(rows, summary_csv)

    per_fuzzer_csv = output_dir / "reproduction_summary_by_fuzzer.csv"
    summarize_per_fuzzer(rows, per_fuzzer_csv)

    if rows and (not args.no_plots):
        make_plots(rows, detailed_csv, img_format=args.img_format)

    # Paper style description
    if args.debug:
        print(f"[DEBUG] Total runtime {time.time()-start:.2f}s")
    print(f"Done in {time.time()-start:.2f}s.")

if __name__ == "__main__":
    main()