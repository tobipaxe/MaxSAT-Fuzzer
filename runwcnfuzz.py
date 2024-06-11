#!/usr/bin/env python3

import threading
import random
import time
import subprocess
import os
import shutil
import sys
import argparse
import re
import math
from datetime import date
import glob
import csv
import statistics
import typing
import psutil
from config import (
    fuzzers,
    wcnf_compare_script,
    delta_debugger_compare_script,
    delta_debugger,
)

minimize = 0
min_free_disk_space = 2  # GB
free_first_time = 0
time_first_time = 0
analyze_fuzzer_instances = False
analyze_solver_timings = False
save_solver_timings = False
analyze_last_xxx = 0
faulty_wcnf_location = ""
timeout = 0
upper_bound = -1  # no upper bound enforced in the fuzzer
location = ""
lg_path = ""
wcnfddmin_lg_path = ""
ddmin_log_path = ""
min_path = ""
log_path = ""
wcnf_path = ""
csv_filename = ""
file_list = None
folder = None
threads: typing.List[threading.Thread] = []  # Add type annotation to "threads" variable
time_first_time = -1
files_processed = 0
files_error_free = 0


# ANSI escape codes for colored output
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
RESET = "\033[0m"

# Lock for updating shared data
global_lock = threading.Lock()
# Termination flag
terminate_flag = False
# Shared counters and timers
overall_non_zero_codes = 0
overall_loops = 0
overall_execution_time = 0
overall_invalid_instances: typing.Dict[int, str] = (
    {}
)  # Add type annotation to "overall_invalid_instances" variable
overall_number_threads = 0
invalid_description = {
    -45: "ddmin received SIGTERM",
    -39: "ddmin received SIGKILL",
    -32: "ddmin received SIGINT",
    -111: "ddmin could not reduce size",
    -110: "ddmin Mode not implemented",
    -109: "ddmin parse error in parseWeight2",
    -108: "ddmin parse error negative weight",
    -107: "ddmin parse error in parseWeight",
    -106: "ddmin parse error in parseInt",
    -105: "ddmin terminate due to signal return smallest instance",
    -104: "ddmin reduced but not copied",
    -103: "ddmin no error after reduction",
    -102: "ddmin non-deterministic error",
    -101: "ddmin parse error",
    -6: "invalid compare.py return value",
    -5: "upper bound violation",
    -4: "fuzzer return code invalid",
    -3: "fuzzer invalid line",
    -2: "weight < 0",
    -1: "invalid p line",
}


def terminate_processes(process, sigterm_only=False):
    try:
        children = process.children(recursive=True)
        for child in children:
            if sigterm_only:
                print(
                    f"  Sending SIGTERM to PID {child.pid} with command line: {' '.join(child.cmdline())}"
                )
                child.terminate()  # Send SIGTERM (terminate)
            else:
                print(
                    f"  Sending SIGKILL to PID {child.pid} with command line: {' '.join(child.cmdline())}"
                )
                child.kill()  # Send SIGKILL (kill)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        # Handle processes that no longer exist or are inaccessible
        pass


# Loop through all files *.lg files attach them to the *.log files
def combine_log_files(origin_path):
    if origin_path == "":
        return
    for file in os.listdir(origin_path):
        source_file_path = os.path.join(origin_path, file)

        # Ensure that the item is a file
        if not os.path.isfile(source_file_path):
            continue

        # Read the entire content of the file and extract the last line
        with open(source_file_path, "r") as file_to_read:
            content = file_to_read.read().strip().splitlines()
            last_line = content[-1] if content else ""

        # Check if the last line starts with "c rc"
        if last_line.startswith("c rc"):
            parts = os.path.basename(file).split("_")
            solver_name = parts[0]
            return_code = parts[1].split(".")[0]

            # name of the log file
            log_file = f"{log_path}/{solver_name}_{return_code}.log"

            # Append the content of the source file to the appropriate target file
            with open(log_file, "a") as target_file:
                target_file.write("\n".join(content) + "\n\n")

            if os.path.exists(source_file_path):
                os.remove(source_file_path)


def print_if_exists(description, path, color):
    reset = "\033[0m"
    if os.path.exists(path):
        if os.path.isfile(path):
            print(f"{color}{description:<30}: {path}{reset}")
        elif os.path.isdir(path):
            # Check if the directory is not empty
            if os.listdir(path):
                print(f"{color}{description:<30}: {path}{reset}")
            else:
                os.rmdir(path)


def cleanup():
    # print("error_tracker.error_occurrences")
    # print(error_tracker.error_occurrences)
    # print("error_tracker.active_debuggers")
    # print(error_tracker.active_debuggers)
    # print("error_tracker.successfully_reduced")
    # print(error_tracker.successfully_reduced)
    # print("error_tracker.unsuccessfully_reduced")
    # print(error_tracker.unsuccessfully_reduced)

    print("Cleaning up, this may take a while.")
    thread_timeout = 5

    print(f"Terminating all subprocesses.")
    if any(t.is_alive() for t in threads):
        print(f"Send SIGTERM to all delta debugger subprocesses.")
        # Get all running processes and check for "compare.py" or "wcnfddmin"
        for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
            try:
                # Check if the process name contains either "compare.py" or "wcnfddmin"
                cmdline = " ".join(proc.info["cmdline"])
                if "wcnfddmin" in cmdline:
                    # print(
                    #     f"  Sending SIGTERM to PID {proc.info['pid']} with command line: {cmdline}"
                    # )
                    proc.terminate()  # Send SIGTERM (terminate)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Handle processes that no longer exist or are inaccessible
                pass

    # Wait for threads with a timeout
    start_time = time.time()

    for t in threads:
        t.join(timeout=0.3)

    if any(t.is_alive() for t in threads):
        print(
            f"Wait (at most) {thread_timeout} seconds if SIGINT/SIGTERM was successful."
        )
        while any(t.is_alive() for t in threads) and (
            thread_timeout - (time.time() - start_time) > 0
        ):
            time.sleep(0.1)

    if any(t.is_alive() for t in threads):
        print(
            "Some threads are still running. Sending SIGTERM to ALL remaining subprocesses."
        )
        current_process = psutil.Process()

        terminate_processes(current_process, sigterm_only=True)
    
    start_time = time.time()

    while any(t.is_alive() for t in threads) and (
        thread_timeout - (time.time() - start_time) > 0
    ):
        time.sleep(0.1)

    if any(t.is_alive() for t in threads):
        print(
            "Some threads are still running. Sending SIGKILL to ALL remaining subprocesses."
        )
        current_process = psutil.Process()
        # Send SIGTERM to all subprocesses
        terminate_processes(current_process, sigterm_only=False)

    print("All threads terminated -- process cleanup done.")

    combine_log_files(lg_path)
    combine_log_files(wcnfddmin_lg_path)
    print("Log files combined.")

    original_stdout = sys.stdout
    with open(log_path + ".log", "w") as file:
        sys.stdout = file  # Change the standard output to the file we created.
        print_status()
        sys.stdout = original_stdout
    print("Overview log file written.")
    if not os.listdir(lg_path):
        os.rmdir(lg_path)
    else:
        print(
            f"Not all log files combined, {lg_path} is not deleted. Please check the folder."
        )
    if not os.listdir(wcnfddmin_lg_path):
        os.rmdir(wcnfddmin_lg_path)
    else:
        print(
            f"Not all log files combined, {wcnfddmin_lg_path} is not deleted. Please check the folder."
        )
    for file in os.listdir(log_path):
        if file.endswith(".log"):
            file_path = os.path.join(log_path, file)
            subprocess.run(["xz", "-z", file_path], check=False)
            # os.remove(file_path)
    print("Log files compressed.")

    if save_solver_timings:
        print("Print solver and fuzzer stats to csv.")
        for fuzz_name, fuzz_details in fuzzers.items():
            if fuzz_name == "DeltaDebugger":
                continue
            fuzz_details["stats"].write_instance_stats_to_csv(
                csv_filename + "_" + fuzz_name + ".csv"
            )
        print("Solver and fuzzer stats written to csv.")

    # Remove all /tmp/*.wcnf files
    wcnf_files = glob.glob("/tmp/*.wcnf")
    for file in wcnf_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except FileNotFoundError:
                print("File not found.")
    # Remove all /tmp/*.pbp files
    # Created with certified solver scripts
    pbp_files = glob.glob("/tmp/*.pbp")
    for file in pbp_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except FileNotFoundError:
                print("File not found.")
    print("/tmp/ cleaned up.")
    print("Done.\n")

    paths_and_descriptions = [
        ("Parallel fuzzer log", f"{log_path}.log"),
        ("Delta debugging logs", ddmin_log_path),
        ("Fault/solver logs", log_path),
        ("Minimized WCNFs", min_path),
        ("Unreduced WCNFs", faulty_wcnf_location),
    ]

    # Print the descriptions and paths only if they exist
    for description, path in paths_and_descriptions:
        print_if_exists(description, path, "\033[1;30;42m")
    csv_files = glob.glob(f"{location}/*.csv")
    for file in csv_files:
        print_if_exists("Solver/fuzzer statistics", file, "\033[1;30;42m")
    # "\033[1;37m" <- bold white didn't work properly
    print()

    exit(0)


def select_fuzzer():
    if folder is not None:
        fuzzer = os.path.basename(os.path.normpath(folder))
        fuzzers[fuzzer]["stats"].active_executions += 1
        return fuzzer

    if overall_loops < 10 * len(fuzzers):
        available_fuzzers = [key for key in fuzzers if key != "DeltaDebugger"]
        fuzzer = random.choice(available_fuzzers)
        # print(f"loops: {overall_loops}, fuzzer: {fuzzer}")
        fuzzers[fuzzer]["stats"].active_executions += 1
        return fuzzer

    min_execution_time = float("inf")

    for fuzzer_name, fuzzer_details in fuzzers.items():
        if fuzzer_name == "DeltaDebugger":
            continue
        avg_execution_time = (
            fuzzer_details["stats"].execution_time / fuzzer_details["stats"].loops
            if fuzzer_details["stats"].loops > 0
            else 0
        )
        if (
            fuzzer_details["stats"].execution_time
            + (fuzzer_details["stats"].active_executions * avg_execution_time)
            < min_execution_time
        ):
            min_execution_time = fuzzer_details["stats"].execution_time
            next_fuzzer = fuzzer_name
    fuzzers[next_fuzzer]["stats"].active_executions += 1
    # print(f"loops: {overall_loops}, min_execution_time: {min_execution_time} fuzzer: {next_fuzzer}")
    return next_fuzzer


# create commands depending on fuzzer / solver
def create_command():
    global fuzzers, file_list, terminate_flag

    instance = error_tracker.get_next_instance_to_reduce()
    if instance is not None:
        # print(instance)
        # print(os.path.abspath(instance[2][0]))
        # print(delta_debugger)
        with global_lock:
            fuzzers["DeltaDebugger"]["stats"].active_executions += 1
        return -1, instance[0], instance[1], os.path.abspath(instance[2][0])

    fuzzer_name = select_fuzzer()
    fuzzer_dict = fuzzers[fuzzer_name]
    seed = random.getrandbits(64)
    # print("seed: " + str(seed) + " < 2**64? " + str(seed < 2**64))
    while fuzzers[fuzzer_name].get("max_seed", 2**64) < seed:
        seed //= 10
    while fuzzers[fuzzer_name].get("min_seed", 0) > seed:
        seed *= 10

    fuzz_command = fuzzer_dict.get("command", "")
    if upper_bound != -1 and fuzzer_dict.get("upper_bound", False):
        fuzz_command += " " + fuzzer_dict["upper_bound"] + " " + str(upper_bound)
    if fuzzer_dict.get("seed", False):
        fuzz_command += " " + fuzzer_dict.get("seed", "")
    fuzz_command += " " + str(seed)
    wcnf_compare = wcnf_compare_script + " " + fuzzer_dict.get("compare_extra", "")

    if folder is not None:
        if file_list:
            with global_lock:
                file = file_list.pop()
        else:
            return False, False, False, False
            # if not terminate_flag:
            #     file = ""
            #     terminate_flag = True
        return 0, fuzzer_name, file, wcnf_compare
    # print(seed, fuzzer_name, fuzz_command, wcnf_compare)
    return seed, fuzzer_name, fuzz_command, wcnf_compare


class ErrorTrackingSystem:
    def __init__(self):
        self.solver_errors = {}  # Nested: solver -> error_code -> fuzzer -> count
        self.error_descriptions = {}  # error_code -> description
        self.error_counter = 0

        # Track the first five instances of each solver/error combination
        self.error_occurrences = (
            {}
        )  # Nested: solver -> error_code -> (wcnf_instance, execution_time) list

        # Track delta debugger counts per solver/error_code combination
        self.active_debuggers = {}  # Nested: solver -> error_code -> count
        self.successfully_reduced = {}  # Nested: solver -> error_code -> count
        self.unsuccessfully_reduced = {}  # Nested: solver -> error_code -> count

    def add_error(
        self,
        solver,
        fuzzer,
        error_code,
        error_description,
        wcnf_instance,
        execution_time,
        do_minimization,
    ):
        self.error_counter += 1

        # Ensure the solver/error structure exists
        if solver not in self.solver_errors:
            self.solver_errors[solver] = {}
        if error_code not in self.solver_errors[solver]:
            self.solver_errors[solver][error_code] = {}
        if fuzzer not in self.solver_errors[solver][error_code]:
            self.solver_errors[solver][error_code][fuzzer] = 0

        # Update counts and descriptions
        self.solver_errors[solver][error_code][fuzzer] += 1
        if error_code not in self.error_descriptions:
            self.error_descriptions[error_code] = error_description

        # Initialize the error_occurrences structure
        if solver not in self.error_occurrences and wcnf_instance != "":
            self.error_occurrences[solver] = {}
        if error_code not in self.error_occurrences[solver] and wcnf_instance != "":
            self.error_occurrences[solver][error_code] = []
        if (
            do_minimization
            and len(self.error_occurrences[solver][error_code]) < minimize
            and wcnf_instance != ""
        ):
            self.error_occurrences[solver][error_code].append(
                (wcnf_instance, execution_time)
            )

        # Initialize debugger counters if necessary
        if solver not in self.active_debuggers:
            self.active_debuggers[solver] = {}
            self.successfully_reduced[solver] = {}
            self.unsuccessfully_reduced[solver] = {}
        if error_code not in self.active_debuggers[solver]:
            self.active_debuggers[solver][error_code] = 0
            self.successfully_reduced[solver][error_code] = 0
            self.unsuccessfully_reduced[solver][error_code] = 0

    def update_debugger_count(
        self, solver, error_code, active=None, success=None, failure=None
    ):
        """Update the counters for active, successfully reduced, or unsuccessfully reduced delta debuggers."""
        if (
            solver not in self.active_debuggers
            or error_code not in self.active_debuggers[solver]
        ):
            return  # The error combination hasn't been initialized

        if active is not None:
            self.active_debuggers[solver][error_code] += active
        if success is not None:
            self.successfully_reduced[solver][error_code] += success
        if failure is not None:
            self.unsuccessfully_reduced[solver][error_code] += failure

    def get_next_instance_to_reduce(self):
        """Find the next error combination to reduce, prioritizing combinations with fewer attempts and lowest execution time.

        Returns the error combination with fewer than two active debuggers and fewer than five successful reductions,
        prioritizing those with fewer active + successful reductions, and selects the instance with the lowest execution time.
        """
        candidates = []

        # Collect all suitable combinations that meet the criteria
        for solver, error_dict in self.active_debuggers.items():
            for error_code, active_count in error_dict.items():
                successful_count = self.successfully_reduced[solver][error_code]
                failure_count = self.unsuccessfully_reduced[solver][error_code]
                error_occurrences = self.error_occurrences.get(solver, {}).get(
                    error_code, []
                )

                # Ensure the combination meets the reduction criteria
                if (
                    active_count < 2
                    and successful_count + active_count < minimize
                    and (
                        failure_count < minimize * successful_count
                        or failure_count < minimize
                    )
                    and error_occurrences
                ):
                    # Calculate the sorting key: fewer active instances + successful reductions
                    priority_score = active_count + successful_count
                    candidates.append(
                        (priority_score, solver, error_code, error_occurrences)
                    )

        # Sort candidates based on priority score
        candidates.sort(key=lambda x: x[0])

        # Return the candidate with the lowest execution time
        if candidates:
            _, solver, error_code, error_occurrences = candidates[0]

            # Find the instance with the lowest execution time
            lowest_time_instance = min(error_occurrences, key=lambda x: x[1])
            error_occurrences.remove(lowest_time_instance)

            # Increment the active count for the selected combination
            self.update_debugger_count(solver, error_code, active=1)

            return solver, error_code, lowest_time_instance

        # No suitable instance found
        return None

    def sum_errors_for_fuzzer(self, fuzzer_name):
        total = 0
        for errors in self.solver_errors.values():
            for fuzz in errors.values():
                if fuzzer_name in fuzz:
                    total += fuzz[fuzzer_name]
        return total

    def all_errors(self):
        messages = []
        for solver, errors in self.solver_errors.items():
            for error_code, fuzz in sorted(errors.items()):
                error_count = sum(fuzz.values())
                messages.append(f"{solver}({error_code}):{error_count}")
        return len(messages), ", ".join(messages)

    def fuzzer_errors(self, specific_fuzzer):
        messages = []
        for solver, errors in self.solver_errors.items():
            for error_code, fuzz in sorted(errors.items()):
                if specific_fuzzer in fuzz:
                    messages.append(f"{solver}({error_code}):{fuzz[specific_fuzzer]}")
        return len(messages), ", ".join(messages)

    # def unique_fuzzer_errors(self, specific_fuzzer):
    #     messages = []
    #     for solver, errors in self.solver_errors.items():
    #         for error_code, fuzz in sorted(errors.items()):
    #             # Check if this error was only found by the specific fuzzer
    #             if specific_fuzzer in fuzz and len(fuzz) == 1:
    #                 messages.append(f"{solver}({error_code}):{fuzz[specific_fuzzer]}")
    #     return len(messages), ", ".join(messages)

    def unique_fuzzer_errors(self, specific_fuzzer, exclude_fuzzer=None):
        messages = []
        for solver, errors in self.solver_errors.items():
            for error_code, fuzz in sorted(errors.items()):
                # Check if this error was found by the specific fuzzer
                if specific_fuzzer in fuzz:
                    # Exclude the excluded fuzzer if specified
                    filtered_fuzzers = {
                        f: c for f, c in fuzz.items() if f != exclude_fuzzer
                    }
                    # Check if the error is unique to the specific fuzzer among the filtered fuzzers
                    if (
                        len(filtered_fuzzers) == 1
                        and specific_fuzzer in filtered_fuzzers
                    ):
                        messages.append(
                            f"{solver}({error_code}):{filtered_fuzzers[specific_fuzzer]}"
                        )
        return len(messages), ", ".join(messages)


class StatsTracker:
    def __init__(self):
        self.min = 2**64
        self.max = -1
        self.instance_stats = {}
        self.general_stats = {
            "HardClauses": {
                "min": self.min,
                "max": self.max,
                "sum": 0,
                "zero": 0,
                "count": 0,
            },
            "SoftClauses": {
                "min": self.min,
                "max": self.max,
                "sum": 0,
                "zero": 0,
                "count": 0,
            },
            "Variables": {
                "min": self.min,
                "max": self.max,
                "sum": 0,
                "count": 0,
            },
            "MaxWeight": {
                "min": self.min,
                "max": self.max,
                "sum": 0,
                "one": 0,
                "count": 0,
            },
            "SumOfWeights": {
                "min": self.min,
                "max": self.max,
                "sum": 0,
                "zero": 0,
                "count": 0,
                "32BitNumber": 0,
            },
            "BestOValue": {
                "min": self.min,
                "max": self.max,
                "sum": 0,
                "zero": 0,
                "count": 0,
            },
            "satisfiable": {"SAT": 0, "UNSAT": 0},
        }
        self.solvers = (
            {}
        )  # Key: solver name, Value: {"time": [min, max, avg, count], "memory": [min, max, avg, count]}
        self.last_seed = 0
        self.loops = 0
        self.compare_loops = 0
        self.execution_time = 0
        self.non_zero_codes = 0
        self.invalid_instances = {}  # invalid error code
        self.active_executions = 0

    def update_general_stat(self, stat_name, value):
        if save_solver_timings and stat_name == "Seed":
            self.last_seed = value
            self.instance_stats[self.last_seed] = {}
            return
        elif stat_name == "Seed":
            return
        stat = self.general_stats[stat_name]
        if stat_name == "satisfiable":
            if save_solver_timings:
                self.instance_stats[self.last_seed][stat_name] = value
            if value:
                stat["SAT"] += 1
            else:
                stat["UNSAT"] += 1
            return
        if stat_name == "SumOfWeights":
            if value < 2**32:
                stat["32BitNumber"] += 1
        stat["min"] = min(stat["min"], value)
        stat["max"] = max(stat["max"], value)
        stat["sum"] += value
        if save_solver_timings:
            self.instance_stats[self.last_seed][stat_name] = value
        stat["count"] += 1
        if stat_name != "Variables" and stat_name != "MaxWeight" and value == 0:
            stat["zero"] += 1
        if stat_name == "MaxWeight" and value == 1:
            stat["one"] += 1

    def update_solver_stat(self, solver_name, time, memory):
        if save_solver_timings:
            self.instance_stats[self.last_seed]["ztime " + solver_name] = time
            self.instance_stats[self.last_seed]["zmem " + solver_name] = memory
        if solver_name not in self.solvers:
            if time != -1 and memory != -1:
                self.solvers[solver_name] = {
                    "time": {"min": time, "max": time, "sum": time},
                    "memory": {"min": memory, "max": memory, "sum": memory},
                    "count": 1,
                    "overall_counter": 1,
                    "timeout": 0,
                }
        else:
            solver = self.solvers[solver_name]
            # Update time stats
            solver["overall_counter"] += 1
            if time != -1 and memory != -1:
                solver["time"]["min"] = min(solver["time"]["min"], time)
                solver["time"]["max"] = max(solver["time"]["max"], time)
                solver["time"]["sum"] += time
                # Update memory stats
                solver["memory"]["min"] = min(solver["memory"]["min"], memory)
                solver["memory"]["max"] = max(solver["memory"]["max"], memory)
                solver["memory"]["sum"] += memory
                # Update counter
                solver["count"] += 1
            else:
                solver["timeout"] += 1

    def dump_instance_analysis(self):
        if self.loops < 15:
            return
        print("")
        stat_names_width = max(len(stat_name) for stat_name in self.general_stats) + 4
        header = f"{'Stat':<{stat_names_width}}Min            Max                      Avg                      Percentage"
        print(header)
        for stat_name, stat in self.general_stats.items():
            if stat_name != "satisfiable":
                avg = stat["sum"] / stat["count"] if stat["count"] > 0 else 0
                if (
                    stat_name == "SumOfWeights"
                    or stat_name == "BestOValue"
                    or stat_name == "MaxWeight"
                ):
                    min_max_avg = f"{stat['min']:<15}{stat['max']:<25}{avg:<25.0f}"
                else:
                    min_max_avg = f"{stat['min']:<15}{stat['max']:<25}{avg:<25.2f}"
                string = f"{stat_name:<{stat_names_width}}{min_max_avg}"
                if stat_name == "MaxWeight":
                    percent_one = (
                        (stat["one"] / stat["count"] * 100) if stat["count"] > 0 else 0
                    )
                    string = f"{string}{percent_one:<7.2f}% unweighted"
                elif stat_name != "Variables":
                    percent_zero = (
                        (stat.get("zero", 0) / stat["count"] * 100)
                        if stat["count"] > 0
                        else 0
                    )
                    string = f"{string}{percent_zero:<7.2f}% of the values are 0"
                print(f"{string}")
                if stat_name == "SumOfWeights":
                    percent_32bit = (
                        (stat["32BitNumber"] / stat["count"] * 100)
                        if stat["count"] > 0
                        else 0
                    )
                    print(
                        f"  {'32BitNumber':<{stat_names_width-2}}{'':<65}{percent_32bit:<7.2f}% are 32-bit numbers"
                    )
            else:
                sat = stat["SAT"]
                unsat = stat["UNSAT"]
                percent_sat = (sat / (sat + unsat) * 100) if (sat + unsat) > 0 else 0
                print(
                    f"{'Satisfiable':<{stat_names_width}}{'':<65}{percent_sat:<7.2f}% of the hard clauses are satisfiable"
                )

    def dump_solver_analysis(self):
        if self.loops < 15:
            return
        print("")
        solver_name_width = max(len(solver_name) for solver_name in self.solvers) + 4
        print(
            f"{'      ':<{solver_name_width}} Time                                                         Memory"
        )
        print(
            f"{'Solver':<{solver_name_width}} Min            Max            Average        Timeouts        Min            Max            Average"
        )
        for solver_name, solver in self.solvers.items():
            avg_time = (
                solver["time"]["sum"] / solver["count"] if solver["count"] > 0 else 0
            )
            avg_memory = (
                solver["memory"]["sum"] / solver["count"] if solver["count"] > 0 else 0
            )
            timeouts = (
                solver["timeout"] / solver["overall_counter"]
                if solver["overall_counter"] > 0
                else 0
            )
            timeouts = f"{timeouts:.4f}%"
            time_stats = f"{solver['time']['min']:<15.3f}{solver['time']['max']:<15.3f}{avg_time:<15.3f}{timeouts:<15}"
            memory_stats = f"{solver['memory']['min']:<15}{solver['memory']['max']:<15}{avg_memory:<15.0f}"
            print(f"{solver_name:<{solver_name_width}} {time_stats:<30} {memory_stats}")

    def dump_fuzzer_stats(self):
        if not self.instance_stats:
            print("No instance statistics available.")
            return

        # Get the last 500 or fewer entries
        last_xxx_entries = list(self.instance_stats.values())[-analyze_last_xxx:]

        # Data structure to store values for calculation
        data = {key: [] for key in self.general_stats.keys()}
        # add solvers here if you want additional statistics over them.
        data["ztime EMS"] = []
        data["ztime RLAut"] = []
        zero_count = {key: 0 for key in self.general_stats.keys()}
        one_count = 0
        sat_count = 0
        thirty_two_bit_number_count = 0

        # Collect data
        for entry in last_xxx_entries:
            for key in data:
                if key in entry and entry[key] is not None:
                    data[key].append(entry[key])
                    if entry[key] == 0:
                        zero_count[key] += 1
                    if key == "SumOfWeights" and entry[key] < 2**32:
                        thirty_two_bit_number_count += 1
                    if key == "MaxWeight" and entry[key] == 1:
                        one_count += 1
                    if key == "satisfiable" and entry[key]:
                        sat_count += 1

        # Print formatted statistics
        print(
            f"\nStatistics over the last {analyze_last_xxx} instances, for fuzzer tuning:"
        )
        print(f"{'Stat':<15}{'Median':<25}{'Avg':<25}{'Percentage':<15}")
        for key, values in data.items():
            if values:
                median_val = statistics.median(values)
                avg_val = sum(values) / len(values)
                line = f"{key:<15}{median_val:<25}{avg_val:<25.2f}"
                if key in ["SumOfWeights", "BestOValue", "HardClauses", "SoftClauses"]:
                    percentage = (zero_count[key] / len(values)) * 100
                    line += f"{percentage:<7.2f}% of the values are 0"
                if key == "MaxWeight":
                    percentage = (one_count / len(values)) * 100
                    line += f"{percentage:<7.2f}% of the instances are unweighted."
                if key == "satisfiable":
                    percentage = (sat_count / len(values)) * 100
                    line += f"{percentage:<7.2f}% of the hard clauses are satisfiable"
                print(line)

        # Print 32-bit number percentage for SumOfWeights
        if data["SumOfWeights"]:
            percentage_32_bit = (
                thirty_two_bit_number_count / len(data["SumOfWeights"])
            ) * 100
            print(
                f"  {'32BitNumber':<13}{'':<36}{percentage_32_bit:.2f}% are 32-bit numbers"
            )

        # Special case for "satisfiable"
        sat = sum(1 for stats in last_xxx_entries if stats.get("satisfiable") == "SAT")
        unsat = sum(
            1 for stats in last_xxx_entries if stats.get("satisfiable") == "UNSAT"
        )
        total_satisfiable = sat + unsat
        if total_satisfiable > 0:
            sat_percentage = (sat / total_satisfiable) * 100
            print(
                f"{'Satisfiable':<15}{'':<50}{sat_percentage:.2f}% of the hard clauses are satisfiable"
            )

    def write_instance_stats_to_csv(self, filename):
        if not self.instance_stats:
            return

        # Collect all unique keys from nested dictionaries
        all_keys = set()
        for stats in self.instance_stats.values():
            all_keys.update(stats.keys())

        fieldnames = ["Seed"] + list(sorted(all_keys))

        # Flatten the data with consideration for missing keys in some entries
        rows = []
        for seed, stats in self.instance_stats.items():
            row = {"Seed": seed}
            # Fill in missing keys with a default value (e.g., None or an appropriate placeholder)
            for key in all_keys:
                row[key] = stats.get(key, "")  # Use None or a suitable default
            rows.append(row)

        # Write to CSV
        with open(filename, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


for fuzzer_name in fuzzers:
    fuzzers[fuzzer_name]["stats"] = StatsTracker()
error_tracker = ErrorTrackingSystem()


def is_valid_file(arg):
    if not os.path.exists(arg):
        print("The file %s does not exist!" % arg)
        exit(1)


def check_if_valid_wcnf(filename):
    sumOfWeights = 0
    hardClauseIndicator = "h"
    top = 0

    if not os.path.exists(filename):
        print(f"The file {filename} does not exist")
        return -4

    with open(filename, "r") as file:
        for line in file:
            line = line.strip()

            # Ignore comments
            if line.startswith("c") or line == "":
                continue

            if line.startswith("p"):
                temp_list = list(map(str, line.split()))
                if temp_list[1] == "wcnf" and len(temp_list) == 5:
                    top = int(temp_list[4])
                else:
                    return -1
                hardClauseIndicator = str(top)
                continue

            if line.startswith(hardClauseIndicator):
                continue
            elif line[0].isdigit():
                weight = int(line.split(" ", 1)[0])
                if weight < 0:
                    return -2
                sumOfWeights += weight
            else:
                return -3

    return sumOfWeights


def check_compare_output(output, fuzzer, save_stats, do_minimization):
    # Pattern to match the lines and capture the error code and the rest of the description
    pattern = r"c (\w+) with ERROR CODE  (\d+)  and FAULT DESCRIPTION:  (.*)$"
    solver_stats = False
    no_error = True
    current_wcnf = ""

    for line in output:
        # print(line)
        if line.startswith("[STDOUT]") or line.startswith("[STDERR]"):
            break

        if line.startswith("c copy saved as: "):
            current_wcnf = line[17:]
        if line.startswith("c Total time...: "):
            compare_time = float(line[17:])
        if save_stats:
            if line.startswith("c SEED.........:"):
                # with global_lock:
                fuzzers[fuzzer]["stats"].update_general_stat("Seed", int(line[17:]))
                continue
            if line.startswith("c HardClauses..:"):
                # with global_lock:
                fuzzers[fuzzer]["stats"].update_general_stat(
                    "HardClauses", int(line[17:])
                )
                continue
            if line.startswith("c SoftClauses..:"):
                # with global_lock:
                fuzzers[fuzzer]["stats"].update_general_stat(
                    "SoftClauses", int(line[17:])
                )
                continue
            if line.startswith("c Variables....:"):
                # with global_lock:
                fuzzers[fuzzer]["stats"].update_general_stat(
                    "Variables", int(line[17:])
                )
                continue
            if line.startswith("c MaxWeight....:"):
                # with global_lock:
                fuzzers[fuzzer]["stats"].update_general_stat(
                    "MaxWeight", int(line[17:])
                )
                continue
            if line.startswith("c SumOfWeights.:"):
                # with global_lock:
                fuzzers[fuzzer]["stats"].update_general_stat(
                    "SumOfWeights", int(line[17:])
                )
                continue
            if line.startswith("c Hard clauses.: SATISFIABLE"):
                # with global_lock:
                fuzzers[fuzzer]["stats"].update_general_stat("satisfiable", True)
                continue
            if line.startswith("c Hard clauses.: UNSATISFIABLE"):
                # with global_lock:
                fuzzers[fuzzer]["stats"].update_general_stat("satisfiable", False)
                continue
            if line.startswith("c Best o value.:"):
                # with global_lock:
                fuzzers[fuzzer]["stats"].update_general_stat(
                    "BestOValue", int(line[17:])
                )
                continue
        if line.startswith("c name            "):
            solver_stats = True  # Start saving lines from the next line
            continue
        elif line.startswith("c faulty solvers: ") or line.startswith("c rc:"):
            solver_stats = False  # Stop saving lines
            continue
        elif solver_stats:
            tokens = re.split(r"\s+", line.strip())
            if save_stats:
                solver_name = tokens[-1]  # solver name
                if tokens[-3] != "TO":
                    solver_time = float(tokens[-3])  # solver time in s
                else:
                    solver_time = -1
                if tokens[-2] != "-":
                    solver_memory = int(tokens[-2])  # solver memory in KB
                else:
                    solver_memory = -1
                # with global_lock:
                fuzzers[fuzzer]["stats"].update_solver_stat(
                    solver_name, solver_time, solver_memory
                )

        match = re.match(pattern, line)
        if match:
            no_error = False
            solver_name, error_code, description = match.groups()
            # with global_lock:
            error_tracker.add_error(
                solver_name,
                fuzzer,
                int(error_code),
                description,
                current_wcnf,
                compare_time,
                do_minimization,
            )
    return no_error


def find_and_sort_files(directory, substring):

    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return []

    if not os.path.isdir(directory):
        print(f"The path {directory} is not a directory.")
        return []

    # Get all files in the directory that contain the given substring
    files_with_substring = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if substring in f and os.path.isfile(os.path.join(directory, f))
    ]

    # Sort files by their size (smallest first)
    # sorted_files = sorted(files_with_substring, key=os.path.getsize)
    # Sort files by their age, (newest first)
    sorted_files = sorted(files_with_substring, key=os.path.getctime, reverse=True)

    # for file in sorted_files:
    #     creation_time = os.path.getctime(file)
    #     print(f"{file}: {creation_time} (creation time)")

    return sorted_files


def combine_log_files_with_substring(directory, substring, solver, fault_code):
    global files_error_free, files_processed
    sorted_files = find_and_sort_files(directory, substring)
    # print(sorted_files)

    # Dictionary to track first occurrences of solver_name and return_code combinations
    first_occurrences = {}
    combination_key = (solver, fault_code)
    first_occurrences[combination_key] = True

    for file in sorted_files:

        # Ensure that the item is a file
        if not os.path.isfile(file):
            continue

        # Read the entire content of the file and extract the last line
        with open(file, "r") as file_to_read:
            content = file_to_read.read().strip().splitlines()
            last_line = content[-1] if content else ""

        # print(content)

        parts = os.path.basename(file).split("_")
        solver_name = parts[0]
        return_code = parts[1].split(".")[0]
        # print(f"solver name: {solver_name}, return code: {return_code}")

        # Determine if this is the first occurrence of the solver_name and return_code combination
        # only then activate the delta debugger for that combination.
        # This will be probably the smallest file for that occurence (as it occured latest in the logs)
        combination_key = (solver_name, return_code)
        do_minimization = combination_key not in first_occurrences
        first_occurrences[combination_key] = True
        error_free = check_compare_output(
            content, "DeltaDebugger", False, do_minimization
        )

        if error_free:
            files_error_free += 1

        # Check if the last line starts with "c rc"
        if last_line.startswith("c rc"):
            log_file = f"{log_path}/{solver_name}_{return_code}.log"

            # Append the content of the source file to the appropriate target file
            with open(log_file, "a", encoding="utf-8") as target_file:
                target_file.write("\n".join(content) + "\n\n")

            if os.path.exists(file):
                os.remove(file)

        files_processed += 1


def call_delta_debugger(solver, fault_code, instance):

    # print(f"Error occurences: {error_tracker.error_occurrences}")
    # global fuzzers, error_tracker
    # print("Calling delta debugger")
    compare_command = delta_debugger_compare_script + "_--solvers_" + solver
    # print(f"compare_command: {compare_command}")
    instance_basename = os.path.basename(instance)
    delta_debugger_command = f"{delta_debugger} -e 0 -r {min_path}/{solver}_{fault_code}_red-{instance_basename} -s {compare_command} {instance}"
    # print(delta_debugger_command)

    delta_debugger_output = subprocess.run(
        delta_debugger_command,
        shell=True,
        capture_output=True,
        text=True,
        check=False,  # check=False is default and does not raise an exception regardless of the exit status of the program
    )

    if delta_debugger_output.returncode != 0:
        ddrv = f"rv_{delta_debugger_output.returncode}_"
    else:
        ddrv = ""

    log_file = os.path.join(
        ddmin_log_path, f"{ddrv}ddmin_{solver}_{fault_code}_{instance_basename}.log"
    )

    with open(log_file, "w") as log:
        log.write(f"Command:\n{delta_debugger_command}\n")
        log.write(delta_debugger_output.stdout)
        log.write(delta_debugger_output.stderr)

    # Prepare prefixes to match against
    prefixes = {
        "c rnd number in file..:": "rnd_number",
        "c Reduction successful:": "reduction_successful",
        "c total solver calls..:": "total_solver_calls",
        "c av solver call time.:": "av_solver_call_time",
    }

    # Create a dictionary to hold extracted values
    extracted_data = {}

    # Process output line by line
    for line in delta_debugger_output.stdout.splitlines():
        for prefix, key in prefixes.items():
            if line.startswith(prefix):
                value_str = line[len(prefix) :].strip()
                # Convert to float if it contains a decimal point, otherwise to an integer
                value = float(value_str) if "." in value_str else int(value_str)
                extracted_data[key] = value

    combine_log_files_with_substring(
        wcnfddmin_lg_path,
        str(extracted_data.get("rnd_number", "   ")),
        solver,
        fault_code,
    )

    fuzzers["DeltaDebugger"]["stats"].compare_loops += extracted_data.get(
        "total_solver_calls", 0
    )

    if (
        delta_debugger_output.returncode == 0
        and extracted_data.get("reduction_successful", -1) == 0
    ):
        delta_debugger_output.returncode = 11

    if delta_debugger_output.returncode != 0:
        failure = 1
        success = 0
    else:
        failure = 0
        success = 1

    error_tracker.update_debugger_count(solver, fault_code, -1, success, failure)

    # print(f"delta_debugger-returncode: {delta_debugger_output.returncode}")

    if delta_debugger_output.returncode == 1:
        print(f"Parsing error in delta debugger.")
    elif delta_debugger_output.returncode == 2:
        print(f"Whole problem does not throw an error. Maybe non deterministic error.")
    elif delta_debugger_output.returncode == 3:
        print(f"Solver runs without throwing an error after reduction.")
    elif delta_debugger_output.returncode == 4:
        print(f"Reduced problem could not be copied.")
    elif delta_debugger_output.returncode == 5:
        print(
            f"The Termination Flag is active, this means the reduction might've been better."
        )
    elif delta_debugger_output.returncode == 6:
        print(f"PARSE ERROR!!! Unexpected char in parseInt.")
    elif delta_debugger_output.returncode == 7:
        print(f"PARSE ERROR! Unexpected char in parseWeight.")
    elif delta_debugger_output.returncode == 8:
        print(f"PARSE ERROR! Unexpected negative weight.")
    elif delta_debugger_output.returncode == 9:
        print(f"PARSE ERROR! Unexpected char in parseWeight.")
    elif delta_debugger_output.returncode == 10:
        print(f"Error: Mode not implemented!!")
    elif delta_debugger_output.returncode == 11:
        print(f"Could not reduce problem size.")
    elif delta_debugger_output.returncode == -2:
        print(f"  ddmin: Received SIGINT and finished successfully")
    elif delta_debugger_output.returncode == -15:
        print(f"  ddmin: Received SIGTERM and finished successfully")
    elif delta_debugger_output.returncode != 0:
        print(f"ddmin strange returncode: {delta_debugger_output.returncode}")

    # print(f"delta_debugger_output.returncode {delta_debugger_output.returncode}")

    return_code = 0
    if delta_debugger_output.returncode < 0:
        return_code = delta_debugger_output.returncode - 30
    elif delta_debugger_output.returncode > 0:
        return_code = -delta_debugger_output.returncode - 100

    return return_code


def call_fuzzer_and_solver(seed, name, fuzzer, wcnfCompare=wcnf_compare_script):
    if seed:
        wcnf = "/tmp/bug-" + str(seed) + "-" + name + ".wcnf"
        fuzzer = fuzzer + " > " + wcnf
        createWCNF = subprocess.run(fuzzer, shell=True, capture_output=True)
    else:
        wcnf = fuzzer

    sum_of_weights = check_if_valid_wcnf(wcnf)
    if sum_of_weights < 0:
        if os.path.exists(wcnf):
            os.remove(wcnf)
        # returns -1 for invalid p line
        # returns -2 for weights < 0
        # returns -3 for some strange input lines
        # returns -4 file does not exist
        return sum_of_weights
    elif seed and createWCNF.returncode != 0:
        if os.path.exists(wcnf):
            os.remove(wcnf)
        return -4
    elif upper_bound != -1 and sum_of_weights > upper_bound:
        if os.path.exists(wcnf):
            os.remove(wcnf)
        return -5

    solverOut = subprocess.run(
        wcnfCompare + " " + wcnf,
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )

    analyze_timings = analyze_fuzzer_instances or analyze_solver_timings

    if solverOut.returncode != 0:
        with global_lock:
            if check_compare_output(
                solverOut.stdout.strip().splitlines(), name, analyze_timings, True
            ):
                if seed and os.path.exists(wcnf):
                    os.remove(wcnf)
                # no error match in compare script
                return -6
    elif analyze_timings:
        with global_lock:
            check_compare_output(
                solverOut.stdout.strip().splitlines(), name, True, True
            )

    if seed and os.path.exists(wcnf):
        os.remove(wcnf)

    return solverOut.returncode


def run_program():
    global overall_execution_time, overall_non_zero_codes, overall_loops, zero_codes, non_zero_codes, execution_times, return_codes, terminate_flag
    while not terminate_flag:
        start_time = time.time()
        fuzzer_and_solver_vec = create_command()
        if not fuzzer_and_solver_vec[2]:
            break
        if fuzzer_and_solver_vec[0] == -1:
            # start delta debugger
            exit_code = call_delta_debugger(*fuzzer_and_solver_vec[1:])
        else:
            exit_code = call_fuzzer_and_solver(*fuzzer_and_solver_vec)
        elapsed_time = time.time() - start_time
        fuzzer_name = fuzzer_and_solver_vec[1]
        if fuzzer_and_solver_vec[0] == -1:
            fuzzer_name = "DeltaDebugger"
        stats = fuzzers[fuzzer_name]["stats"]
        with global_lock:
            stats.active_executions -= 1
            assert stats.active_executions >= 0
            if type(exit_code) is int and exit_code < 0:
                stats.invalid_instances[exit_code] = (
                    stats.invalid_instances.get(exit_code, 0) + 1
                )
                overall_invalid_instances[exit_code] = (
                    overall_invalid_instances.get(exit_code, 0) + 1
                )
                continue
            overall_loops += 1
            stats.loops += 1
            overall_execution_time += elapsed_time
            stats.execution_time += elapsed_time
            if exit_code != 0:
                stats.non_zero_codes += 1
                overall_non_zero_codes += 1
    if (threading.active_count() == 2):
        print_status(True)
        terminate_flag = True


def print_status(last_run = False):
    if terminate_flag:
        RED = GREEN = YELLOW = BLUE = RESET = ""
    else:
        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        RESET = "\033[0m"

    iterator = 0

    while True:
        with global_lock:
            iterator += 1
            if not terminate_flag:
                print("\033c", end="")
            print(f"{RED}=== Overall Stats ==={RESET}")
            # if overall_non_zero_codes > 0:
            #     non_zero_percent = round(overall_non_zero_codes / count, 3)
            # else
            #     non_zero_percent = 0

            compare_exit_codes = files_processed - files_error_free
            exit_codes_solvers = (
                str(overall_non_zero_codes + compare_exit_codes)
                + "/"
                + str(error_tracker.error_counter)
            )
            print(
                # f"{RED}Non-zero Exit Codes        : {overall_non_zero_codes:>14}{RESET}"
                f"{RED}Bugs: Exit Codes/Solvers   : {exit_codes_solvers:>14}{RESET}"
            )
            print(f"{YELLOW}Total Executions (Script)  : {overall_loops:>14}{RESET}")
            print(
                f"{YELLOW}Execution Time             : {round(overall_execution_time, 2):>14}{RESET}"
            )
            print(
                f"{YELLOW}Number Threads             : {overall_number_threads:>14}{RESET}"
            )
            error_counter, error_details = error_tracker.all_errors()
            # print(error_tracker.
            print(f"{YELLOW}Errors Found               : {error_counter:>14}{RESET}")
            print(f"{YELLOW}Solver(Bug):Count          : {error_details:>14}{RESET}")
            print(f"{YELLOW}================================={RESET}")

            for fuzzer_name, fuzzer_details in fuzzers.items():
                stats = fuzzer_details["stats"]
                print(f"{YELLOW}===== {fuzzer_name} Stats ====={RESET}")

                exit_codes_solvers = (
                    str(stats.non_zero_codes)
                    + "/"
                    + str(error_tracker.sum_errors_for_fuzzer(fuzzer_name))
                )
                if fuzzer_name == "DeltaDebugger":
                    compare_exit_codes = files_processed - files_error_free
                    exit_codes_solvers = f"{compare_exit_codes}/{error_tracker.sum_errors_for_fuzzer(fuzzer_name)}"
                    active_execs = fuzzers["DeltaDebugger"]["stats"].active_executions
                    print(
                        f"{BLUE}Active DeltaDebugger Runs  : {active_execs:>14}{RESET}"
                    )
                    print(
                        f"{BLUE}Finished DeltaDebugger Runs: {stats.loops:>14}{RESET}"
                    )
                    print(
                        f"{BLUE}Bugs: Exit Codes/Solvers   : {exit_codes_solvers:>14}{RESET}"
                    )
                    print(
                        f"{BLUE}Compare Executions         : {stats.compare_loops:>14}{RESET}"
                    )
                else:
                    print(
                        f"{BLUE}Bugs: Exit Codes/Solvers   : {exit_codes_solvers:>14}{RESET}"
                    )
                    print(
                        f"{BLUE}Total Executions           : {stats.loops:>14}{RESET}"
                    )
                average_time = (
                    0 if stats.loops == 0 else stats.execution_time / stats.loops
                )
                avg_tot = (
                    str(round(average_time, 2))
                    + "/"
                    + str(round(stats.execution_time, 2))
                )
                print(f"{BLUE}Time Avgerage/Total        : {avg_tot:>14}{RESET}")
                exclude_fuzzer = "DeltaDebugger"
                bug_string = "Bugs: Unique/All           : "
                if fuzzer_name == "DeltaDebugger":
                    exclude_fuzzer = None
                    bug_string = "Bugs: Only DDMin/All       : "
                unique_error_counter, error_details = (
                    error_tracker.unique_fuzzer_errors(fuzzer_name, exclude_fuzzer)
                )
                error_counter, error_details = error_tracker.fuzzer_errors(fuzzer_name)
                unique_all_rc_counter = (
                    str(unique_error_counter) + "/" + str(error_counter)
                )
                print(f"{BLUE}{bug_string}{unique_all_rc_counter:>14}{RESET}")
                if terminate_flag:
                    if unique_error_counter > 0:
                        print(
                            f"{BLUE}Unique Solver(Bug):Count   : {error_details:>14}{RESET}"
                        )
                    print(
                        f"{BLUE}Solver(Bug):Count          : {error_details:>14}{RESET}"
                    )
                if stats.invalid_instances:
                    invalid_string = "{"
                    for ret_val, count in stats.invalid_instances.items():
                        description = invalid_description.get(
                            ret_val, f"return value {100-ret_val}"
                        )
                        invalid_string += f"{description}: {str(count)}; "
                    invalid_string += "}"
                    print(
                        f"{BLUE}Invalid return codes       : {str(invalid_string):>14}{RESET}"
                    )
                if analyze_fuzzer_instances:
                    stats.dump_instance_analysis()
                if analyze_solver_timings:
                    stats.dump_solver_analysis()
                if analyze_last_xxx:
                    stats.dump_fuzzer_stats()
            print(f"{YELLOW}==== Bug Descriptions ============================={RESET}")
            for code, description in sorted(error_tracker.error_descriptions.items()):
                print(f"{RED}Error {code}: {description}{RESET}")
        
        if (last_run):
            return

        # if iterator % 1 == 1 and not terminate_flag:
        if not terminate_flag:
            check_disk_space(".")

        if int(overall_loops) < 100:
            time_to_sleep = 2
        else:
            time_to_sleep = int(math.log10(int(overall_loops)))

        combine_log_files(lg_path)
        combine_log_files(wcnfddmin_lg_path)

        cnt = 0
        while (not terminate_flag) and cnt < 50:
            time.sleep(time_to_sleep / 50)
            cnt += 1

        if terminate_flag:
            break


def check_disk_space(folder_path):
    global time_first_time, free_first_time
    total, used, free = shutil.disk_usage(folder_path)
    print("")
    disk_space_string = f"Free Disk Space, abort if < {min_free_disk_space} GB: {total / (2**30):.2f} total, {used / (2**30):.2f} used, {free / (2**30):.2f} free,"
    # print(f"Disk Space Usage. Abort if less than {min_free_disk_space} GB of free space.")
    # print(
    #     f"Total, Used, Free (in GB): \t\t\t{total / (2**30):.2f}, {used / (2**30):.2f}, {free / (2**30):.2f}"
    # )

    if time_first_time == -1:
        time_first_time = int(time.time())
        free_first_time = free / (2**20)
    else:
        MBperSecond = -(free_first_time - (free / (2**20))) / (
            time_first_time - int(time.time())
        )
        if MBperSecond != 0:
            seconds_until_full = (
                free / (2**20) - min_free_disk_space * (2**10)
            ) / MBperSecond
        else:
            seconds_until_full = 2**64
        if seconds_until_full > 0 and seconds_until_full < 3600 * 48:
            print(
                f"{disk_space_string} {MBperSecond:.2f} MB/Second --> {seconds_until_full/3600:.2f} hours until full"
            )
        # print(
        #     f"MB/Second --> time until full:\t\t{MBperSecond:.2f} --> {seconds_until_full/3600:.2f} Hours"
        # )

    # Check if free space is below a certain threshold
    if free / (2**30) < min_free_disk_space:
        print("Warning: Low disk space!")
        raise KeyboardInterrupt("Low disk space - Interrupting!")


def main():
    global ddmin_log_path, min_path, wcnfddmin_lg_path, delta_debugger_compare_script, minimize, analyze_last_xxx, folder, file_list, analyze_fuzzer_instances, analyze_solver_timings, save_solver_timings, terminate_flag, timeout, overall_number_threads, wcnf_compare_script, faulty_wcnf_location, log_path, lg_path, upper_bound, threads, csv_filename, location
    parser = argparse.ArgumentParser(
        description="This is a parallel MaxSAT fuzzing script!!"
    )
    parser.add_argument(
        "-t",
        "--threads",
        default=1,
        help="Number of threads to run this script on.",
    )
    parser.add_argument(
        "--timeout",
        default=20,
        type=int,
        help="Timeout for each complete MaxSAT Solver (as argument for compare script).",
    )
    parser.add_argument(
        "--minimize",
        default=5,
        type=int,
        help="Start delta debugger for the first X occurences of a error (Default == 5).",
    )
    parser.add_argument(
        "--upperBound",
        default=-1,
        type=int,
        help="Upper bound for the sum of weights in the fuzzer. (Default == 2^64 - 1)",
    )
    parser.add_argument(
        "--analyzeFuzzer",
        action="store_true",
        help="Analyze all instances of each fuzzer.",
    )
    parser.add_argument(
        "--analyzeSolver",
        action="store_true",
        help="Analyze timing and memory usage of each solver. Writes a csv file at the end.",
    )
    parser.add_argument(
        "--saveSolverTimings",
        action="store_true",
        help="Writes a csv file with instance and timing information of each solver. Sets --analyzeSolver and --analyzeFuzzer flags automatically to True.",
    )
    parser.add_argument(
        "--bigNumbers",
        action="store_true",
        help="Give numbers > 2**32 a different fault code (standard compare.py behaviour).",
    )
    parser.add_argument(
        "--lastxxx",
        type=int,
        default=0,
        help="Give average/median values for the last xxx (>= 100) instances -- for fuzzer tuning.",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Run the compare script in parallel on a given folder instead of randomly generating instances.",
    )

    # argcomplete.autocomplete(parser)
    args = parser.parse_args()
    analyze_last_xxx = args.lastxxx
    if args.lastxxx:
        args.saveSolverTimings = True
    analyze_fuzzer_instances = args.analyzeFuzzer
    analyze_solver_timings = args.analyzeSolver
    if args.saveSolverTimings:
        save_solver_timings = True
        analyze_solver_timings = True
        analyze_fuzzer_instances = True
    else:
        save_solver_timings = False

    if args.folder is not None:
        if not os.path.exists(args.folder) or not os.path.isdir(args.folder):
            print("The given folder does not exist.")
            exit(1)
        files = os.listdir(args.folder)
        file_list = [
            args.folder + "/" + file
            for file in files
            if file.endswith(".wcnf") or file.endswith(".xz")
        ]
        if not file_list:
            print("The given folder is empty.")
            exit(1)
        folder = os.path.basename(os.path.normpath(args.folder))
        fuzzers[folder] = {"stats": StatsTracker()}
        folder = args.folder

    minimize = args.minimize
    if minimize != 0:
        fuzzers["DeltaDebugger"] = {"stats": StatsTracker()}

    upper_bound = args.upperBound
    overall_number_threads = int(args.threads)
    # Get the current time as a struct_time object
    now = time.localtime()
    # Calculate the number of seconds since the start of the day
    seconds_since_start_of_day = now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec
    # Format the seconds as a five-digit string with leading zeros
    formatted_seconds = f"{seconds_since_start_of_day:05d}"
    location = f"Logs/{date.today()}-{formatted_seconds}-runwcnfuzz/"
    faulty_wcnf_location = location + "FaultyWCNFs"
    lg_path = location + "FaultLogs"
    wcnfddmin_lg_path = location + "FaultLogsDDMin"
    log_path = location + "FaultOverview"
    ddmin_log_path = location + "DeltaDebuggerLogs"
    min_path = location + "FaultsMinimized"
    if args.saveSolverTimings:
        csv_filename = location + "SolverTimings.csv"
    timeout = args.timeout
    wcnf_compare_script += f" --saveWCNFFolder {faulty_wcnf_location} --logPath {lg_path} --timeout {args.timeout}"
    delta_debugger_compare_script += f"_--saveWCNFFolder_{faulty_wcnf_location}_--logPath_{wcnfddmin_lg_path}_--timeout_{args.timeout}_--logAll"
    if not args.bigNumbers:
        wcnf_compare_script += " --noWeightSplit"
        delta_debugger_compare_script += "_--noWeightSplit"
    # print(wcnf_compare_script)
    # time.sleep(1)
    if not os.path.exists("Logs") or not os.path.isdir("Logs"):
        os.makedirs("Logs")
    if not os.path.exists(location) or not os.path.isdir(location):
        os.makedirs(location)
    if not os.path.exists(faulty_wcnf_location) or not os.path.isdir(
        faulty_wcnf_location
    ):
        os.makedirs(faulty_wcnf_location)
    if not os.path.exists(log_path) or not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.exists(lg_path) or not os.path.isdir(lg_path):
        os.makedirs(lg_path)
    if not os.path.exists(wcnfddmin_lg_path) or not os.path.isdir(wcnfddmin_lg_path):
        os.makedirs(wcnfddmin_lg_path)
    if not os.path.exists(min_path) or not os.path.isdir(min_path):
        os.makedirs(min_path)
    if not os.path.exists(ddmin_log_path) or not os.path.isdir(ddmin_log_path):
        os.makedirs(ddmin_log_path)

    try:
        for i in range(overall_number_threads):
            t = threading.Thread(target=run_program)
            threads.append(t)
            t.start()

        print_status()

        if terminate_flag:
            cleanup()

    except KeyboardInterrupt:
        print("\nCaught Keyboard Interrupt")
        terminate_flag = True
        cleanup()


if __name__ == "__main__":
    main()
