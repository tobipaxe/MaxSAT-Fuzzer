#!/usr/bin/env python3
# import shutil  # copying wcnf files
# import lzma  # compressing wncf files
import random
import os.path
import argparse
import subprocess
import sys
import time
from datetime import date
import statistics
import psutil
import wcnfTool
from config import solvers, timeoutFactor, mempeakFactor


def is_valid_file(arg):
    if not os.path.exists(arg):
        print("The file %s does not exist." % arg)
        exit(1)


parser = argparse.ArgumentParser(
    description="This is a awesome MaxSAT Solver comparison script."
)
parser.add_argument(
    "wcnfFile", help="WCNF file which should be checked (in relation to current path)."
)
parser.add_argument(
    "--logging",
    action="store_true",
    help="Activates logging. A logfile for ALL SOLVER FROM SOLVERS LIST is created. (if --solver is not used this equals all solvers)",
)
parser.add_argument(
    "--logAll",
    action="store_true",
    help="Activates logging. A logfile for EVERY erroneous solver is created, independent on if the return code is due to only one specific solver (--solver).",
)
parser.add_argument(
    "--saveWCNF",
    action="store_true",
    help=" Copies the WCNF to folder: (default: <date>__WCNFs/).",
)
parser.add_argument(
    "--saveWCNFFolder",
    default=str(date.today()) + "__WCNFs",
    help="Add error WCNFs to folder (default: <date>__WCNFs/).",
)
parser.add_argument(
    "--logPath",
    default=str(date.today()) + "__FaultLogs",
    help="Add logfiles for each erroneous solver in folder (default ./<date>_FaultLogs).",
)
parser.add_argument(
    "--timeout", default=20, type=int, help="Timeout for each MaxSAT Solver."
)
parser.add_argument(
    "--solvers",
    default=solvers.keys(),
    type=lambda s: [item for item in s.split(",")],
    help="Only return an invalid return code if a solver from this list is faulty, but all solvers run to have a better mayority voting. Write them without empty space. --solvers a,b,c --> a,b,c in "
    + str(solvers.keys()),
    required=False,
)
parser.add_argument(
    "--solverList",
    default=solvers.keys(),
    type=lambda s: [item for item in s.split(",")],
    help="Only run solvers from the following list. This can be helpful for delta debugging. Write them without empty space. --solvers a,b,c --> a,b,c in "
    + str(solvers.keys()),
    required=False,
)
parser.add_argument(
    "--satSolver",
    nargs="?",
    default="./kissat/build/kissat",
    help="How to call the SAT solver to check the UNSAT results -- used by wcnfTool (total path or in relation to this script directory).",
)
parser.add_argument(
    "--reWriteAllWCNFs",
    action="store_true",
    help="Normally the wcnf given in old/new format won't be rewritten. But incorrect header (weights / variables / clauses) can lead to wrong results.",
)
parser.add_argument(
    "--noWeightSplit",
    action="store_true",
    help="Normally the weight is splitted in between small (up to 2^^32) and big sum of weights (smaller than 2^^64). For big weights the 40 is added to the fault number. This is deactivated with this flag!",
)
parser.add_argument(
    "--printModel",
    action="store_true",
    help="Print the model if a solver returns SAT.",
    default=False,
)

start_time = time.time()
args = parser.parse_args()

is_valid_file(str(args.wcnfFile))


def resolve_solvers(arguments):
    resolved = []
    for arg in arguments:
        # Check if it's directly a key
        if arg in solvers:
            resolved.append(arg)
        else:
            # Check if it's a "short" alias
            matched_key = None
            for key, details in solvers.items():
                if details.get("short") == arg:
                    matched_key = key
                    break

            if matched_key:
                resolved.append(matched_key)
            else:
                print(f"{arg} is not a recognized MaxSAT solver from the list:")
                print(list(solvers.keys()))
                exit(1)
    return resolved


args.solvers = resolve_solvers(args.solvers)
args.solverList = resolve_solvers(args.solverList)

toDelete = []
for solarg in solvers.keys():
    if solarg not in args.solverList:
        toDelete.append(solarg)

for solarg in toDelete:
    del solvers[solarg]


# seed to add to each file created in /tmp/, try to add seed of current filename:
seed = ""
if seed == "" and any(
    char.isdigit() for char in os.path.splitext(os.path.basename(args.wcnfFile))[0]
):
    seed = str(
        int(
            "".join(
                filter(
                    str.isdigit, os.path.splitext(os.path.basename(args.wcnfFile))[0]
                )
            )
        )
    )
if seed == "":
    seed = str(random.randrange(1, 2**32 - 1))
long_seed = seed + "-" + str(random.randrange(1, 2**32 - 1))

current_directory = os.getcwd()
current_script_directory = os.path.dirname(os.path.abspath(__file__))

if args.satSolver[0] == ".":
    args.satSolver = current_script_directory + args.satSolver[1:]
if args.logging or args.logAll:
    if not os.path.exists(args.logPath):
        os.makedirs(args.logPath)

is_valid_file(str(args.satSolver))

anytime_timeout = round(random.uniform(0.1, 1.5), 4)
# print(f"anytime_timeout: {anytime_timeout}")

# check needed needed input formats and create needed output formats
newwcnf = ""
oldwcnf = ""
mps = ""
mps_itad = ""
wasxzBefore = False
wcnfTool.parse_wcnf(args.wcnfFile)
if args.wcnfFile.endswith(".xz"):
    wasxzBefore = True
    args.wcnfFile = f"/tmp/{seed}.wcnf"
    wcnfTool.WriteToFile(args.wcnfFile, "new", False)
    wcnfTool.wcnfInputFormat == "new"
if wcnfTool.wcnfInputFormat == "new" and not args.reWriteAllWCNFs:
    newwcnf = os.path.abspath(args.wcnfFile)
elif wcnfTool.wcnfInputFormat == "old" and not args.reWriteAllWCNFs:
    oldwcnf = os.path.abspath(args.wcnfFile)

for solver in solvers:
    if solvers[solver]["input_format"] == "wcnf":
        if newwcnf == "":
            newwcnf = "/tmp/" + str(long_seed) + ".new.wcnf"
            wcnfTool.WriteToFile(newwcnf, "new", False)
            is_valid_file(newwcnf)
        solvers[solver]["input_file"] = newwcnf
    elif solvers[solver]["input_format"] == "oldwcnf":
        if oldwcnf == "":
            oldwcnf = "/tmp/" + str(long_seed) + ".old.wcnf"
            wcnfTool.WriteToFile(oldwcnf, "old", False)
            is_valid_file(oldwcnf)
        solvers[solver]["input_file"] = oldwcnf
    elif solvers[solver]["input_format"] == "mps":
        if mps == "":
            mps = "/tmp/" + str(long_seed) + ".wcnf.mps"
            wcnfTool.WriteToFile(mps, "mps", False)
            is_valid_file(mps)
        solvers[solver]["input_file"] = mps
    elif solvers[solver]["input_format"] == "mps_itad":
        if mps_itad == "":
            mps_itad = "/tmp/" + str(long_seed) + ".wcnf.itad.mps"
            wcnfTool.WriteToFile(mps_itad, "mps_itad", False)
            is_valid_file(mps_itad)
        solvers[solver]["input_file"] = mps_itad


def terminate_process_and_children(process):
    killed = False
    try:
        parent = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        # The process does not exist anymore
        return (process.communicate(), killed)

    # Get child processes recursively
    children = parent.children(recursive=True)

    # Try to terminate all child processes
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass  # The process does not exist anymore

    process.terminate()
    gone, alive = psutil.wait_procs([parent] + children, timeout=3)
    for p in alive:
        try:
            killed = True
            # os.kill(p.pid, signal.SIGKILL)
            p.kill()
        except ProcessLookupError:
            pass  # The process does not exist anymore

    return (process.communicate(), killed)


def SolverCall(solver):
    if solver["type"] == "anytime":
        solver["anytime"] = True
        time_out = anytime_timeout
    elif solver["type"] == "certified":
        solver["anytime"] = False
        time_out = 30 * args.timeout
    else:
        solver["anytime"] = False
        time_out = args.timeout

    solver["solver_call"] = solver["solver_call"].replace(
        "./", current_script_directory + "/"
    )

    # application = "/usr/bin/timeout --verbose --kill-after=10 " + str(args.timeout) + \
    #               " /usr/bin/time -q -f \"c mempeak %M\" " + \
    #               solver["solver_call"] + " " + solver["input_file"]

    command_str = solver["solver_call"] + " " + solver["input_file"]
    command = ["/usr/bin/time", "-q", "-f", "c mempeak %M"] + command_str.split()
    killed = False
    solver_start_time = time.time()
    # print(command)

    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as process:
        try:
            while True:
                elapsed_time = time.time() - solver_start_time  # Calculate elapsed time
                # Poll process for status (Non-Blocking)

                return_code = process.poll()
                if return_code is not None:  # Process has terminated
                    stdout, stderr = process.communicate()  # Collect final outputs
                    break  # Exit the loop as process has terminated
                if (
                    elapsed_time > time_out
                ):  # Overall timeout, change it to your desired timeout
                    print("Exception due to elapsed time:")
                    raise Exception("Overall Timeout Occured")
                stdout, stderr = process.communicate(timeout=time_out)

        except Exception:
            (stdout, stderr), killed = terminate_process_and_children(process)
            if not solver["anytime"]:
                solver["s-status"] = "TIMEOUT"
            # Capturing whatever was output before the error/exception

    solver_end_time = time.time()

    # Output results and time
    solver["time"] = solver_end_time - solver_start_time
    solver["return_code"] = process.returncode
    solver["stdout"] = stdout
    solver["stderr"] = ""

    for line in stderr.split("\n"):
        if line.startswith("c mempeak "):
            solver["mempeak"] = int(line[10:])
        elif line:
            solver["stderr"] += line + "\n"

    if (solver["anytime"] and killed) or solver.get("s-status", "") == "TIMEOUT":
        solver["s-status"] = "TIMEOUT"
        solver["checked_result"] = "TIMEOUT"
        del solver["time"]
        return

    vline = ""
    for line in solver["stdout"].split("\n"):
        if line.startswith("s "):
            solver["s-status"] = line[2:]
            if solver["s-status"] == "SATISFIABLE":
                solver["s-status"] = "OPTIMUM FOUND"
        if "c SCIP result: UNSATISFIABLE" == line:
            solver["s-status"] = "UNSATISFIABLE"
        if "UNSATISFIABLE" == line:
            solver["s-status"] = "UNSATISFIABLE"
        if line.startswith("o "):
            if line[2:] != "-oo":
                try:
                    solver["o-value"] = int(line[2:])
                except ValueError:
                    solver["ERROR_solver"] = f"string in o-value: {line[2:]}"
            else:
                # only hard clauses -- but they are satisfiable
                solver["o-value"] = 0
        if line.startswith("v "):
            solver["model"] = line[2 : 2 + 2 * wcnfTool.vars]
            if len(line) > (2 + (5 * wcnfTool.vars)):
                vline = line
                if len(line) > (2 + (10 * wcnfTool.vars)):
                    solver["model_length"] = len(line) - 2
                    line = (
                        solver["model"]
                        + " + "
                        + str(solver["model_length"])
                        + " additional literals."
                    )

        if (
            "Error" in line
            or "ERROR" in line
            or "error" in line
            and "0 errors" not in line
        ):
            solver["stdout_error"] = line
    if vline:
        solver["stdout"] = solver["stdout"].replace(
            vline, vline[: 2 + 2 * wcnfTool.vars], 1
        )

    if "s-status" not in solver:
        solver["s-status"] = "EMPTY"

    if solver["s-status"] == "OPTIMUM FOUND":
        if "model" not in solver:
            solver["ERROR_solver"] = "opt_but_no_model"
        elif "o-value" not in solver:
            solver["ERROR_solver"] = "opt_but_no_o-value"
    else:
        if "model" in solver and "o-value" in solver:
            solver["ERROR_solver"] = "model_and_o-value_but_no_opt"

    if "model" in solver and solver["model"]:
        wcnfTool.CheckModel(solver["model"])

        # wcnfTool.bestSolution should be one of "OPTIMUM FOUND", "UNSATISFIABLE", "MODEL ERROR"
        solver["ver_model"] = wcnfTool.bestSolution
        solver["ver_o-value"] = wcnfTool.bestOptimum
        # this should never happen!!
        assert (
            wcnfTool.bestSolution
        ), "ERROR: No return value of the model checker. Should never happen."

    solver["checked_result"] = ""

    if solutionHardClauses == "UNSATISFIABLE" and solver["s-status"] == "UNSATISFIABLE":
        solver["checked_result"] = "UNSATISFIABLE"
    elif solutionHardClauses == "SATISFIABLE" and solver["s-status"] == "OPTIMUM FOUND":
        if (
            "ver_o-value" in solver
            and "o-value" in solver
            and solver["ver_o-value"] == solver["o-value"]
        ):
            solver["checked_result"] = "OPTIMUM FOUND"
        else:
            solver["checked_result"] = "FALSE VALUE"
    elif solutionHardClauses == "UNSATISFIABLE" and solver["s-status"] == "OPTIMUM FOUND":
        solver["checked_result"] = "UNSAT BUT VALUE"
    elif solutionHardClauses == "SATISFIABLE" and solver["s-status"] == "UNSATISFIABLE":
        solver["checked_result"] = "FALSE UNSAT"
    else:
        solver["checked_result"] = "UNKNOWN"


instance_stats = "c HardClauses..: " + str(wcnfTool.nbHard) + "\n"
instance_stats += "c SoftClauses..: " + str(wcnfTool.nbSoft) + "\n"
instance_stats += "c Variables....: " + str(wcnfTool.vars) + "\n"
instance_stats += "c MaxWeight....: " + str(wcnfTool.maxWeight) + "\n"
instance_stats += "c SumOfWeights.: " + str(wcnfTool.sumOfWeights) + "\n"

wcnfTool.CheckIfHardClausesSAT(args.satSolver, long_seed)
solutionHardClauses = wcnfTool.solutionHardClauses

solversToRemove = []
anytime_solver = False
for solver in solvers:
    if solvers[solver]["type"] == "anytime":
        anytime_solver = True
    if solvers[solver].get("upper_bound", 2**64) < wcnfTool.sumOfWeights:
        solversToRemove.append(solver)

for solver in solversToRemove:
    del solvers[solver]

if len(solvers) < 2:
    print(solvers.keys())
    print(
        "ERROR: Only "
        + str(len(solvers))
        + " solvers remain. These are too few solvers to compare.",
        file=sys.stderr,
    )
    exit(0)

minVerifiedOValue = 2**64
# minValueHitBySolver = []
# solutionHardClausesHitBySolver = []
solverid = 0
totalTime = 0
numberTimeouts = 0

for solverName, solver in solvers.items():
    solverid += 1
    SolverCall(solver)
    if len(args.solvers) == 1:
        solver["id"] = 0
    else:
        solver["id"] = solverid
    if solver.get("checked_result", "") != "TIMEOUT":
        totalTime += solver["time"]
    else:
        numberTimeouts += 1
    solver["name"] = solverName

    if (
        solver["checked_result"] == "OPTIMUM FOUND"
        and minVerifiedOValue >= solver["o-value"]
        and solver["o-value"] >= 0
    ):
        if minVerifiedOValue > solver["o-value"]:
            minVerifiedOValue = solver["o-value"]
    #         minValueHitBySolver = []
    #         minValueHitBySolver.append(solverName)
    #     else:
    #         minValueHitBySolver.append(solverName)
    # elif solver["checked_result"] == "UNSATISFIABLE":
    #     solutionHardClausesHitBySolver.append(solverName)

# remove newly generated input files
if wcnfTool.wcnfInputFormat == "new" and oldwcnf:
    os.remove(oldwcnf)
elif wcnfTool.wcnfInputFormat == "old" and newwcnf:
    os.remove(newwcnf)

if mps:
    os.remove(mps)

if mps_itad:
    os.remove(mps_itad)

# if wasxzBefore:
#     os.remove(args.wcnfFile)

mempeaks = [
    solvers[solver]["mempeak"] for solver in solvers if "mempeak" in solvers[solver]
]
if mempeaks:
    total_mp = sum(mempeaks)
    maximum_mp = max(mempeaks)
    minimum_mp = min(mempeaks)
    average_mp = total_mp / len(mempeaks)
    median_mp = statistics.median(mempeaks)
    instance_stats += (
        "c Mempeaks (KB): "
        + str(minimum_mp)
        + " min, "
        + str(maximum_mp)
        + " max, "
        + str(round(average_mp))
        + " avg, "
        + str(total_mp)
        + " sum, "
        + str(round(median_mp, 2))
        + " med, "
        + str(len(mempeaks))
        + " count\n"
    )

timings = [solvers[solver]["time"] for solver in solvers if "time" in solvers[solver]]
if timings:
    total_ti = sum(timings)
    maximum_ti = max(timings)
    minimum_ti = min(timings)
    average_ti = total_ti / len(timings)
    median_ti = statistics.median(timings)
    instance_stats += (
        "c solver times.: "
        + str(round(minimum_ti, 2))
        + " min, "
        + str(round(maximum_ti, 2))
        + " max, "
        + str(round(average_ti, 2))
        + " avg, "
        + str(round(total_ti, 2))
        + " sum, "
        + str(round(median_ti, 2))
        + " med, "
        + str(len(timings))
        + " count (without timeouts)"
    )

BiggerUINT32 = 1000 if int(wcnfTool.sumOfWeights) >= 2**32 else 0
if args.noWeightSplit:
    BiggerUINT32 = 0

if int(wcnfTool.sumOfWeights) > 2**64 - 1:
    print("c Sum of weights are bigger than 2^64-1; tool not tested for those numbers!")
if int(wcnfTool.maxWeight) > 2**63:
    print("c Biggest weight is bigger than 2^63; tool not tested for those numbers!")


# for solver in solvers:
for solverName, solver in solvers.items():
    # Calculate potential timeout failures
    if solver["checked_result"] == "TIMEOUT" and not (  # it is a timeout
        len(solver) - len(timings) < len(solvers) / 2
        and median_ti * timeoutFactor  # less than half of the solvers have a timeout
        < args.timeout
    ):  # the actual timeout is bigger than factor * median of all times
        solver["checked_result"] = "TIMEOUT -- ignored"

    if mempeaks and solver.get("mempeak", -1) > mempeakFactor * median_mp:
        solver["mempeak_error"] = True
    else:
        solver["mempeak_error"] = False

overall_return_code = 0
return_code = 0
faulty_solvers = []
logging_files = []

for solverName, solver in solvers.items():
    solver["error_code"] = 0

    # X
    if solver["checked_result"].startswith("TIMEOUT") and solver["mempeak_error"]:
        solver["error_code"] = 501 + BiggerUINT32
        solver["fault_description"] = (
            "POTENTIAL ERROR: TIMEOUT and MEMPEAK "
            + " Timeout and Memory peak ("
            + str(solver["mempeak"])
            + ") is "
            + str(mempeakFactor)
            + " times bigger than the median memory peak."
        )
    elif solver["checked_result"] == "TIMEOUT":
        solver["error_code"] = 502 + BiggerUINT32
        solver["fault_description"] = (
            "POTENTIAL ERROR: TIMEOUT is "
            + str(timeoutFactor)
            + " times bigger than median time of all other solvers."
        )
    elif (
        solver.get("return_code", 0) != 0
        and solver.get("return_code", 0) != 10
        and solver.get("return_code", 0) != 20
        and solver.get("return_code", 0) != 30
    ):
        curr_rc = solver.get("return_code", 0)
        solver["error_code"] = curr_rc + BiggerUINT32
        solver["fault_description"] = (
            f"Invalid Return Code of MaxSAT solver == {curr_rc}"
        )
    elif (
        not solver["anytime"]
        and minVerifiedOValue == 2**64
        and solutionHardClauses == "SATISFIABLE"
    ):
        solver["error_code"] = 601 + BiggerUINT32
        solver["fault_description"] = (
            "All solver have a wrong result, as hard clauses are SATISFIABLE but no solver found a correct solution."
        )
    elif solutionHardClauses == "SATISFIABLE" and solver["s-status"] == "UNSATISFIABLE":
        solver["error_code"] = 602 + BiggerUINT32
        solver["fault_description"] = (
            "Hard clauses are SATISFIABLE, but solver states s UNSATISFIABLE."
        )
    elif solver.get("checked_result", "") == "UNSAT BUT VALUE":
        # elif solutionHardClauses == "UNSATISFIABLE" and solver["s-status"] == "OPTIMUM FOUND":
        solver["error_code"] = 603 + BiggerUINT32
        solver["fault_description"] = (
            "Verifier returned, that hard clauses are UNSATISFIABLE but solver states otherwise."
        )
    elif solver.get("checked_result", "") == "FALSE UNSAT":
        solver["error_code"] = 604 + BiggerUINT32
        solver["fault_description"] = (
            "s UNSATISFIABLE but the hard clauses are satisfiable."
        )
    elif solver["s-status"] == "EMPTY":
        solver["error_code"] = 605 + BiggerUINT32
        solver["fault_description"] = "s status line NOT in solver output."
    elif not solver["anytime"] and solver.get("checked_result", "") == "UNKNOWN":
        solver["error_code"] = 606 + BiggerUINT32
        solver["fault_description"] = (
            "Solver status = "
            + solver["s-status"]
            + " Unexpected result either in the status line."
        )
    elif solver.get("ver_model", "") == "MODEL ERROR":
        solver["error_code"] = 607 + BiggerUINT32
        solver["fault_description"] = (
            "Verifier returned, that given model is too small."
        )
    elif solver.get("ver_model", "") == "UNSATISFIABLE" and solver["model"]:
        solver["error_code"] = 608 + BiggerUINT32
        solver["fault_description"] = (
            "Verifier returned that given model is UNSATISAFIABLE."
        )
    # should never happen, as before the solver"checked_result is unknown
    # elif solver["s-status"] == "UNKNOWN":
    #     solver["error_code"] = 13 + BiggerUINT32
    #     solver["fault_description"] = "s UNKNOWN status line in solver output."
    elif solver.get("ERROR_solver", "").startswith("string in o-value"):
        solver["error_code"] = 609 + BiggerUINT32
        solver["fault_description"] = f"s {solver['ERROR_solver']}"
    elif solver.get("ERROR_solver", "") == "opt_but_no_model":
        solver["error_code"] = 610 + BiggerUINT32
        solver["fault_description"] = "s OPTIMUM FOUND - but no model given"
    elif solver.get("ERROR_solver", "") == "opt_but_no_o-value":
        solver["error_code"] = 611 + BiggerUINT32
        solver["fault_description"] = "s OPTIMUM FOUND - but no o value given"
    elif (
        not solver["anytime"]
        and solver.get("ERROR_solver", "") == "model_and_o-value_but_no_opt"
    ):
        solver["error_code"] = 612 + BiggerUINT32
        solver["fault_description"] = (
            "s status line is NOT s OPTIMUM FOUND but model and o value are given"
        )
    elif solver.get("o-value", 2**64) < 0:
        solver["error_code"] = 613 + BiggerUINT32
        solver["fault_description"] = (
            "The given o value is negative, probably because of an overflow."
        )
    elif (
        solver.get("o-value", -1) != solver.get("ver_o-value", -1)
        and solver.get("ver_o-value", 2**64) != minVerifiedOValue
        and solver.get("o-value", 2**64) != minVerifiedOValue
    ):
        solver["error_code"] = 650 + BiggerUINT32
        solver["fault_description"] = (
            "MaxSAT solver o-values given by solver, model and the minimal o value are three different values."
        )
    elif (
        solver.get("o-value", -1) == minVerifiedOValue
        and solver.get("ver_o-value", -1) != minVerifiedOValue
    ):
        solver["error_code"] = 651 + BiggerUINT32
        solver["fault_description"] = (
            "MaxSAT o-value equals minimal o-value BUT given model has a bigger o-value."
        )
    elif (
        solver.get("o-value", -1) > solver.get("ver_o-value", -1)
        and solver.get("ver_o-value", -1) == minVerifiedOValue
    ):
        solver["error_code"] = 652 + BiggerUINT32
        solver["fault_description"] = (
            "MaxSAT solver o value is bigger than the o-value of its model which equals the minimal o-value."
        )
    elif (
        solver.get("o-value", 2**64) < solver.get("ver_o-value", -1)
        and solver.get("ver_o-value", -1) == minVerifiedOValue
    ):
        solver["error_code"] = 653 + BiggerUINT32
        solver["fault_description"] = (
            "MaxSAT solver o value is smaller than the o-value of its model, but the o-value of it's model equals the minimal o-value.."
        )
    elif (
        solver.get("o-value", 2**64) < solver.get("ver_o-value", -1)
        and solver.get("ver_o-value", -1) == minVerifiedOValue
    ):
        solver["error_code"] = 654 + BiggerUINT32
        solver["fault_description"] = (
            "MaxSAT solver o value is smaller than the o-value of its model, but the o-value of it's model equals the minimal o-value.."
        )
    elif (
        not solver["anytime"]
        and solver.get("ver_o-value", -1) > minVerifiedOValue
        and solver.get("ver_o-value", -2) == solver.get("o-value", -1)
    ):
        solver["error_code"] = 655 + BiggerUINT32
        solver["fault_description"] = (
            "o-value of MaxSAT solver model equals o value of the solver but it is bigger than the minimal o-value."
        )
    elif not solver["anytime"] and (
        solver.get("o-value", -1) != solver.get("ver_o-value", -1)
        or solver.get("ver_o-value", 2**64) != minVerifiedOValue
        or solver.get("o-value", 2**64) != minVerifiedOValue
    ):
        solver["error_code"] = 656 + BiggerUINT32
        solver["fault_description"] = "The o-values are otherwise inconsistent."
    elif solver.get("model_length", -1) != -1:
        solver["error_code"] = 701 + BiggerUINT32
        solver["fault_description"] = (
            f"No fault but the length of the model is {solver.get('model_length', -1)} but we only have {wcnfTool.vars} variables."
        )
    elif solver.get("stderr", "") != "":
        solver["error_code"] = 702 + BiggerUINT32
        solver["fault_description"] = "MaxSAT solver returned something in stderr."
    elif solver.get("stdout_error", "") != "":
        solver["error_code"] = 703 + BiggerUINT32
        solver["fault_description"] = (
            "MaxSAT solver had ERROR written in some form in stdout"
        )

    if solver["checked_result"] == "TIMEOUT -- ignored" and not solver["mempeak_error"]:
        solver["error_code"] = 0
        solver["fault_description"] = ""

    if solver["error_code"] != 0 and solverName in args.solvers:
        overall_return_code += solver["id"] + solver.get("error_code", 0)
        if args.logging or args.logAll:
            logging_files.append(
                args.logPath
                + "/"
                + solverName
                + "_"
                + str(solver["error_code"])
                + "."
                + str(long_seed)
                + ".lg"
            )
    elif solver["error_code"] != 0:
        if args.logAll:
            logging_files.append(
                args.logPath
                + "/"
                + solverName
                + "_"
                + str(solver["error_code"])
                + "."
                + str(long_seed)
                + ".lg"
            )
    if solver["error_code"] != 0:
        faulty_solvers.append(solverName)

while overall_return_code > 255:
    overall_return_code = round(overall_return_code / 2 + 0.5)


if len(logging_files) > 0:
    print("c Create the following logfile(s): " + str(logging_files))

files = [open(file, "w") for file in logging_files]


def Logging(string, fd_list=files, to_console=True, prefix=""):
    for fd in fd_list:
        for line in string.split("\n"):
            fd.write(prefix + line + "\n")
    if to_console:
        print(prefix + string)


Logging("c WCNF-FILE....: " + args.wcnfFile)

destFile = ""
if len(faulty_solvers) > 0 and args.saveWCNF:
    if not os.path.exists(args.saveWCNFFolder):
        os.makedirs(args.saveWCNFFolder)
    if destFile == "":
        destFile = (
            args.saveWCNFFolder
            + "/"
            + os.path.basename(args.wcnfFile)[:-5]
            + "-"
            + str(random.randrange(1, 2**32 - 1))
            + ".wcnf"
        )
        # shutil.copy(args.wcnfFile, destFile)  ## saving without compression
        # Use subprocess to execute the xz command for compression
        # subprocess.run(["xz", "-z", "-k", args.wcnfFile, "-o", destFile], check=False)
        # wcnfTool.WriteToFile(destFile + ".xz", "new", True)
        wcnfTool.WriteToFile(destFile, "new", False)

    Logging("c copy saved as: " + destFile)

Logging("c SEED.........: " + seed)
Logging(instance_stats)

all_solvers_timout_ignored = True
for solverName, solver in solvers.items():
    if solver["checked_result"] != "TIMEOUT -- ignored" and solver["error_code"] == 0:
        all_solvers_timout_ignored = False
        break
if all_solvers_timout_ignored:
    Logging("c ISSUE: No valid results due to timeout or all solver had issues!!!")

if minVerifiedOValue == 2**64:
    minVeri = "-"
else:
    minVeri = str(minVerifiedOValue)

if BiggerUINT32 == 0:
    Logging("c SumOfWeightsi: < UINT32")
else:
    Logging("c SumOfWeightsi: > UINT32")
if anytime_solver:
    Logging("c Anytime TO...: " + str(anytime_timeout))
Logging("c Hard clauses.: " + str(solutionHardClauses))
# Logging("c Verified opt.: " + str(minVeri))
best_model = ""
certified = False

for solverName, solver in solvers.items():
    if solver["error_code"] == 0 and solver["checked_result"] != "TIMEOUT -- ignored":
        if solver["type"] == "certified":
            certified = True
            if minVeri != "-" and solver.get("model", "") != "":
                best_model = solver["model"]
            break
        if minVeri != "-" and solver.get("model", "") != "":
            best_model = solver["model"]
        # print(f"{solverName}, {best_model}")


if minVeri != "-":
    Logging(f"c Best o value.: {minVeri}")
    if args.printModel:
        Logging(f"c Best model...: {best_model}")
if certified:
    Logging("c Certified....: YES")

elapsed_time = round(time.time() - start_time, 4)
Logging(f"c Total time...: {elapsed_time}")

# Logging("c Cert o value.: " + str(minVeri))
# Logging("c solver times.: " + str(round(totalTime, 2)) + "  (without timeouts)")
Logging(
    "c "
    + "solver".ljust(22)
    + "error".rjust(10)
    + "exit".rjust(10)  # + "checked_result".rjust(22) +
    +
    # + "min_ver_o-value".rjust(22) + "solver_error".rjust(22))
    " ".rjust(22)
    + "o-value (calculated)".rjust(22)
    + "time".rjust(14)
    + "mempeak".rjust(14)
)
Logging(
    "c "
    + "name".ljust(22)
    + "code".rjust(10)
    + "status".rjust(10)  # + "checked_result".rjust(22) +
    +
    # + "min_ver_o-value".rjust(22) + "solver_error".rjust(22))
    "o-value".rjust(22)
    + "of model".rjust(22)
    + "seconds".rjust(14)
    + "KB".rjust(14)
    + "short".rjust(10)
)


for solverName, solver in solvers.items():
    if solver.get("error_code", 0) != 0:
        current_error = "!" + str(solver.get("error_code", 0))
    else:
        current_error = "0"

    tmp_time = (
        str(round(solver.get("time", -1), 4)) if solver.get("time", -1) != -1 else "TO"
    )

    solver["short"] = solver.get("short", "")

    Logging(
        "c "
        + solverName.ljust(22)
        + current_error.rjust(10)
        + str(solver.get("return_code", "-")).rjust(10)
        + str(solver.get("o-value", "-")).rjust(22)
        + str(solver.get("ver_o-value", "-")).rjust(22)
        + tmp_time.rjust(14)
        + str(solver.get("mempeak", "-")).rjust(14)
        + str(solver["short"].rjust(10))
    )

if faulty_solvers:
    Logging("c faulty solvers: " + str(faulty_solvers))

for solver in faulty_solvers:
    if solvers[solver].get("fault_description", "") != "":
        Logging(
            "c "
            + solvers[solver].get("short", solver)
            + " with ERROR CODE  "
            + str(solvers[solver].get("error_code", 0))
            + "  and FAULT DESCRIPTION:  "
            + solvers[solver].get("fault_description", "")
        )

if args.logging or args.logAll:
    for solver, file in zip(faulty_solvers, files):
        if solvers[solver].get("stdout", "") != "":
            Logging(
                str(solvers[solver].get("stdout", "")), [file], False, "[STDOUT" + "] "
            )
        if solvers[solver].get("stderr", "") != "":
            Logging(
                str(solvers[solver].get("stderr", "")), [file], False, "[STDERR" + "] "
            )

Logging("c rc:" + str(overall_return_code))

[file.close() for file in files]
exit(overall_return_code)
