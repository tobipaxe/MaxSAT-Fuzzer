#!/usr/bin/env python3

import lzma
import sys
import os.path
import argparse
import contextlib
import subprocess
import random

vars = 0
nbHard = 0
nbSoft = 0
maxWeight = 0
sumOfWeights = 0
wcnfInputFormat = ""
clauses = []
model = []
optimum = -99999
solution = ""
bestSolution = ""
bestOptimum = -99999
solutionHardClauses = ""
error_dict = {}
issue_dict = {}

def reset_values():
    global vars, nbHard, nbSoft, maxWeight, sumOfWeights, wcnfInputFormat, clauses, model, optimum, solution, bestSolution, bestOptimum, solutionHardClauses, error_dict, issue_dict
    vars = 0
    nbHard = 0
    nbSoft = 0
    maxWeight = 0
    sumOfWeights = 0
    wcnfInputFormat = ""
    clauses = []
    model = ""
    optimum = -99999
    solution = ""
    bestSolution = ""
    bestOptimum = -99999
    solutionHardClauses = ""
    error_dict = {}
    issue_dict = {}

def is_valid_file(arg):
    if not os.path.exists(arg):
        print("The file %s does not exist!" % arg)
        exit(1)

def parse_wcnf(filename):
    global sumOfWeights, clauses, wcnfInputFormat, nbHard, nbSoft, vars, maxWeight
    hardClauseIndicator = "h"
    top = 0

    # Determine if the file is compressed based on the extension
    is_compressed = filename.endswith(".xz")

    # Open the file accordingly
    open_func = lzma.open if is_compressed else open

    with open_func(filename, "rt") as file:
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
                    print(
                        f"ERROR: Expected exactly three values to unpack but got: {len(temp_list)}",
                        file=sys.stderr,
                    )
                    exit(0)
                hardClauseIndicator = str(top)
                wcnfInputFormat = "old"
                continue

            if line.startswith(hardClauseIndicator):
                # Parse hard clause
                clause = list(map(int, line[len(hardClauseIndicator) + 1: -2].split()))
                nbHard += 1
                weight = -1
            elif line[0].isdigit():
                # Parse soft clause (weighted)
                weight, *clause = map(int, line[:-2].split())
                sumOfWeights += weight
                if weight > maxWeight:
                    maxWeight = weight
                nbSoft += 1
            else:
                print("c WARNING: read in line (ignored): " + line)
                continue

            maxVar = max(abs(lit) for lit in clause)
            if maxVar > vars:
                vars = maxVar
            clauses.append((weight, clause))
    if hardClauseIndicator == "h":
        wcnfInputFormat = "new"

def ParseSolutionMSEFormat(solver_output):
    global model, solution, optimum, error_dict, issue_dict
    model = "v"
    solution = "s"
    optimum = -99999

    lines = solver_output.split('\n')
    for line in lines:
        line = line.strip()

        # Ignore comments
        if line.startswith("c") or line == "":
            continue

        # solution line
        if line.startswith("s "):
            sol = line[2:]
            if solution != "s":
                issue_dict["Multiple lines starting with s"] = 1
            if sol == "OPTIMUM FOUND":
                solution = "OPTIMUM FOUND"
            elif "optimum" in sol.lower():
                solution = "OPTIMUM FOUND"
                error_dict["{line} instead of s OPTIMUM FOUND."] = 1
            elif sol == "UNSATISFIABLE":
                solution = "UNSATISFIABLE"
            elif "unsat" in sol.lower():
                solution = "UNSATISFIABLE"
                error_dict["{line} instead of s UNSATISFIABLE."] = 1
            elif sol == "SATISFIABLE":
                solution = "SATISFIABLE"
            elif "satisfiable" in sol.lower():
                solution = "SATISFIABLE"
                error_dict["{line} instead of s SATISFIABLE."] = 1
            elif sol == "UNKNOWN":
                solution = "UNKNOWN"
            elif "unkn" in sol.lower():
                solution = "UNKNOWN"
                error_dict["{line} instead of s UNKNOWN."] = 1
            else:
                error_dict["'{line}' should be one of the following 's [OPTIMUM FOUND, UNSATISFIABLE, SATISFIABLE, UNKNOWN]'."] = 1
            continue

        if line.startswith("o"):
            try:
                optimum = int(line[2:])
            except ValueError:
                error_dict[f"Invalid value for o: {line[2:]}"] = 1
                optimum = -1
            continue

        if line.startswith("v"):
            model = line[2:]
            continue
        issue_dict[f"Line doesn't start with c, s, o, v: {line}"] = 1

    if solution == "s":
        error_dict["No line starting with s"] = 1
        solution = ""
    if solution == "OPTIMUM FOUND" or solution == "SATISFIABLE":
        if optimum == -99999:
            error_dict["No line starting with o"] = 1
        elif optimum == "":
            error_dict["No optimum value given"] = 1
        if model == "v":
            error_dict["No line starting with v"] = 1
            model = ""
        elif model == "":
            issue_dict["No model given"] = 1

def CheckModel(givenModel=""):
    global bestSolution, bestOptimum, error_dict, issue_dict
    global clauses
    if givenModel == "":
        givenModel = model

    if len(givenModel) < vars:
        error_dict["Model with less variables than WCNF"] = 1
        bestSolution = "MODEL ERROR"
        bestOptimum = ""
        return
    if len(givenModel) > 10*vars:
        issue_dict["Model with 10x more variables than WCNF"] = 1

    index = -1
    bestOptimum = 0
    bestSolution = "OPTIMUM FOUND"
    for weight, clause in clauses:
        index = index + 1
        sat = False
        for lit in clause:
            if int(givenModel[abs(lit) - 1]) == int(lit > 0):
                sat = True
                break
        if sat is False:
            if weight == -1:
                bestSolution = "UNSATISFIABLE"
                bestOptimum = ""
                break
            bestOptimum += weight

def DumpWCNF(format):
    global sumOfWeights, vars, clauses, nbHard
    if format == "old":
        print(
            "p wcnf "
            + str(vars)
            + " "
            + str(len(clauses))
            + " "
            + str(sumOfWeights + 1)
        )
        hardClauseIndicator = str(sumOfWeights + 1) + " "
    elif format == "new":
        hardClauseIndicator = "h "
    elif format == "cnf":
        print("p cnf " + str(vars) + " " + str(nbHard))
        hardClauseIndicator = ""
    else:
        assert False, "ERROR: Wrong format: " + format

    for weight, clause in clauses:
        if weight == -1:
            print(hardClauseIndicator + " ".join(map(str, clause)) + " 0")
        elif format != "cnf":
            print(str(weight) + " " + " ".join(map(str, clause)) + " 0")


def DumpAtMPSCols(values):
    positions = [0, 1, 4, 14, 24, 39, 49]  # Define the starting positions

    removeList = []
    for i, value in enumerate(values):
        if not value:
            removeList = [i] + removeList
            continue

    for ele in removeList:
        positions.pop(ele)
        values.pop(ele)

    if positions[0] != 0:
        print("", end=" " * positions[0])

    for i, value in enumerate(values):
        if i + 1 < len(values):
            maxSize = positions[i + 1] - positions[i]
        else:
            maxSize = 0
        if i + 1 < len(values) and len(value) > maxSize:
            value = value[:maxSize]
        print(str(value), end=" " * (maxSize - len(value)))
    print("")


def IsTautologyClause(clause, ignore):
    if not ignore:
        return False
    tmpVarVec = []
    for var in clause:
        tmpVarVec.append(var)
        if -var in tmpVarVec:
            return True
    return False


def DumpAsMPS(format, name="WCNF2MPS"):
    global sumOfWeights, clauses, vars, nbHard, nbSoft

    variableMatrix = [[] for i in range(2 * (vars + 1))]
    rhsVec = []
    hclauses = sclauses = 0
    ignoretad = format == "mps_itad"  # ignore tautologies and duplicates
    if ignoretad:
        vars += 1
        clauses.append((-1, [vars]))
    for weight, clause in clauses:
        if IsTautologyClause(clause, ignoretad):
            continue
        if weight == -1:
            hclauses += 1
            clauseString = "HC" + str(hclauses)
        else:
            sclauses += 1
            clauseString = "SC" + str(sclauses)
        rhs = 1
        alreadyProcessed = []
        for var in clause:
            if ignoretad:
                if var in alreadyProcessed:
                    continue
                alreadyProcessed.append(var)
            if var > 0:
                currString = clauseString
            else:
                currString = "-" + clauseString
                rhs -= 1
            variableMatrix[abs(var)].append(currString)
        rhsVec.append((clauseString, rhs))

    DumpAtMPSCols(["NAME", "", "", name])
    print("ROWS")
    DumpAtMPSCols(["", "N", "OBJ"])
    for id in range(1, sclauses + 1):
        DumpAtMPSCols(["", "G", "SC" + str(id)])
    for id in range(1, hclauses + 1):
        DumpAtMPSCols(["", "G", "HC" + str(id)])

    # Bounds = []
    print("COLUMNS")
    for var in range(1, vars + 1):
        for cls in variableMatrix[var]:
            if cls[0] == "-":
                DumpAtMPSCols(["", "", "X" + str(var), cls[1:], "-1"])
            else:
                DumpAtMPSCols(["", "", "X" + str(var), cls, "1"])

    id = 0
    for weight, clause in clauses:
        if IsTautologyClause(clause, ignoretad):
            continue
        if weight != -1:
            id += 1
            DumpAtMPSCols(["", "", "B" + str(id), "SC" + str(id), "1"])
            DumpAtMPSCols(["", "", "B" + str(id), "OBJ", str(weight)])

    print("RHS")
    for type, rhs in rhsVec:
        DumpAtMPSCols(["", "", "RHS1", type, str(rhs)])

    print("BOUNDS")
    # for var in range(1, vars + 1):

    for var in range(1, vars + 1):
        if variableMatrix[var]:
            DumpAtMPSCols(["", "BV", "BND", "X" + str(var), "1"])

    # for id, sc in enumerate(softClauses):
    # for bid in range(1, nbSoft + 1):
    for bid in range(1, id + 1):
        DumpAtMPSCols(["", "BV", "BND", "B" + str(bid), "1"])
    print("ENDATA")
    if ignoretad:
        vars -= 1
        clauses.pop()


def WriteToFile(filename, format, compress=True):
    open_func = lzma.open if compress else open
    if not filename.endswith(".xz"):
        filename = f"{filename}.xz" if compress else filename
    with open_func(filename, "wt") as file:
        with contextlib.redirect_stdout(file):
            if format.startswith("mps"):
                DumpAsMPS(format)
            else:
                DumpWCNF(format)


def CheckIfHardClausesSAT(satSolver, seed=None):
    global nbHard, solutionHardClauses
    if nbHard == 0:
        solutionHardClauses = "SATISFIABLE"
        return
    if seed is None:
        seed = random.getrandbits(64)
    filename = "/tmp/" + str(seed) + ".cnf"
    WriteToFile(filename, "cnf", False)
    solverOut = subprocess.run(
        satSolver + " --time=60 " + filename, shell=True, capture_output=True
    )
    os.remove(filename)
    if solverOut.returncode == 10:
        solutionHardClauses = "SATISFIABLE"
    elif solverOut.returncode == 20:
        solutionHardClauses = "UNSATISFIABLE"
    else:
        solutionHardClauses = "UNKNOWN"

def CheckSolutionFile(filename=None):
    if filename:
        with open(filename, 'r') as file:
            solver_output = file.read()
    else:
        solver_output = sys.stdin.read()
    print("c Parse solver output...")
    ParseSolutionMSEFormat(solver_output)
    print("c Checking model...")
    CheckModel()


def main():
    parser = argparse.ArgumentParser(
        description="This is an awesome MaxSAT analysis and conversion script!!"
    )
    parser.add_argument(
        "wcnfFile",
        help="WCNF file which should be converted, in relation to current path.",
    )
    parser.add_argument(
        "-s",
        "--solutionFile",
        help="Solution file to check against the given WCNF file. If not provided, solution can be piped via stdin.",
    )
    parser.add_argument(
        "-n",
        "--newFilePrefix",
        default="",
        help="New file name prefix, in relation to current path. The suffix will be .wcnf or .mps.",
    )
    parser.add_argument(
        "-c",
        "--conversionTo",
        choices=["old", "new", "mps", "mps_itad"],
        default="new",
        help="Convert input file into one of the following formats: old (MSE format), new (MSE format), MPS (MIP solver)",
    )
    parser.add_argument(
        "-x",
        "--noCompression",
        action="store_false",
        help="Do not xz compress converted File.",
    )

    args = parser.parse_args()

    is_valid_file(str(args.wcnfFile))

    if not args.newFilePrefix:
        args.newFilePrefix = args.wcnfFile[:-5]

    parse_wcnf(args.wcnfFile)
    
    if args.solutionFile or not sys.stdin.isatty():
        print("c Checking output validity...")
        CheckSolutionFile(args.solutionFile)
        print(f"Solution Check Result: {bestSolution}, Optimum: {bestOptimum}")
        if error_dict:
            print("Errors:")
            for key, value in error_dict.items():
                print(f"  {key}")
            exit(1)
        if issue_dict:
            print("Issues:")
            for key, value in issue_dict.items():
                print(f"  {key}")
            exit(2)
        if bestSolution != solution:
            print(f"Expected s line: {solution}, but checker gives: {bestSolution}")
            exit(3)
        if optimum != bestOptimum:
            print(f"Given o value: {optimum}, but model gives: {bestOptimum}")
            exit(4)
    else:
        if "old" in args.conversionTo:
            print("c Convert to old MSE format:")
            WriteToFile(args.newFilePrefix + ".old.wcnf", "old", args.noCompression)
        elif "new" in args.conversionTo:
            print("c Convert to new MSE format:")
            WriteToFile(args.newFilePrefix + ".new.wcnf", "new", args.noCompression)
        elif "mps_itad" in args.conversionTo:
            print("c Convert to MPS format, ignore tautologies and duplicates:")
            WriteToFile(args.newFilePrefix + ".wcnf.mps", "mps_itad", args.noCompression)
        elif "mps" in args.conversionTo:
            print("c Convert to MPS format:")
            WriteToFile(args.newFilePrefix + ".wcnf.mps", "mps", args.noCompression)

if __name__ == "__main__":
    main()
