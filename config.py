#!/usr/bin/env python3

timeoutFactor = 50
mempeakFactor = 50
new_rules = False
# mem limit in MB
mem_limit_for_one_solver_call = 32000

# SOLVER CONFIGURATION
# PLEASE ADD OWN SOLVERS TO THE DICTIONARY IN THE FOLLOWING WAY
# It is possible to run the script with only one solver, but for being able to compare results of solvers
# you should add at least two solvers. Best if these solvers are not only variations of one solver, but
# completely differnt. You can add as many solvers as you want, but then the execution time will be high for one run.
# Until now I only found two solvers without bugs -
#   EvalMaxSAT from the 2022 MSE hompage
#   sat4j from the github repository -- it needs to be started from a shell script
#                                       reformatting the model output to the new format
# This solvers should be able to accept produce the results in the "new" MSE format. If not see
# in the folder scripts sat4j.sh to get a hint how you can transform the results to that format.
# YOURSOLVERNAME should be shorter than 22 symbols and only contains - letters, numbers and -;
# DONT USE THE UNDERSCORE SYMBOL _
# solver["YOURSOLVERNAME"] = {
#     "solver_call" : "SOLVERCALL is the binary of the current script with the whole path toghether with arguments ATTENTION: WITHOUT UNDERSCORE",
#     "input_format": "wcnf" | "oldwcnf" | "mps"
#     "type": "complete" | "anytime" | "certified"
#     "short": "up to 6 characters to identify the solver. Has to be unique and ATTENTION: WITHOUT UNDERSCORE!"
#     "upper_bound": 1000000 This is a non mandatory parameter with an upper bound for the sum of weights.

# Attention MIP solver have all their own output format. You have to transform that to the new MSE format first
solvers = {
    ## MSE 22 solver
    "MSE22-EvalMaxSAT": {
        "solver_call": "/usr/local/scratch/paxiant/MaxSATFuzzer/MaxSATSolver/MSE22/EvalMaxSAT/bin/EvalMaxSAT_bin",
        "input_format": "wcnf",
        "type": "complete",
        "short": "EMS",
    },
    # "Pac-PC-DPW-Release": {
    #     "solver_call": "/usr/local/scratch/paxiant/MaxSATFuzzer/MaxSATSolver/CertifiedSolver/certified-pacose.sh",
    #     "input_format": "wcnf",
    #     "type": "certified",
    #     "short": "PPCDPW",
    # },
    # "MaxHSCadical": {
    #     "solver_call": "/usr/local/scratch/paxiant/gitlab/MaxHSCadical/MaxHS/code/build/release/bin/maxhs -printSoln",
    #     "input_format": "wcnf",
    #     "type": "complete",
    #     "short": "MHC",
    # },

    "MaxHSSCIP": {
        "solver_call": "/usr/local/scratch/paxiant/MaxSATFuzzer/MaxSATSolver/MaxHSBinaries/maxhsScip -printSoln",
        "input_format": "wcnf",
        "type": "complete",
        "short": "MHSS",
    },
    # "MaxHSSCIPExact": {
    #     "solver_call": "/usr/local/scratch/paxiant/MaxSATFuzzer/MaxSATSolver/MaxHSBinaries/maxhsScipExact -printSoln",
    #     "input_format": "wcnf",
    #     "type": "complete",
    #     "short": "MHSSE",
    # },
    # "MaxHSCPLEX": {
    #     "solver_call": "/usr/local/scratch/paxiant/MaxSATFuzzer/MaxSATSolver/MaxHSBinaries/maxhsCplexNew -printSoln",
    #     "input_format": "wcnf",
    #     "type": "complete",
    #     "short": "MHSC",
    # },
    # "MaxHSCPLEXOld": {
    #     "solver_call": "/usr/local/scratch/paxiant/MaxSATFuzzer/MaxSATSolver/MaxHSBinaries/maxhsCplexOld -printSoln",
    #     "input_format": "wcnf",
    #     "type": "complete",
    #     "short": "MHSCO",
    # },


    # "sat4j": {
    #     "solver_call": "/usr/local/scratch/paxiant/MaxSATFuzzer/Others/sat4j.sh",
    #     "input_format": "oldwcnf",
    #     "type": "complete",
    #     "short": "S4J",
    # }

    # MIP Solver
    # "SCIP": {
    #     "solver_call": "/usr/local/scratch/paxiant/MaxSATFuzzer/MIPSolver/SCIP.sh",
    #     "input_format": "mps",
    #     "upper_bound": 1000000,
    #     "type": "complete"
    # },

    # MSE 23 Anytime Solver
    # "NuWLS-c-static": {
    #     "solver_call": "/usr/local/scratch/paxiant/MaxSATFuzzer/MSE23/anytime/NuWLS-c-2023/bin/NuWLS-c_static",
    #     "input_format": "wcnf",
    #     "type": "anytime",
    #     "short": "NuWcs",
    # },

    # MSE 23 Complete Solver
    # "EvalMaxSAT-SCIP": {
    #     "solver_call": "/usr/local/scratch/paxiant/MaxSATFuzzer/MSE23/exact/EvalMaxSAT/bin/EvalMaxSAT --timeUB 0 --minRefTime 5",
    #     "input_format": "wcnf",
    #     "type": "complete",
    #     "short": "EMSSC",
    # },
}

wcnf_compare_script = "./compare.py --logging --saveWCNF"
delta_debugger_compare_script = "./compare.py_--saveWCNF"
delta_debugger = "DeltaDebugger/wcnfddmin/wcnfddmin "

# dictionary with the following entries
#   name of fuzzer      this contains an additional dictionary with the following entries
#       command         how to start the fuzzer in the command line
#       upper-bound     upper-bound argument if possible, then the upper bound will be added after this argument
#       compare_extra   this argument will be added only for this fuzzer to the compare script
#                       for example it has been shown that in very rare cases the manthey fuzzer produces
#                       some strange numbers higher than top. Therefore it is good to rewrite that instance
#                       again. (arg=="compare_extra": " --reWriteAllWCNFs")
#       seed            if necessary -- after this argument the 20 digit seed follows. If not given the seed
#                       is added after an empty space.
#       max_seed         upper bound for a seed of this fuzzer
fuzzers = {
    "Paxian": {"command": "Fuzzer/wcnfuzz/wcnfuzz --wcnf", "upper_bound": "-u"},
    # "PaxianPy": {
    #     "command": "Fuzzer/wcnfuzz.py",
    #     "upper_bound": "--upperBound",
    #     "seed": "--seed",
    # },
    # "PaxianPySmall": {
    #     "command": "Fuzzer/wcnfuzz.py --small",
    #     "upper_bound": "--upperBound",
    #     "seed": "--seed",
    # },
    "PaxianPyTiny": {
        "command": "Fuzzer/wcnfuzz.py --tiny",
        "upper_bound": "--upperBound",
        "seed": "--seed",
    },
    "PaxianPyGBMO": {
        "command": "Fuzzer/wcnfuzz.py --gbmo",
        "upper_bound": "--upperBound",
        "seed": "--seed",
    },
    # # "PaxianPySmallGBMO": {
    # #     "command": "Fuzzer/wcnfuzz.py --small --gbmo",
    # #     "upper_bound": "--upperBound",
    # #     "seed": "--seed",
    # # },
    # # "PaxianPyTinyGBMO": {
    # #     "command": "Fuzzer/wcnfuzz.py --tiny --gbmo",
    # #     "upper_bound": "--upperBound",
    # #     "seed": "--seed",
    # # },
    "Pollitt": {
        "command": "Fuzzer/generateFlorianPollitWCNF.sh",
        "max_seed": 10**20,
        "min_seed": 10**19,
    },
    # # The following two fuzzers have to be downloaded and are not part of the project!
    # # But I've included the necessary scripts to reproduce the results anyhow!
    "Manthey": {
        "command": "Fuzzer/generateNorbertMantheyWCNF.sh",
        "compare_extra": "--reWriteAllWCNFs",
    },
    "Soos": {
        "command": "Fuzzer/generateMateSoosWCNF.sh"
    },
}

