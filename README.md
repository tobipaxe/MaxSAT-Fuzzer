# MaxSAT Fuzzer
The MaxSAT Fuzzer is a powerful tool designed for uncovering and classifying bugs in MaxSAT solvers through fuzzing and delta debugging. This fuzzer was created by Tobias Paxian, with an accompanying POS paper published in 2023, and a journal article planned. For more information, please cite:

Paxian, T., & Biere, A. (2023). Uncovering and Classifying Bugs in MaxSAT Solvers through Fuzzing and Delta Debugging. In Proceedings of the Pragmatics of SAT (POS) 2023.

## Installation

### Python Requirements
Ensure you have Python 3 installed. You will also need `psutil`. Install it with:
```sh
pip install psutil
```

### Submodule Kissat (or Other SAT Solver)
Initialize and update the submodule, then install Kissat:
```sh
apt-get install submodule
git submodule init
git submodule update
cd kissat
./configure && make test
```

### Configure `config.py` File with Your Solvers / Fuzzers
In `config.py`, configure your solvers and fuzzers:
- **Solvers**: Add at least two solvers to the `solvers` dictionary for comparison.
- **Fuzzers**: Add at least one fuzzer to the `fuzzers` dictionary (standard fuzzers are included).

### Build Delta Debugger
Build the delta debugger:
```sh
cd DeltaDebugger/wcnfddmin
make
```

### Build Fuzzer (if needed)
Build the fuzzer:
```sh
cd Fuzzer/wcnfuzz
./configure
make
```
```sh
cd Fuzzer/FlorianPollitt
./configure
make
```
Additional fuzzers/instance generators can be found and used with the provided scripts:
- [GaussMaxHS](https://github.com/meelgroup/gaussmaxhs): Generally produces large instances.
- [MaxSAT Fuzzer](https://github.com/conp-solutions/maxsat-fuzzer): Produces partly incorrect instances, but these are identified and ignored by the fuzzing script.

## Usage

### Getting Help
All tools can be started with `-h` or `--help` for more information about the parameters.

### Configuring `config.py`
Configure your solvers and fuzzers in this file. You need at least two MaxSAT solvers for `compare.py` to compare. More details are available in the file.

### `compare.py`
This script compares the results of at least two MaxSAT solvers:
```sh
./compare.py problem.wcnf
```
It checks for issues and handles both old and new WCNF input formats, but only the new output format. Ensure `compare.py` is properly configured and test it with any WCNF file:
```sh
./compare.py anyWCNF.wcnf
```

### `wcnftool.py`
A utility to read, convert, and check WCNF files. It can be used standalone or as an import in `compare.py`.

### `runwcnfuzz.py`
Run this tool with:
```sh
./runwcnfuzz.py -t [number_of_threads]
```
End the program with `ctrl+c` to close subprocesses and clean up folders. Ensure `compare.py` works properly as it is called within this script. The delta debugger can be deactivated to save time. More information is available with `--help`.

### Fuzzer
Each fuzzer should produce a single instance and print it to the console. Configure fuzzers in `config.py`.

### Delta Debugger
The delta debugger wcnfddmin reduces instance size when a bug is found. It can be used directly with any MaxSAT solver. More information with `./wcnfddmin -h`

## Log Files
All log files are saved in `Logs/date-secondsOfTheDay-runwcnfuzz/xxx`. The program will indicate the log file location at the end of usage.
