#!/usr/bin/bash

# this fuzzer is not part of the current git repository but can be downloaded in https://github.com/meelgroup/gaussmaxhs
# SSH: git clone git@github.com:meelgroup/gaussmaxhs.git

set -e

#directory=$(dirname $0)
directory=/usr/local/scratch/paxiant/MaxSATFuzzer/MateSoos/gaussmaxhs/build/release/bin/
cd $directory

if [ $# -eq 2 ]; then
  seed=$1
  wcnfLocation=$2
elif [ $# -eq 1 ]; then
  seed=$1
  wcnfLocation=$(pwd)
elif [ $# -eq 0 ]; then
  seed=$(od -vAn -N8 -tu8 < /dev/urandom | tr -d '[:space:]')
  wcnfLocation=$(pwd)
else
  echo "Invalid seed or location given."
  exit 1
fi


echo "c seed $seed"
python cnf-fuzz-brummayer.py -s $seed | grep -v "^c " | python cnf_to_wcnf_and_xors.py | python xor_to_cnf.py | python strip_wcnf.py -x #| "$directory"/maxsat_benchmarks_code_base/bin/std_wcnf -preserve #> "$wcnfLocation/bugSoos-$seed.old.wcnf"

# ./cnf-fuzz-brummayer.py -s $1 | grep -v "^c " > "/tmp/in_cnf-$1"
# cat "/tmp/in_cnf-$1" | ./cnf_to_wcnf_and_xors.py > "/tmp/in_wcnf_xor-$1"
# cat "/tmp/in_wcnf_xor-$1" | ./xor_to_cnf.py > "/tmp/in_wcnf_xor_blasted-$1"
# # strip xor for orig maxhs
# cat "/tmp/in_wcnf_xor_blasted-$1" | ./strip_wcnf.py -x > "bug-$1.wcnf"
# rm -rf "/tmp/in_cnf-$1"
# rm -rf "/tmp/in_wcnf_xor-$1"
# rm -rf "/tmp/in_wcnf_xor_blasted-$1"
