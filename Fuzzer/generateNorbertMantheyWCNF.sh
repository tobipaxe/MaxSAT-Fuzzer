#!/usr/bin/bash

# this fuzzer is not part of the current git repository but can be downloaded in https://github.com/conp-solutions/maxsat-fuzzer
# SSH: git clone git@github.com:conp-solutions/maxsat-fuzzer.git
# it produces partly incorrect wcnfs in the old format -- but the original fuzzer can handle it by checking the validity of the wcnf

#seed max = 4294967295 after that it is an overflow and start at 0 again
directory=/usr/local/data/MaxSATFuzzer/NorbertManthey/maxsat-fuzzer/
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
./wcnffuzzer $seed 2>&1 | grep -v "^c" #| $directory/maxsat_benchmarks_code_base/bin/std_wcnf -preserve 2>&1 #> "$wcnfLocation/bugManthey-$seed.old.wcnf" 2>&1
