#!/usr/bin/bash
directory=$(dirname $0)

# This fuzzer is integrated in the current project. A big thank you to Florian Pollitt creating this fuzzer as a student work in the Debugging and Fuzzing lecture at the University of Freiburg.

#seed EXACTLY 20 digits
#--max-weight -- could be interesting

if [ $# -eq 2 ]; then
  seed=$1
  wcnfLocation=$2
elif [ $# -eq 1 ]; then
  seed=$1
  wcnfLocation=$(pwd)
elif [ $# -eq 0 ]; then
  seed=$RANDOM$RANDOM$RANDOM$RANDOM$RANDOM$RANDOM
  wcnfLocation=$(pwd)
else
  echo "Invalid seed or location given."
  exit 1
fi

seed="${seed:0:20}"
#echo "../FlorianPollitt/fuzzer.py --path stdout --seed $seed #> $wcnfLocation/bugPollitt-$seed.new.wcnf"
echo "c seed $seed"
python $directory/FlorianPollitt/fuzzer.py --path stdout --seed "$seed" # > "$wcnfLocation/bugPollitt-$seed.new.wcnf"
