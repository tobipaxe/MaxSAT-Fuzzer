#!/bin/bash

directory=$(dirname $0)
rand=$RANDOM$RANDOM$RANDOM
echo "c Random prefix: $rand"

output=$(eval java -jar "$directory"/sat4j/dist/CUSTOM/sat4j-maxsat.jar "$1")
ret=$?

echo "$output" | grep -v '^v '
values=$(echo "$output" | grep '^v ')
echo "c $values"
values=($values)
count=0
emptyVar=0

for value in "${values[@]}"
do
  if [ "$value" == "v" ]; then
    echo -n "v "
    continue
  fi
  ((count++))
  if [ "$value" = "0" ]; then
    continue
  fi

  while [[ $value -ne $count ]] && [[ $value -ne -$count ]]; do
    ((count++))
    ((emtyVar++))
    echo -n "0"
  done

  if [ $value = "-$count" ]; then
    if [ "$value" != "-$count" ]; then
      echo "error in outputting result $count $value"
      exit 1
    fi
    echo -n "0"
  else
    if [ "$value" != "$count" ]; then
      echo "error in outputting result $count $value"
      exit 1
    fi

    echo -n "1"
  fi
done

echo ""
exit "$ret"

