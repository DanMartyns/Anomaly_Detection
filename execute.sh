#!/bin/bash

declare -i ARG1="243"
declare -i ARG2="81"
declare -i ARG3="-66"

declare -i sec=81

echo 
echo "Testing Observation Window with $((ARG1/sec)) sec, Observation Offset $((ARG2/sec)) sec and Threshold $ARG3 "
echo

bash 02_data_processing/execute.sh "$ARG1" "$ARG2" "$ARG3"
for n in {3..18}
do
    echo
    echo "Testing for $n components"
    python3 03_data_classification/train.py -d 02_data_processing/data/32_15/ -w normal -n $n
done
