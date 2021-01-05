#!/bin/bash

for n in {3..15}
do
    echo
    echo "Testing for $n components"
    python3 train.py -d ../02_data_processing/data/32_15/ -w normal -n $n
done
