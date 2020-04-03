#!/bin/sh

counter=1
while [ $counter -le 10 ]
    do
        echo **Simulação nr $counter**
        python3 simulateHuman.py
        $((counter+=1))
    done