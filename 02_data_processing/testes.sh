#!/bin/sh

recursiveDir(){
    for i in 1 2 5 10
    do
        for f in $1/*
        do 
            if [ -d "$f" ]
            then 
                echo $f
                recursiveDir $f
            elif [ -f "$f" ]
            then
                echo $f
                python3 Processing.py -f "${f}" -t -47 -wo 50 -ssw $i >> out2.txt
            fi
        done
    done
}
start=$SECONDS

DIRECTORY="../01_data_conversion/data/16_0/vazio/"
recursiveDir $DIRECTORY

duration=$(( SECONDS - start ))
echo "Duration : $duration" 


