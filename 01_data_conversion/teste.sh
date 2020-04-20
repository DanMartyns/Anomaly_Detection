#!/bin/sh

recursiveDir(){
    for f in $1/*
    do 
        if [ -d "$f" ]
        then 
            echo "Directório " $f
            recursiveDir $f
        elif [[ -f "$f" ]]; 
        then
            echo "É um ficheiro " $f
            python3 hackrfreadbinfileWithGraphic.py -f $f
        fi
    done
}
recursiveDir $1
 