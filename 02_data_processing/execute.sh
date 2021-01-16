#!/bin/bash -x

if (($# == 0)); then
    echo "Please pass arguments"
    exit 2
fi

recursiveDir(){
    declare -i ws=$2
    declare -i wo=$3
    declare -r thr=$4
    for f in $1/*
    do 
        if [ -d "$f" ]
        then 
            #echo "Directório " $f
            size=$(echo $f | cut -d'/' -f 1- | tr "/" "\n" | wc -l)
            file=$(echo $f | cut -d'/' -f 3-$size)
            if [ ! -d $(pwd)/02_data_processing/data/$file ]; then
                mkdir -p $(pwd)/02_data_processing/data/$file;
            fi
            recursiveDir $f $ws $wo $thr
        elif [ -f "$f" ]
        then
            #echo "É um ficheiro " $f
            path=$(echo $f | cut -d'/' -f 3-)
            
            size=$(echo $path | cut -d'/' -f 1- | tr "/" "\n" | wc -l)
            dir=$(echo $path | cut -d'/' -f 1-$((size-=1)))
            
            file=$(echo $path | cut -d'/' -f $size- )
            
            file_count=$(find $(pwd)/02_data_processing/data/$dir -name $file | wc -l)

            if [ $file_count -gt 0 ]; then
                echo "Warning: $file found $file_count times in $dir!"
            else
                echo "The directory $dir not contain $file"
                echo "Processing features" -t "${thr}"
                python3 02_data_processing/Processing_scenario_1.py -f "${f}" -ws "${ws}" -wo "${wo}" -t "${thr}"
                mv $(pwd)/02_data_processing/data/$file $(pwd)/02_data_processing/data/$dir
            fi
            echo
        fi
    done
}

start=$SECONDS

DIRECTORY="01_data_conversion/data"
recursiveDir $DIRECTORY $1 $2 $3

duration=$(( SECONDS - start ))
echo "Duration : $duration" 


