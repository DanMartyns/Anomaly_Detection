#!/bin/sh

recursiveDir(){
    for f in $1/*
    do 
        if [ -d "$f" ]
        then 
            #echo "Directório " $f
            size=$(echo $f | cut -d'/' -f 1- | tr "/" "\n" | wc -l)
            file=$(echo $f | cut -d'/' -f 3-$size)
            if [ ! -d $(pwd)/$file ]; then
                mkdir -p $(pwd)/$file;
            fi
            recursiveDir $f
        elif [ -f "$f" ]
        then
            #echo "É um ficheiro " $f
            path=$(echo $f | cut -d'/' -f 3-)
            
            size=$(echo $path | cut -d'/' -f 1- | tr "/" "\n" | wc -l)       
            dir=$(echo $path | cut -d'/' -f 1-$((size-=1)))
            file=$(echo $path | cut -d'/' -f $size- )
            file_count=$(find $(pwd)/$dir -name "*$file" | wc -l)
            
            if [ $file_count -gt 0 ]; then
                echo "Warning: $file found $file_count times in $dir!"
            else
                echo "The directory $dir not contain $file"
                python3 Processing.py -f "${f}" -ws 1000 -wo 200
                mv $(pwd)/data/$file $(pwd)/$dir
            fi
            echo
        fi
    done
}
start=$SECONDS

DIRECTORY="../01_data_conversion/data"
recursiveDir $DIRECTORY

duration=$(( SECONDS - start ))
echo "Duration : $duration" 


