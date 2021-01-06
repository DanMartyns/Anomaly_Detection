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
        elif [[ -f "$f" && $f == *".bin"* ]]
        then
            #echo "É um ficheiro " $f
            path=$(echo $f | cut -d'/' -f 3-)

            size=$(echo $path | cut -d'/' -f 1- | tr "/" "\n" | wc -l)       
            dir=$(echo $path | cut -d'/' -f 1-$((size-=1)))
            
            file=$(echo $path | cut -d'/' -f $size- )
            filename=$(echo $file | cut -d'.' -f 1)
            
            file_count=$(find $(pwd)/$dir -name "*$filename.dat" | wc -l)
            
            if [ $file_count -gt 0 ]; then
                echo "Warning: $filename.dat found $file_count times in $dir!"
            else
                echo "The directory $dir not contain $filename.dat"
                if [[ $filename == *"anomaly"* ]];
                then
                    echo "Input File: "$f
                #     txt=$(find ../00_data_extraction/$dir -iname '*.txt')
                #     echo "Auxiliar File: "$txt
                #     python3 hackrfreadbinfile.py -f $f -fi $txt
                else
                    echo "Input File: "$f
                #     python3 hackrfreadbinfile.py -f $f 
                fi
                echo "Group Samples"
                python3 agroup.py -f $(pwd)/data/$filename.dat -s 30
                mv $(pwd)/data/32_15/$filename.dat $(pwd)/$dir
            fi
        fi
    done
}
DIRECTORY="../00_data_extraction/data"
recursiveDir $DIRECTORY
