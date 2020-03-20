#!/bin/sh

counter=1
while [ $counter -le 3 ]
    do
        echo **Extração nr $counter**
        timeout 3600 python3 hackrf2data.py
        sleep 3
        $((counter+=1))
    done

if [ ! -d $(pwd)/data ]; then
  mkdir -p $(pwd)/data;
fi

mv *.bin $(pwd)/data  
