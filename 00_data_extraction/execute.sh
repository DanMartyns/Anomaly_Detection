#!/bin/sh
# $1 - numero de iterações
# $2 - wildcard
# $3 - RX LNA (IF) gain, 0-40dB, 8dB steps
# $4 - RX VGA (baseband) gain, 0-62dB, 2dB steps
# $5 - tempo de cada iteração

dir=$3_$4

if [ ! -d $(pwd)/data/$dir ]; then
  mkdir -p $(pwd)/data/$dir;
fi

counter=1
while [ $counter -le $1 ]
    do
        echo **Extração nr $counter**
        if [ $2 == "vazio"]; 
        then
          python3 simulateHumanBluClient.py &
        fi 
        timeout $5 python3 hackrf2data.py -w $2 -l $3 -g $4
        sleep 3
        $((counter+=1))
    done

rename "s/ //" *.bin

mv *.bin $(pwd)/data/$dir
