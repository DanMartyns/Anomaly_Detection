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

counter=0
while [ $((counter+=1)) -le $1 ]
    do
        echo **Extração nr $counter**
        now=$(date +"%d-%m-%y#%H-%M")
        if [ ! -d $(pwd)/data/$dir/$now ]; then
          mkdir -p $(pwd)/data/$dir/$now;
        fi
        if [ $2 == 'vazio' ]; then
          echo "-------------------> Servidor Bluetooth ligado"
          cd ../DataSimulation
          sudo sh executeServer.sh $5 $dir/$now
        fi
        cd ../00_data_extraction
        timeout $5 python3 hackrf2data.py -w $2 -l $3 -g $4
        rename "s/ //" *.bin
        mv *.bin $(pwd)/data/$dir/$now
        sleep 10
    done

sudo chown -R danielmartins data/$dir/

