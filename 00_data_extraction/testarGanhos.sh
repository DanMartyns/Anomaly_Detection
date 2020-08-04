#!bin/sh
counter=0
while [ $((counter+=5)) -le 50 ]
    do
        echo "Teste nr "$counter
        sudo bash execute.sh 1 mixed $1 $counter 300
    done
