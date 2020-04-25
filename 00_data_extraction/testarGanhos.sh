#!bin/sh
counter=0
while [ $counter -le 50 ]
    do
        echo "Teste nr "$counter
        sh execute.sh 1 teste $1 $counter 300
        $((counter+=5))
    done
