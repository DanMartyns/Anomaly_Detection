#!bin/sh
counter=0
while [ $counter -le 50 ]
    do
        echo "Teste nr "$counter
        sh execute.sh 1 teste 8 $counter 120
        $((counter+=5))
    done