bold=$(tput bold)
normal=$(tput sgr0)
echo "${bold}*** Deployment Script ***${normal}"

export SSHPASS='labcom'

echo "\n${bold}->${normal} Remove versão antiga da máquina ${bold}salN34${normal}"
sshpass -e ssh -tt -o StrictHostKeyChecking=no labcom@10.0.190.5 << EOF
    cd Documents 
    rm -rf Project
    mkdir Project
    exit 
EOF

sleep 5

echo "\n${bold}->${normal} A mover ficheiros essenciais para a máquina ${bold}salN34${normal}"
sshpass -e sftp -o StrictHostKeyChecking=no labcom@10.0.190.5  << !
    put -r 00_data_extraction/ Documents/Project/00_data_extraction
    put -r 01_data_conversion/ Documents/Project/01_data_conversion
    put -r 02_data_processing/ Documents/Project/02_data_processing
    put -r executeAll.sh Documents/Project/executeAll.sh
    put simulateHuman/simulateHumanBluServer.py Documents/Project/server.py
    ls Documents/Project/
    bye
!

sleep 10

echo "\n${bold}->${normal} A executar na máquina ${bold}salN34${normal}"
sshpass -e ssh -tt -o StrictHostKeyChecking=no labcom@10.0.190.5 << EOF
    python3 Documents/Project/server.py &
    sh Documents/Project/executeAll.sh $1 $2 &
    exit
EOF
