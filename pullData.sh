bold=$(tput bold)
normal=$(tput sgr0)
echo "${bold}*** Get Script ***${normal}"

export SSHPASS='labcom'

echo -e "\n${bold}->${normal} A recolher os dados da máquina ${bold}salN34${normal}"
sshpass -e sftp -o StrictHostKeyChecking=no labcom@10.0.190.5 << EOF
	get -r Documents/Project/00_data_extraction/data 00_data_extraction/
	bye
EOF
