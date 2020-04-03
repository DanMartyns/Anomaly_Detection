bold=$(tput bold)
normal=$(tput sgr0)
echo "${bold}*** Get Script ***${normal}"

export SSHPASS='labcom'

echo -e "\n${bold}->${normal} A recolher os dados da máquina ${bold}salN34${normal}"
sshpass -e sftp -o StrictHostKeyChecking=no labcom@10.0.190.5 << EOF
	get -r Documents/Project/00_data_extraction/data 00_data_extraction/
	get -r Documents/Project/01_data_conversion/data 01_data_conversion/
	get -r Documents/Project/02_data_processing/data 02_data_processing/
	bye
EOF