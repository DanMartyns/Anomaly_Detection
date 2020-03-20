# $1 indica o número de horas a extrair
# $2 indica a label que cada ficheiro terá

cd 00_data_extraction 
sh execute.sh $1 $2
cd ..
cd 01_data_conversion
sh execute.sh
cd ..
cd 02_data_classification
sh execute.sh