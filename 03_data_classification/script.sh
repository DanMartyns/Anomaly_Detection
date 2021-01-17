for i in {1,2,3,4,5}
do
	echo "\n Test "$i" components"
	python3 feature_reduction.py -d ../02_data_processing/data/32_15/ -w normal -c $i
done
