#!/bin/bash
FILES=$(find ../../data/16_0/ -name *.png)

echo $FILES
for f in $FILES; 
do 
	echo "File: "$f
	cp $f ./ 
done
