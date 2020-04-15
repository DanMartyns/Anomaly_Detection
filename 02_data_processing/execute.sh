#!/bin/sh

# create the data folder if it's not exists
if [ ! -d $(pwd)/data ]; then
  mkdir -p $(pwd)/data;
fi

DIRECTORY=../01_data_conversion/data/*
DIRECTORY_OUT=data/
suffix=.dat

start=$SECONDS

for f in $DIRECTORY; do

  # get only the name file
  file=$(echo $f | cut -d'.' -f 3)
  # get the name file without extension
  filename=$(echo $file | cut -d'/' -f 4)

  # count the number of files with a specific name
  file_count=$(find $DIRECTORY_OUT -name "*$filename$suffix" | wc -l)

  # check if the output directory has the already processed file 
  if [ $file_count -gt 0 ]; then
      echo "Warning: $filename$suffix found $file_count times in $DIRECTORY_OUT!"
  # if not, the file is processed
  else
      echo "The directory $DIRECTORY_OUT not contain $filename$suffix"
      python3 Processing.py -f "${f}" -ws 50 -wo 5 &
  fi

done

duration=$(( SECONDS - start ))

echo "Duration : $duration" 