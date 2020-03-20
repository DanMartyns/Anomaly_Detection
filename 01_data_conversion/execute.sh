#!/bin/sh

if [ ! -d $(pwd)/data ]; then
  mkdir -p $(pwd)/data;
fi

DIRECTORY=../00_data_extraction/data/*
DIRECTORY_OUT=data/
suffix=.dat

for f in $DIRECTORY; do

  file=$(echo $f | cut -d'.' -f 3)
  filename=$(echo $file | cut -d'/' -f 4)

  file_count=$(find $DIRECTORY_OUT -name "*$filename$suffix" | wc -l)

  if [ $file_count -gt 0 ]; then
      echo "Warning: $filename$suffix found $file_count times in $DIRECTORY_OUT!"
  else
      echo "The directory $DIRECTORY_OUT not contain $filename$suffix"
      python3 hackrfreadbinfile.py -f "${f}"
  fi

done
