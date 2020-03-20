import os
import math
import struct
import argparse
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler

def readFileToMatrix(files) :
    assert len(files) > 0
    fb = open(files[0], "rb")
    
    matrix = []
    try:
        while True:
            # read each line of the file
            record=fb.read(32)

            # break when the line was empty
            if(len(record) < 32): break
            
            line = list(struct.unpack('=4d',record))

            matrix.append(line)

        matrix = np.array(matrix)

        if len(files) > 1:
            for f in files[1:]:
                f = open(f, "rb")
                while True:
                    # read each line of the file
                    record=fb.read(32)
                    # break when the line was empty
                    if(len(record) < 32): break

                    matrix = np.concatenate((matrix, list(struct.unpack('=4d',record))))
    finally:
        fb.close()

    return matrix

def predict(data, scaler, clf):
    scaled_data = scaler.transform(data)
    return clf.predict(scaled_data)

def main():

    parser = argparse.ArgumentParser()
    # Read from files
    parser.add_argument("-f", "--files", nargs='+')
    # Read from an directory
    parser.add_argument("-d", "--directory", nargs='+')
    # Wildcard to detect what is normal
    parser.add_argument("-w", "--wildcard", required=True)
    # Assure at least one type of this capture goes to training
    #parser.add_argument("-a", "--assure", nargs='+')
    # Wants to export files
    #parser.add_argument("-e","--export",  action='store_true')
    # Print confusion matrix for each algorithm
    #parser.add_argument("-v","--verbose",  action='store_true')
    args=parser.parse_args()

    if not (args.files or args.directory):
        parser.error("No files given, add --files or --directory.")

    if not args.files:
        args.files = []

    # get all filenames from directory
    if args.directory:
        for dir in args.directory:
            for r, d, f in os.walk(dir):
                for file in f:
                    if ".dat" in file:
                        args.files.append(os.path.join(r, file))

    print("Ficheiros a serem processados: ",args.files)
    train_files = []
    anomaly_test_files = []
    #regular_test_files = []

    # divide filenames in true pc or other
    for f in args.files:
        if args.wildcard in f:
            train_files.append(f)
        else:
            anomaly_test_files.append(f)
    
    ratio = 0.8

    # split data between train and test data
    data = readFileToMatrix(args.files)
    train_data = data[:math.floor(len(data)*ratio)]
    test_data = data[math.floor(len(data)*ratio):]

    #print("Dados de treino: ", train_data,'\n')
    #print("Dados de teste: ",test_data,'\n')

    scaller = StandardScaler()
    scaller.fit(train_data)
    train_data = scaller.transform(train_data)
    print(train_data)

    classifier = svm.OneClassSVM(gamma='auto', kernel='rbf')

    classifier.fit(train_data)
    an = predict(test_data, scaller, classifier).reshape(-1,1)
    
    print(an)
if __name__ == '__main__':
	main()