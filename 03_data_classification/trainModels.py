import os
import sys
import math
import heapq
import struct
import random
import argparse
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import f_classif
# The idea behind StandardScaler is that it will transform your data such 
# that its distribution will have a mean value 0 and standard deviation of 1.
# In case of multivariate data, this is done feature-wise (in other words 
# independently for each column of the data). Given the distribution of the data, 
# each value in the dataset will have the mean value subtracted, and then divided 
# by the standard deviation of the whole dataset (or feature in the multivariate case).
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
	
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def readFileToMatrix(files) :
    assert len(files) > 0
    
    matrix = []
    for f in files:
        fb = open(f, "rb")
        try:
            while True:
                # read each line of the file
                record=fb.read(152)
                
                # break when the line was empty
                if(len(record) < 152): break
                
                # unpack the record
                line = list(struct.unpack('=19d',record))

                # append the line to matrix
                matrix.append(line)
    
        finally:
            fb.close()

    matrix = np.array(matrix)
    return matrix

def combinations(gamma, kernel, nu):
    result = ( ["classifier.append(svm.OneClassSVM(gamma={}, nu={}, kernel='{}'))".format(g,n,k)]
            for g in gamma
            for k in kernel
            for n in nu)

    result = [item for sublist in list(result) for item in sublist]

    return dict( (index,['OC-SVM', value]) for index, value in enumerate(result)) 

def calc_score(true_dataset, predicted_dataset):
    tp, fn, fp, tn = confusion_matrix(true_dataset, predicted_dataset, labels=[1,-1]).reshape(-1)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return 2 * ((precision * recall)/(precision + recall))

def print_results(true_dataset, predicted_dataset):
    tp, fn, fp, tn = confusion_matrix(true_dataset, predicted_dataset, labels=[1,-1]).reshape(-1)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1Score = 2 * ((precision * recall)/(precision + recall))*100
    # print("----------------------------------------------------------")
    # print(color.BOLD+"Number of Anomalies : "+color.END,true_dataset[true_dataset == -1].size)
    # print(color.BOLD+"Number of Normals : "+color.END, true_dataset[true_dataset == 1].size) 
    # print(color.BOLD+"Precision:"+color.END, "{:.2f}%".format(round(precision,3)*100),"%")
    # print(color.BOLD+"Recall:"+color.END, "{:.2f}%".format(recall))
    # print(color.BOLD+"F1-Score: {:.2f}%".format(f1Score)+color.END)
    # print(color.BOLD+"Confusion matrix:"+color.END)
    # print(pd.DataFrame([[tp, fp], [fn, tn]], ["Pred Normal", "Pred Anomaly"], ["Actual Normal", "Actual Anomaly"]))
    # print("----------------------------------------------------------")
    
    # classification report for precision, recall f1-score and accuracy
    matrix = classification_report(true_dataset,predicted_dataset,labels=[1,-1])
    # print('Classification report : \n',matrix)

    return f1Score

def remove_algorithms(score):
    score.sort(reverse=True)
    median = np.median(score)
    step = float("{:.4f}".format((score[0] - score[len(score)-1])/len(score)))
    values = []

    for i in range(1, len(score)):
        if score[i] < median and (math.floor(score[i-1] - score[i] >= step) or median - score[i] > 2*step):
            values.append(score[i])

    return [i for i, x in enumerate(score) if x in values]

def decide(pred, ignore=[]):
    pred = np.delete(pred, ignore, axis=1)
    l = []
    for i in range(0, pred.shape[0]):
        col = pred[i,:]
        if col.tolist().count(-1) > math.ceil(pred.shape[1]*0.4):
            l.append(-1)
        else:
            l.append(1)
    return np.array(l)

def main():

    parser = argparse.ArgumentParser()
    # Read from files
    parser.add_argument("-f", "--files", nargs='+')
    # Read from an directory
    parser.add_argument("-d", "--directory", nargs='+')
    # Wildcard to detect what is normal
    parser.add_argument("-w", "--wildcard", required=True)
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

    anomaly_files = []
    regular_files = []

    # divide filenames in normal or anomaly
    for f in args.files:
        if args.wildcard in f:
            regular_files.append(f)
        else:
            anomaly_files.append(f)

    # Feature Selection
    features = [ 'mean_activity', 'std_activity', 'mean_silence', 'std_silence', \
    'percentil_75', 'percentil_90', 'percentil_95', 'percentil_99', 'median', \
    'max', 'mean_active_frequencies', 'diff_first_last_one', 'DMI', 'AroonUP', \
    'diff_max_min', 'meanConseqAct', 'meanConseqActiveFreq', 'EMA']

    dataset_clean = pd.DataFrame(data=readFileToMatrix(regular_files))
    dataset_dirty = pd.DataFrame(data=readFileToMatrix(anomaly_files))

    dataset_clean.columns = features + ['target']
    dataset_dirty.columns = features + ['target']

    Y_clean = dataset_clean['target']
    X_clean = dataset_clean.drop(['diff_max_min','meanConseqAct','max','EMA','DMI','target'], axis = 1)

    Y_dirty = dataset_dirty['target']
    X_dirty = dataset_dirty.drop(['diff_max_min','meanConseqAct','max','EMA','DMI','target'], axis = 1)

    result = pd.DataFrame(columns=['Classifier Group', 'K-Fold', 'F1-Score']) 

    for k in range(2,10):
        # prepare cross validation
        kfold = KFold(k, True, 1)
        f1Score = []

        for clean, dirty in zip(kfold.split(X_clean), kfold.split(X_dirty)):
            train_clean_index, clean_cross_validation_index = clean
            train_dirty_index, dirty_cross_validation_index = dirty

            X_train_clean, X_cross_validation_clean = X_clean.iloc[train_clean_index], X_clean.iloc[clean_cross_validation_index]
            y_train_clean, y_cross_validation_clean = Y_clean.iloc[train_clean_index], Y_clean.iloc[clean_cross_validation_index]

            X_train_dirty, X_cross_validation_dirty = X_dirty.iloc[train_dirty_index], X_dirty.iloc[dirty_cross_validation_index]
            y_train_dirty, y_cross_validation_dirty = Y_dirty.iloc[train_dirty_index], Y_dirty.iloc[dirty_cross_validation_index]

            cv_test = pd.concat([X_train_clean, X_cross_validation_clean, X_cross_validation_dirty])
            cv_label = pd.concat([y_train_clean, y_cross_validation_clean, y_cross_validation_dirty])

            ######################################################################
            ############### Standarize the values of the train data ##############
            ######################################################################

            scaler = StandardScaler()

            # Fit on training set only.
            scaler.fit(X_train_clean)
            
            # Apply transform to both the training set and the test set.
            X_train_clean = scaler.transform(X_train_clean)
            cv_test = scaler.transform(cv_test)

            score = []
            flag = True

            cv_label[cv_label == 1] = -1
            cv_label[cv_label == 0] = 1
            cv_label = np.array(cv_label).reshape(-1,1)

            classifier_group = []
            classifier = []

            #######################################################################
            ############ Test the features using the OC-SVM Model #################
            #######################################################################  
            gamma = [ 0.001, 0.01, 0.1]
            kernel = ['linear']
            nu = [ 0.001, 0.01, 0.1]

            comb = {}
            comb = combinations(gamma, kernel, nu)

            for index, cl in comb.items():
                exec(cl[1])

            classifier_group.append(classifier)

            #######################################################################
            ########## Test the features using the Isolation Forest ###############
            #######################################################################  

            classifier = []
            classifier.append(IsolationForest(behaviour='new', max_samples='auto', contamination=0.1))
            classifier.append(IsolationForest(behaviour='new', max_samples=int(X_train_clean.shape[0]/2), contamination=0.2))
            classifier_group.append(classifier)

            #######################################################################
            ######## Test the features using the Local Outlier Factor #############
            #######################################################################  

            classifier = []    
            classifier.append(LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1))
            classifier.append(LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.2))
            classifier_group.append(classifier)

            for clGroup in classifier_group:
                #print("Classifier:", str(clGroup[0]).split("(")[0],end='\n')
                for index, cl in enumerate(clGroup):
                    cl.fit(X_train_clean)

                    # Predict
                    predicted_label = cl.predict(cv_test).reshape(-1,1)

                    if flag:
                        predicted = predicted_label
                        flag = False
                    else:
                        predicted = np.concatenate((predicted, predicted_label), axis=1)

                    calc = calc_score(cv_label, predicted_label)
                    
                    if np.isnan(calc): 
                        del comb[index]
                        continue
                    else: score.append(calc)

                ignore = remove_algorithms(score)
                f1 = print_results(cv_label, decide(predicted, ignore=ignore))
                f1 = round(f1,3)
                
                #print("Result:", result)
                result = result.append({'Classifier Group': str(clGroup[0]).split("(")[0], 'K-Fold': k, 'F1-Score': f1}, ignore_index=True)
    
    dataframe = result.groupby(['Classifier Group','K-Fold']).agg({"F1-Score" : 'mean'})
    dataframe.columns = ['F1-Score_mean']
    dataframe = dataframe.reset_index()
    print("Dataframe Content:\n", dataframe) 
   
if __name__ == '__main__':
	main()