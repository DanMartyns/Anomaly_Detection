import os
import sys
import math
import time
import copy
import pandas
import struct
import random
import pickle
import argparse
import warnings
import scipy.stats
import numpy as np
import scipy.signal
import pandas as pd
from sklearn import svm
from sklearn import tree
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from pandas.plotting import autocorrelation_plot
from sklearn.linear_model  import LogisticRegression 
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings("ignore")

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

# features name
features = [ 'mean_activity_f', 'median_activity_f', 'std_activity_f', 'max_activity_f', 'min_activity_f', 
'percentil75_activity_f', 'percentil90_activity_f', 'percentil95_activity_f', 'percentil99_activity_f',
'mean_silence_f', 'median_silence_f', 'std_silence_f', 'max_silence_f', 'min_silence_f',
'percentil75_silence_f', 'percentil90_silence_f', 'percentil95_silence_f', 'percentil99_silence_f',
'DFLP_dispersion_f', 'EMA_trading_f', 'DMI_trading_f', 'aroonUp_trading_f',
'mean_activity_fc', 'median_activity_fc', 'std_activity_fc', 'max_activity_fc', 'min_activity_fc',
'percentil75_activity_fc', 'percentil90_activity_fc', 'percentil95_activity_fc', 'percentil99_activity_fc',
'mean_activity_t', 'median_activity_t', 'std_activity_t', 'max_activity_t', 'min_activity_t',
'percentil75_activity_t', 'percentil90_activity_t', 'percentil95_activity_t', 'percentil99_activity_t',
'mean_silence_t', 'median_silence_t', 'std_silence_t', 'max_silence_t', 'min_silence_t',
'percentil75_silence_t', 'percentil90_silence_t', 'percentil95_silence_t', 'percentil99_silence_t']


'''
    Select the features that maximize the F1-Score using the Extra Tree Classifier. 
    The features that maximize the result have been previously calculated.
'''
def featuresToKeep(data, components):

    dataset = pd.DataFrame(data=data)
    dataset.columns = features + ['target']

    ######################################################################
    ######################### Feature Selection ##########################
    ######################################################################    

    classification_system = { feature: 0 for feature in features}
    
    for i in range(1,50):
        
        # Building the model 
        extra_tree_forest = ExtraTreesClassifier(n_estimators = 5, criterion ='entropy', max_features = 2)

        # Training the model 
        extra_tree_forest.fit(dataset.drop('target', axis=1), dataset['target'])

        # Computing the importance of each feature 
        feature_importance = extra_tree_forest.feature_importances_ 

        # Normalizing the individual importances 
        feature_importance_normalized = np.std([tree.feature_importances_ for tree in 
                                            extra_tree_forest.estimators_], 
                                            axis = 0)
                
        sort = sorted( zip(features, feature_importance_normalized), key=lambda x: x[1], reverse=True)

        for index, tupl in enumerate(sort):
            classification_system[tupl[0]] += index

    classification_system = dict(sorted(classification_system.items(), key=lambda x: x[1]))    

    featuresKeeped = [ feature for index, feature in enumerate(classification_system.keys()) if index < components]
    result = {}

    for index, feature in enumerate(features):
        if feature in featuresKeeped:
            result[index] = feature
    
    print("Result:", result)
    return result

'''
    Calculate the mean confidance with a confidance interval of 95%
'''
def mean_confidence_interval(data, alpha=5.0):
    median = np.median(data)
    # calculate lower percentile (e.g. 2.5)
    lower_p = alpha / 2.0
    # retrieve observation at lower percentile
    lower = max(0.0, np.percentile(data, lower_p))
    # calculate upper percentile (e.g. 97.5)
    upper_p = (100 - alpha) + (alpha / 2.0)
    # retrieve observation at upper percentile
    upper = np.percentile(data, upper_p)

    interval = median - lower

    return median, lower, upper, interval

'''
    Read a file to matrix
'''
def readFileToMatrix(files) :
    assert len(files) > 0
    
    matrix = []
    for f in files:

        fb = open(f, "rb")
        filename = f.split('/')[-1].split("_")[0]

        try:
            while True:
                # read each line of the file
                record=fb.read(400)
                
                # break when the line was empty
                if(len(record) < 400): break
                
                # unpack the record
                line = list(struct.unpack('=50d',record))
                
                if ('anomaly' in filename and line[-1] == 1) or ('normal' in filename):    
                    # append the line to matrix
                    matrix.append(line)
    
        finally:
            fb.close()

    matrix = np.array(matrix)
    return matrix

'''
    Calculate the F1-Score 
'''
def calc_score(anomaly_pred, regular_pred):
    
    # Samples predicted and true
    predicted = np.concatenate((anomaly_pred, regular_pred))
    true = np.concatenate((np.full(anomaly_pred.shape[0], -1), np.full(regular_pred.shape[0], 1)))
    
    # Calculate precision, recall and F1-Score
    tn, fp, fn, tp = confusion_matrix(true, predicted).ravel()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    
    return 2 * ((precision * recall)/(precision + recall)) * 100


'''
    Print the performance results 
'''
def print_results(anomaly_pred, regular_pred):

    print("----------------------------------------------------------")
    
    # Number of Anomalies and Clean Samples
    y_pred = np.concatenate((anomaly_pred, regular_pred))
    y_true = np.concatenate((np.full(anomaly_pred.shape[0], -1), np.full(regular_pred.shape[0], 1)))
    nrAnomalies = y_true[y_true == -1].size
    nrClean = y_true[y_true == 1].size
    print("Number of Anomalies : ", nrAnomalies)
    print("Number of Normals : ", nrClean ) 
    
    # Hit Rate
    hitRateAnomaly = ((anomaly_pred[anomaly_pred == -1].size)/anomaly_pred.shape[0])*100
    hitRateClean = ((regular_pred[regular_pred == 1].size)/regular_pred.shape[0])*100
    print("Average success anomaly: {:.4f}%".format(hitRateAnomaly))
    print("Average success regular: {:.4f}%".format(hitRateClean))

    # Calculate precision, recall and F1-Score    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp/(tp+fp) * 100
    recall = tp/(tp+fn) * 100
    f1_score = 2 * ((precision * recall)/(precision + recall))
    print("Score: {:.2f}%".format(f1_score))
    print("Confusion matrix:")
    print(pandas.DataFrame([[tp, fp], [fn, tn]], ["Pred Normal", "Pred Anomaly"], ["Actual Normal", "Actual Anomaly"]))
    print("----------------------------------------------------------")

    return f1_score, precision, nrAnomalies, nrClean, tp, fp, fn, tn

'''
    Remove the algorithms with worse score
'''
def remove_algorithms(score):
    remv = copy.deepcopy(score)
    score.sort(reverse=True)
    median = score[math.floor(len(score)/2)]
    step = math.floor((score[0] - score[len(score)-1])/len(score))
    values = []

    for i in range(0, len(score)):
        if score[i] < median and (math.floor(score[i-1] - score[i]) >= step or median - score[i] > 2*step):
            values.append(score[i])

    return [i for i, x in enumerate(remv) if x in values]

'''
    Decide the algorithms to keep
'''
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

'''
    Predict the values
'''
def predict(files, scaler, clf, pca, features):
    data = pd.DataFrame(data=readFileToMatrix(files))
    data = data.iloc[:, list(features.keys()) + [-1]]
    data.columns = list(features.values()) + ['target']
    
    Y = data['target']
    X = data.drop('target', axis=1)

    Y[Y == 1] = -1
    Y[Y == 0] = 1
    Y = np.array(Y).reshape(-1,1)

    data = scaler.transform(X)
    data = pca.transform(data)

    return clf.predict(data)

def choose_algorithm(argument): 
    algorithms = { 
        "SVM": [  
                svm.OneClassSVM(gamma=0.001, kernel='linear', nu=0.001),
                svm.OneClassSVM(gamma=0.01, kernel='poly', nu=0.01),
                svm.OneClassSVM(degree=1, gamma='auto', kernel='poly'),
                svm.OneClassSVM(gamma=0.001, nu=0.001),
                svm.OneClassSVM(gamma=0.001, nu=0.01),
                svm.OneClassSVM(gamma=0.01, kernel='sigmoid', nu=0.01),
                svm.OneClassSVM(gamma=0.01, kernel='sigmoid', nu=0.001)
            ],
        "IF": [
                IsolationForest(max_samples=1),
                IsolationForest(max_samples=4999),
                IsolationForest(contamination=0.1),
                IsolationForest(contamination=0.2)
            ], 
        "LOF": [
                LocalOutlierFactor(n_neighbors=500, novelty=True, contamination=0.1),
                LocalOutlierFactor(n_neighbors=5000, novelty=True, contamination=0.1),
                LocalOutlierFactor(n_neighbors=5000, novelty=True, contamination=0.2)
            ] 
    } 
  
    # get() method of dictionary data type returns  
    # value of passed argument if it is present  
    # in dictionary otherwise second argument will 
    # be assigned as default value of passed argument 
    return algorithms.get(argument, "SVM") 

def main():

    parser = argparse.ArgumentParser()
    # Read from files
    parser.add_argument("-f", "--files", nargs='+')
    # Read from an directory
    parser.add_argument("-d", "--directory", nargs='+')
    # Wildcard to detect what is normal
    parser.add_argument("-w", "--wildcard", required=True)
    # Type of anomaly
    parser.add_argument("-a", "--anomaly")
    # Wants to export files
    parser.add_argument("-e","--export",  action='store_true')
    # Wants to export files
    parser.add_argument("-alg","--algorithm", required=True)  
    # Print confusion matrix for each algorithm
    parser.add_argument("-v","--verbose",  action='store_true')  
    args=parser.parse_args()

    # Check if we have arguments to work on
    if not args.files:
        args.files = []
    elif not (args.files or args.directory):
        parser.error("No files given, add --files or --directory.")

    # Inicialize the variables
    
    clean_files = []
    anomaly_files = []
    
    CV_train = []
    CV_test_normal = []
    CV_test_anomaly = []
    
    kfold_splits = 5
    kf = KFold(n_splits=kfold_splits)

    # Prealloc dataFrame; it's much more faster when add a new row
    results = pd.DataFrame(
        np.zeros((kfold_splits,9)), 
        columns = [ "Precision", "F1-Score", "TP", "FP", "FN", "TN", "NrAnomalies", "NrClean", "Time"]
    )

    meanResultsWithIntervals = pd.DataFrame(
        np.zeros((9,4)), 
        columns = ["Mean", "Interval", "Min", "Max"],
        index=[ "Precision", "F1-Score", "TP", "FP", "FN", "TN", "NrAnomalies", "NrClean", "Time"]
    )

    # get all filenames from directory
    if args.directory:
        for dir in args.directory:
            for r, d, f in os.walk(dir):
                for file in f:
                    if ".dat" in file:
                        args.files.append(os.path.join(r, file))

    
    # Read files and keep only the features selected
    # feat = featuresToKeep(readFileToMatrix(args.files), 5)
    feat = {0: 'mean_activity_f', 29: 'percentil95_activity_fc', 43: 'max_silence_t', 45: 'percentil75_silence_t', 47: 'percentil95_silence_t'}

    
    # Divide filenames in normal or anomaly depending on the wildcard
    for f in args.files:
        if args.wildcard in f:
            clean_files.append(f)  
        elif args.anomaly in f:
            anomaly_files.append(f)

    
    clean_files = np.array(clean_files)
    anomaly_files = np.array(anomaly_files)
    
    # print("\nClean Files: ", clean_files, sep='\n')
    # print("\nAnomaly Files: ", anomaly_files, sep='\n')

    
    # Split Clean and Anomalous files into train and test datasets
    for clean, anomaly in zip(kf.split(clean_files), kf.split(anomaly_files)):
        train_index_clean, test_index_clean = clean
        train_index_anomaly, test_index_anomaly = anomaly

        
        CV_train.append(train_index_clean)
        CV_test_normal.append(test_index_clean) 

        CV_test_anomaly.append(test_index_anomaly)     
 
    # Test each algorithm for each k-Fold Split
    for kfold_index, values in enumerate(zip(CV_train, CV_test_normal, CV_test_anomaly)):
        train, testN, testA = values
        print("\nFILE INDEXES")
        print("TRAIN:",train,"TEST NORMAL:", testN, "TESTE ANOMALY:", testA)

        # Select the files to each group
        cv_train = clean_files[train]
        cv_test_normal = clean_files[testN]
        cv_test_anomaly = anomaly_files[testA]
        print("\nAnomaly Files:\n",*cv_test_anomaly, sep="\n")
        print("\nClean Files:\n",*cv_test_normal, sep="\n")

        # Read file into a DataFrame
        train_data = pd.DataFrame(data=readFileToMatrix(cv_train))
        # Select the columns corresponding to the selected features
        train_data = train_data.iloc[:,list(feat.keys()) + [-1] ]
        # To each column assign the feature name
        train_data.columns = list(feat.values()) + ['target']    

        # Split the data between data and labels
        Y = train_data['target']
        X = train_data.drop('target', axis=1)


        # Standarize the values of the train data
        scaler = StandardScaler()

        # Fit on training set only.
        X = scaler.fit_transform(X)
        
        # Apply PCA to feature reduction        
        pca = PCA(n_components=4)
        X = pca.fit_transform(X)
        
        score = []
        flag = True
        start = time.time()
        for cl in choose_algorithm(args.algorithm):
            
            print("\n",color.BOLD+"Classificador:"+color.END,cl)
                    
            cl.fit(X)

            # Predict
            regu_data = predict( cv_test_normal, scaler, cl, pca, feat ).reshape(-1,1)
            anom_data = predict( cv_test_anomaly, scaler, cl, pca, feat ).reshape(-1,1)

            calc = calc_score(anom_data, regu_data)
            print("SCORE:", calc)

            if not np.isnan(calc):
            
                if flag:
                    regular_pred = regu_data
                    anomaly_pred = anom_data
                    flag = False
                else:
                    regular_pred = np.concatenate((regular_pred, regu_data), axis=1)      
                    anomaly_pred = np.concatenate((anomaly_pred, anom_data), axis=1)

                score.append(calc)
            
            else: continue

            if args.verbose:
                print_results(predict(cv_test_anomaly, scaler, cl, pca, features ), predict(cv_test_normal, scaler, cl, pca, features ))  

        # Check what models were ignored
        ignore = remove_algorithms(score)
        print("\nIgnored Algorithms:", ignore)
        
        # For each K-Fold Slipt calculate the F1-Score, Precision and Confusion Matrix
        f1_score, precision, nrAnomalies, nrClean, tp, fp, fn, tn = print_results(decide(anomaly_pred, ignore=ignore), decide(regular_pred, ignore=ignore))

        results.loc[kfold_index] =  { "Precision" : precision, 
                                       "F1-Score" : f1_score, 
                                             "TP" : tp, 
                                             "FP" : fp, 
                                             "FN" : fn, 
                                             "TN" : tn,
                                    "NrAnomalies" : nrAnomalies,
                                        "NrClean" : nrClean,
                                           "Time" : time.time() - start 
                                        } 

    print("\n",results)

    # Results with a mean confidance interval of 95%
    for function in results.columns:

        mean, minInterval, maxInterval, interval = mean_confidence_interval(results[function])
        meanResultsWithIntervals.loc[function] = {'Mean' : round(mean,2),
                                              'Interval' : round(interval,2),
                                                   'Min' : round(minInterval,2),
                                                   'Max' : round(maxInterval,2)    
                                            }

    print("\n",meanResultsWithIntervals)

    print(color.BOLD+"\nConfusion matrix:"+color.END)
    print(pd.DataFrame([[ int(float(meanResultsWithIntervals.loc['TP']['Mean'])), int(float(meanResultsWithIntervals.loc['FP']['Mean'])) ], [ int(float(meanResultsWithIntervals.loc['FN']['Mean'])) , int(float(meanResultsWithIntervals.loc['TN']['Mean'])) ]], ["Actual Normal", "Actual Anomaly"], ["Predicted Normal", "Predicted Anomaly"]))
    
if __name__ == '__main__':
	main()
