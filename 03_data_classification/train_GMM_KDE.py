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
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h, h


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

def calc_score(true_normal, true_outliers, detected_outliers):
    print("#####################################################")
    print("True Outliers:", len(true_outliers))
    print("True Normal:", len(true_normal))
    tn = len([x for x in true_outliers if x in detected_outliers])
    fp = len(set(true_outliers)) - tn
    fn = len([x for x in true_normal if x in detected_outliers])
    tp = len(set(true_normal)) - fn
    
    print("Confusion matrix:")
    print(pd.DataFrame([[tp, fp], [fn, tn]], ["Pred Normal", "Pred Anomaly"], ["Actual Normal", "Actual Anomaly"]))
    acc = (tn+tp)/(tn+fp+fn+tp)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2 * ((precision * recall)/(precision + recall))
    print(color.BOLD+"Accuracy:"+color.END, "{:.2f}%".format(round(acc,3)*100),"%")
    print(color.BOLD+"Precision:"+color.END, "{:.2f}%".format(round(precision,3)*100),"%")
    print(color.BOLD+"Recall:"+color.END, "{:.2f}%".format(recall*100))
    print(color.BOLD+"F1-Score: {:.2f}%".format(f1_score*100)+color.END)
    print("#####################################################")    
    return precision, f1_score*100, tp, fp, fn, tn

def print_results(anomaly_pred, regular_pred):
    an = ((anomaly_pred[anomaly_pred == -1].size)/anomaly_pred.shape[0])*100
    re = ((regular_pred[regular_pred == 1].size)/regular_pred.shape[0])*100
    y_pred = np.concatenate((anomaly_pred, regular_pred))
    y_true = np.concatenate((np.full(anomaly_pred.shape[0], -1), np.full(regular_pred.shape[0], 1)))
    # print("----------------------------------------------------------")
    # print("Number of Anomalies : ",y_true[y_true == -1].size)
    # print("Number of Normals : ", y_true[y_true == 1].size) 
    # print("Average success anomaly: {:.4f}%".format(an))
    # print("Average success regular: {:.4f}%".format(re))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2 * ((precision * recall)/(precision + recall))*100
    # print("Score: {:.2f}%".format(f1_score))
    # print("Confusion matrix:")
    # print(pandas.DataFrame([[tp, fp], [fn, tn]], ["Pred Normal", "Pred Anomaly"], ["Actual Normal", "Actual Anomaly"]))
    # print("----------------------------------------------------------")
    # return (int(an), int(re))

    return f1_score, tn, fp, fn, tp

def choose_algorithm(argument): 
    algorithms = { 

        "GMM": [  
            GaussianMixture(2, max_iter=1, covariance_type='full'),
            GaussianMixture(3, max_iter=1, covariance_type='full'),
            GaussianMixture(2, max_iter=50, covariance_type='full'),
            GaussianMixture(3, max_iter=50, covariance_type='full')
            ],
        "KDE": [
            KernelDensity(bandwidth=0.01),
            KernelDensity(bandwidth=0.1)
            ]
    } 
  
    # get() method of dictionary data type returns  
    # value of passed argument if it is present  
    # in dictionary otherwise second argument will 
    # be assigned as default value of passed argument 
    return algorithms.get(argument, "SVM")

def remove_algorithms(score):
    remv = copy.deepcopy(score)
    score.sort(reverse=True)
    median = score[math.floor(len(score)/2)]
    print("Median:", median)
    step = math.floor((score[0] - score[len(score)-1])/len(score))
    values = []

    for i in range(1, len(score)):
        if score[i] < median and (math.floor(score[i-1] - score[i]) >= step or median - score[i] > 2*step):
            print("Score removed:", score[i])
            values.append(score[i])

    return [i for i, x in enumerate(remv) if x in values]

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


def predict(files, scaler, clf, pca, feat):
    data = pd.DataFrame(data=readFileToMatrix(files))
    data = data.iloc[:, list(feat.keys()) + [-1]]
    data.columns = list(feat.values()) + ['target']
    
    Y = data['target']
    X = data.drop('target', axis=1)

    data = scaler.transform(X)
    data = pca.transform(data)

    return clf.predict(data)


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
    parser.add_argument("-alg","--algorithm", required=True)    
    # Wants to export files
    parser.add_argument("-e","--export",  action='store_true')
    # Print confusion matrix for each algorithm
    parser.add_argument("-v","--verbose",  action='store_true')
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

    clean_files = []
    anomaly_files = []

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
    
    CV_train = []
    CV_test_normal = []
    CV_test_anomaly = []
    print("\nClean Files: ", clean_files, sep='\n')
    print("\nAnomaly Files: ", anomaly_files, sep='\n')

    kf = KFold(n_splits=6)
    conf_matrix = (0,0,0,0)

    for train_index, test_index in kf.split(clean_files):
        CV_train.append(train_index)
        CV_test_normal.append(test_index)

    for train_index, test_index in kf.split(anomaly_files):
        CV_test_anomaly.append(test_index)

    f1Score = []
    timer = []
    precis = []
    size_train = []
    size_test = []
    print("\n\n\n")
    for train, testN, testA in zip(CV_train, CV_test_normal, CV_test_anomaly):
        print("File indexes")
        print("TRAIN:",train,"TEST NORMAL:", testN, "TESTE ANOMALY:", testA)

        cv_train = clean_files[train]
        cv_test_normal = clean_files[testN]
        cv_test_anomaly = anomaly_files[testA]
        print("\nAnomaly Files:\n",*cv_test_anomaly, sep="\n")
        print("\nClean Files:\n",*cv_test_normal, sep="\n")

        train_data = pd.DataFrame(data=readFileToMatrix(cv_train))
        train_data = train_data.iloc[:,list(feat.keys()) + [-1] ]
        train_data.columns = list(feat.values()) + ['target']
        size_train.append(train_data.shape[0])
        print("Train Data shappe:", train_data.shape)

        test_data = pd.DataFrame(data=readFileToMatrix(np.concatenate((cv_test_normal, cv_test_anomaly))))
        test_data = test_data.iloc[:,list(feat.keys()) + [-1] ]
        test_data.columns = list(feat.values()) + ['target']
        size_test.append(test_data.shape[0])
        print("Test Data shappe:", test_data.shape)

        Y = train_data['target']
        X = train_data.drop('target', axis=1)

        Y_test = test_data['target']
        X_test = test_data.drop('target', axis=1)

        ######################################################################
        ############### Standarize the values of the train data ##############
        ######################################################################

        scaler = StandardScaler()

        # Fit on training set only.
        X = scaler.fit_transform(X)
        X_test = scaler.transform(X_test)

        ######################################################################
        ############### Apply PCA to feature reduction #######################
        ######################################################################    

        # Feature Dimensionality Reduction
        pca = PCA(n_components=4)
        X = pca.fit_transform(X)
        X_test = pca.transform(X_test)

        ######################################################################
        ######################## GMM implementation ##########################
        ######################################################################  

        ############################################################################
        ####################### How many Gaussians? ################################
        # Given a model, we can use one of several means to evaluate how well ######
        # it fits the data. For example, there is the Aikaki Information ###########
        # Criterion (AIC) and the Bayesian Information Criterion (BIC) #############
        ############################################################################

        # The BIC criterion can be used to select the number of components in a 
        # Gaussian Mixture in an efficient way. In theory, it recovers the true number 
        # of components only in the asymptotic regime (i.e. if much data is available 
        # and assuming that the data was actually generated i.i.d. from a mixture 
        # of Gaussian distribution)

        # dataset_clean = pd.DataFrame(data=readFileToMatrix(clean_files))
        # dataset_clean = dataset_clean.iloc[:,list(features.keys()) + [-1] ]
        # dataset_clean.columns = list(features.values()) + ['target']
        
        # dataset_dirty = pd.DataFrame(data=readFileToMatrix(anomaly_files))
        # dataset_dirty = dataset_dirty.iloc[:,list(features.keys()) + [-1] ]
        # dataset_dirty.columns = list(features.values()) + ['target']

        # dataset_all = pd.concat((dataset_clean.drop('target', axis=1),dataset_dirty.drop('target', axis=1)) , axis=0)
        # dataset_all = scaler.fit_transform(dataset_all)

        # n_estimators = np.arange(1, 20)
        # cv_types = ['spherical', 'tied', 'diag', 'full']
        # clfs = [GaussianMixture(n, max_iter=5000, covariance_type='spherical').fit(dataset_all) for n in n_estimators]
        # bics = [clf.bic(dataset_all) for clf in clfs]
        # aics = [clf.aic(dataset_all) for clf in clfs]

        # print(bics)
        # print(aics)

        # # Bayesian Information Criterion
        # plt.plot(n_estimators, bics, label='BIC')
        # # Akaike Information Criterion
        # plt.plot(n_estimators, aics, label='AIC')
        # plt.legend()
        # plt.show()

        Y_test[Y_test == 1] = -1
        Y_test[Y_test == 0] = 1
        Y_test = np.array(Y_test).reshape(-1,1)

        true_normal = np.where(Y_test == 1)[0]
        true_outliers = np.where(Y_test == -1)[0] 


        start = time.time()
        for cl in choose_algorithm(args.algorithm):
            print("Classifier:", cl)
            
            clas = cl.fit(X)
            
            scores = clas.score_samples(X_test)
            thresh = np.quantile(scores, .4)
            do = np.where(scores < thresh)[0]
            values = X_test[do]

            # plt.scatter(X_test[:,0],X_test[:,1])
            # plt.scatter(values[:,0], values[:,1], color='r')
            # plt.show()

            precision, f1_score, tp, fp, fn, tn = calc_score(true_normal, true_outliers, do)
            f1Score.append(f1_score)
            precis.append(precision*100)
            print("F1-Score:", f1_score,"Precision:", precision*100)
            conf_matrix = (conf_matrix[0]+tp,conf_matrix[1]+fp,conf_matrix[2]+fn,conf_matrix[3]+tn)

        end = time.time() - start
        timer.append(end)
        print("Time:", end)
    pr, minInterval, maxInterval, interval = mean_confidence_interval(precis)
    print("\nPrecision:",round(pr,2),"Intervalo Minimo:", round(minInterval,2), "Intervalo Max:", round(maxInterval,2), "Interval:", round(interval,2))
    
    f1, minInterval, maxInterval, interval = mean_confidence_interval(f1Score)
    print("F1-Score:",round(f1,2),"Intervalo Minimo:", round(minInterval,2), "Intervalo Max:", round(maxInterval,2), "Interval:", round(interval,2))    
    
    
    print(color.BOLD+" Confusion matrix:"+color.END)
    print(pd.DataFrame([[ int(conf_matrix[0]/len(f1Score)), int(conf_matrix[1]/len(f1Score)) ], [ int(conf_matrix[2]/len(f1Score)) , int(conf_matrix[3]/len(f1Score)) ]], ["Actual Normal", "Actual Anomaly"], ["Predicted Normal", "Predicted Anomaly"]))
    mean, minInterval, maxInterval, interval = mean_confidence_interval(timer)
    print("Mean-time:", round(mean,2) ,"Intervalo Minimo:", round(minInterval,2), "Intervalo Max:", round(maxInterval,2), "Interval:", round(interval,2))
    print(mean_confidence_interval(size_train))
    print(mean_confidence_interval(size_test))
if __name__ == '__main__':
	main()
