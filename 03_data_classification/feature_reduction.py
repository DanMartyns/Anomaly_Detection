import os
import sys
import math
import copy
import time
import heapq
import struct
import random
import argparse
import scipy.stats
import numpy as np
import pandas as pd
from sklearn import svm
import dash
import plotly.express as px
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.neighbors import LocalOutlierFactor

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

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
    
    for i in range(1,200):
        
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
    
    '''
        At this moment we find the best features to model our problem, using the strategy of the points-based classification system.
        We repeat the process n times. The n=10 was sufficient to features keep in the same position of the classification system.
    '''

    #####################################################################
    #################### Plot the feature importance ####################
    #####################################################################

    # z = [y for x,y in classification_system.items()]
    # x = [x for x,y in classification_system.items()]
    # ind = np.arange(len(z))
    # fig = make_subplots(rows=1, cols=1)
    # fig.add_trace(go.Bar(
    #                         x = x, 
    #                         y = z,
    #                     ),
    #                     row=1, col=1) 
    # fig.update_traces(texttemplate='%{y:.2s}', textposition='outside')
    # fig.update_layout(
    #     xaxis_title="Features",
    #     yaxis_title="Points",
    #     xaxis_tickangle=-45,
    #     font_family="Courier New",
    #     title={
    #         'text': "Points-based Classification System",
    #         'y':0.95,
    #         'x':0.5,
    #         'xanchor': 'center',
    #         'yanchor': 'top'},
    #     autosize = False,
    #     width= 1500,
    #     height= 600
    #     )

    # app.layout = html.Div(children=[

    #     dcc.Graph(
    #         id='example-graph',
    #         figure=fig
    #     )

    # ])

    # if __name__ == '__main__':
    #     app.run_server(debug=False, port=8055)

    classification_system = dict(sorted(classification_system.items(), key=lambda x: x[1]))

    featuresKeeped = [ feature for index, feature in enumerate(classification_system.keys()) if index < components]
    result = {}

    for index, feature in enumerate(features):
        if feature in featuresKeeped:
            result[index] = feature
    
    print("Result:", result)
    return result

def combinations(gamma, kernel, nu, degree):
    result = ( ["classifier.append(svm.OneClassSVM(gamma={}, nu={}, kernel='{}', degree={}))".format(g,n,k,d)]
            for g in gamma
            for k in kernel
            for n in nu
            for d in degree)

    result = [item for sublist in list(result) for item in sublist]
    return dict( (index,['OC-SVM', value]) for index, value in enumerate(result)) 

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
    print("F1-Score: {:.2f}%".format(f1_score))
    print("Precision:{:.2f}%".format(precision))
    print("Recall:{:.2f}%".format(recall))
    print("Confusion matrix:")
    print(pd.DataFrame([[tp, fp], [fn, tn]], ["Pred Normal", "Pred Anomaly"], ["Actual Normal", "Actual Anomaly"]))
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
        if score[i] < median and (math.floor(score[i-1] - score[i]) >= 0.5*step or median - score[i] > 2*step):
            values.append(score[i])

    print([x for i, x in enumerate(remv) if x not in values])
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
def predict(files, scaler, clf, pca, feat):
    # print("Files 0:\n", *files, sep='\n')
    data = pd.DataFrame(data=readFileToMatrix(files))
    # print("Data 0 :\n",data)
    # print("Features :\n",feat)
    data = data.iloc[:, list(feat.keys()) + [-1] ]
    data.columns = list(feat.values()) + ['target']
    # print("Data 1 :\n",data)
    # data.columns = features + ['target']

    Y = data['target']
    X = data.drop('target', axis=1)

    Y[Y == 1] = -1
    Y[Y == 0] = 1
    Y = np.array(Y).reshape(-1,1)

    data = scaler.transform(X)
    # data = pca.transform(data)

    return clf.predict(data)    

def choose_algorithm(argument): 

    gamma = [0.001, 0.01, 0.1, 0.5]
    kernel = ['poly', 'linear', 'sigmoid']
    nu = [0.001, 0.01, 0.5 ]
    degree = [3]

    classifier = []
    comb = combinations(gamma, kernel, nu, degree)

    for index, cl in comb.items():
        exec(cl[1])

    classifier =  [ 
                svm.OneClassSVM(gamma=0.001, kernel='linear', nu=0.001),
                svm.OneClassSVM(gamma=0.01, kernel='poly', nu=0.01),
                svm.OneClassSVM(degree=1, gamma='auto', kernel='poly'),
                svm.OneClassSVM(gamma=0.001, nu=0.001),
                svm.OneClassSVM(gamma=0.001, nu=0.01),
                svm.OneClassSVM(gamma=0.01, kernel='sigmoid', nu=0.01),
                svm.OneClassSVM(gamma=0.01, kernel='sigmoid', nu=0.001)
            ]
    algorithms = { 
        "SVM": classifier
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
    # Read from files
    parser.add_argument("-c", "--comp", type=int, required=True)
    # Wildcard to detect what is normal
    parser.add_argument("-w", "--wildcard", required=True)
    # Print confusion matrix for each algorithm
    parser.add_argument("-v","--verbose",  action='store_true')              
    args=parser.parse_args()

    comp = args.comp

    # Check if we have arguments to work on
    if not args.files:
        args.files = []
    elif not (args.files or args.directory):
        parser.error("No files given, add --files or --directory.")

    
    ## Inicialize the variables

    clean_files = []
    anomaly_files = []
    
    CV_train_clean = []
    CV_train_anomaly = []
    CV_test_normal = []
    CV_test_anomaly = []

    kfold_splits = 3
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
    feat = featuresToKeep(readFileToMatrix(args.files), comp)

    # Divide filenames in normal or anomaly depending on the wildcard
    for f in args.files:
        if args.wildcard in f:
            clean_files.append(f)  
        else:
            anomaly_files.append(f)

    
    clean_files = np.array(clean_files)
    anomaly_files = np.array(anomaly_files)

    # classif = {}

    # Split Clean and Anomalous files into train and test datasets
    for clean, anomaly in zip(kf.split(clean_files), kf.split(anomaly_files)):
        train_index_clean, test_index_clean = clean
        train_index_anomaly, test_index_anomaly = anomaly

        CV_train_clean.append(train_index_clean)
        CV_train_anomaly.append(train_index_anomaly)

        CV_test_normal.append(test_index_clean) 
        CV_test_anomaly.append(test_index_anomaly)     
 
    # Test each algorithm for each k-Fold Split
    for kfold_index, values in enumerate(zip(CV_train_clean, CV_train_anomaly, CV_test_normal, CV_test_anomaly)):
        train_N, train_A, testN, testA = values
        print("\nFILE INDEXES")
        print("TRAIN CLEAN:",train_N ,"TRAIN ANOMALY:", train_A ,"TEST NORMAL:", testN , "TESTE ANOMALY:", testA)

        # Select the files to each group
        cv_train_normal = clean_files[train_N]
        cv_train_anomaly = anomaly_files[train_A]
        cv_test_normal = clean_files[testN]
        cv_test_anomaly = anomaly_files[testA]
        print("\nAnomaly Files:\n",*cv_test_anomaly, sep="\n")
        print("\nClean Files:\n",*cv_test_normal, sep="\n")
        
        # Read file into a DataFrame
        train_data = pd.DataFrame(data=readFileToMatrix(cv_train_normal))
        # Select the columns corresponding to the selected features
        train_data = train_data.iloc[:,list(feat.keys()) + [-1] ]
        # To each column assign the feature name
        train_data.columns = list(feat.values()) + ['target']    
        # train_data.columns = features + ['target']
        
        # Split the data between data and labels
        Y = train_data['target']
        X = train_data.drop('target', axis=1)

        # Standarize the values of the train data
        scaler = StandardScaler()

        # Fit on training set only.
        X = scaler.fit_transform(X)
        
        # Apply PCA to feature reduction        
        pca = PCA(n_components=comp)
        # X = pca.fit_transform(X)
        
        score = []
        flag = True
        start = time.time()
        for cl in choose_algorithm("SVM"):
            
            print("\n",color.BOLD+"Classificador:"+color.END,cl)
                    
            cl.fit(X)

            # Predict
            regu_data = predict( cv_test_normal, scaler, cl, pca, feat ).reshape(-1,1)
            anom_data = predict( cv_test_anomaly, scaler, cl, pca, feat ).reshape(-1,1)

            calc = calc_score(anom_data, regu_data)
            print("Calc:", calc)

            # if cl not in classif:
            #     classif[cl] = [calc]
            # else:
            #     classif[cl] += [calc]

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
  

    #print(*classif.items())

    ######################################################################
    ######### Plot F1-Score depending the number of Features #############
    ######################################################################  

    # fig = make_subplots(rows=1, cols=1)
    # fig.add_trace(go.Bar(
    #                         x = [ x for x,y in f1Score.items()], 
    #                         y = [ y for x,y in f1Score.items()]
    #                     ),
    #                     row=1, col=1) 
    # fig.update_traces(texttemplate='%{y:.2s}%', textposition='outside')
    # fig.update_yaxes(range=[0,100], dtick=5, row=1, col=1)
    # fig.update_layout(
    #     font_family="Courier New",
    #     font_size=15,
    #     title={
    #         'text': "F1-Score using ETC",
    #         'y':0.95,
    #         'x':0.5,
    #         'xanchor': 'center',
    #         'yanchor': 'top'},
    #     autosize = False,
    #     width= 1700,
    #     height= 800
    #     )

    # app.layout = html.Div(children=[

    #     dcc.Graph(
    #         id='example-graph',
    #         figure=fig
    #     )

    # ])

    # if __name__ == '__main__':
    #     app.run_server(debug=False, port=8051)


if __name__ == '__main__':
	main()