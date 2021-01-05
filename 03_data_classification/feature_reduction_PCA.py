import os
import sys
import math
import copy
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

def readFileToMatrix(files) :
    assert len(files) > 0
    
    matrix = []
    for f in files:

        fb = open(f, "rb")
        filename = f.split('/')[-1].split("_")[0]

        try:
            while True:
                # read each line of the file
                record=fb.read(424)
                
                # break when the line was empty
                if(len(record) < 424): break
                
                # unpack the record
                line = list(struct.unpack('=53d',record))
                
                if not (filename == 'anomaly' and line[-1] == 0):
                    # append the line to matrix
                    matrix.append(line)
    
        finally:
            fb.close()

    matrix = np.array(matrix)
    return matrix

def combinations(gamma, kernel, nu, degree):
    result = ( ["classifier.append(svm.OneClassSVM(gamma={}, nu={}, kernel='{}', degree={}))".format(g,n,k,d)]
            for g in gamma
            for k in kernel
            for n in nu
            for d in degree)

    result = [item for sublist in list(result) for item in sublist]
    return dict( (index,['OC-SVM', value]) for index, value in enumerate(result)) 

def calc_score(true_dataset, predicted_dataset):
    tn, fp, fn, tp = confusion_matrix(true_dataset, predicted_dataset, labels=[1,-1]).ravel()
    precision = precision_score(true_dataset, predicted_dataset, zero_division=1) * 100
    recall    = recall_score(true_dataset, predicted_dataset, zero_division=1) * 100
    f1Score   = f1_score(true_dataset, predicted_dataset, zero_division=1) * 100
    return f1Score

def print_results(true_dataset, predicted_dataset):
    tn, fp, fn, tp = confusion_matrix(true_dataset, predicted_dataset, labels=[1,-1]).ravel()
    precision = precision_score(true_dataset, predicted_dataset, zero_division=1) * 100
    recall    = recall_score(true_dataset, predicted_dataset, zero_division=1) * 100
    # accuracy  = accuracy_score(true_dataset, predicted_dataset) * 100
    f1Score   = f1_score(true_dataset, predicted_dataset, zero_division=1) * 100
    print("----------------------------------------------------------")
    print(color.BOLD+" Number of Anomalies : "+color.END,true_dataset[true_dataset == -1].size)
    print(color.BOLD+" Number of Normals : "+color.END, true_dataset[true_dataset == 1].size) 
    print(color.BOLD+" Precision:"+color.END, "{:.2f}%".format(precision))
    print(color.BOLD+" Recall:"+color.END, "{:.2f}%".format(recall))
    print(color.BOLD+" F1-Score:"+color.END, "{:.2f}%".format(f1Score))
    # print(color.BOLD+" Accuracy:"+color.END, "{:.2f}%".format(accuracy))
    print(color.BOLD+" Confusion matrix:"+color.END)
    print(pd.DataFrame([[tn, fp], [fn, tp]], ["Actual Normal", "Actual Anomaly"], ["Predicted Normal", "Predicted Anomaly"]))
    print("----------------------------------------------------------")
    
    # matrix = classification_report(true_dataset,predicted_dataset,labels=[1,-1])

    return f1Score, tn, fp, fn, tp

def remove_algorithms(score):
    remv = copy.deepcopy(score)
    values = []
    score.sort(reverse=True)

    median = score[math.floor(len(score)/2)]
    step = math.floor((score[0] - score[len(score)-1])/len(score))
    

    for i in range(1, len(score)):
        if score[i] <= median and (math.floor(score[i-1] - score[i]) >= step or median - score[i] > 2*step):
            values.append(score[i])
    
    removed = [i for i, x in enumerate(remv) if x in values]
    
    # Impedir que elimine todos os valores caso sejam todos iguais, obrigando a manter um
    if len(removed) == len(remv):
        if all(np.isnan(x) for x in removed):
            removed = []
        else:
            removed = range(1,len(remv))
    
    return removed 

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
    # Read from files
    parser.add_argument("-c", "--comp", type=int, required=True)    
    args=parser.parse_args()

    comp = args.comp

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

    data = readFileToMatrix(args.files)

    # Feature Selection
    features = [ 'mean_activity_f', 'median_activity_f', 'std_activity_f', 'max_activity_f', 'min_activity_f', \
            'percentil75_activity_f', 'percentil90_activity_f', 'percentil95_activity_f', 'percentil99_activity_f', \
            'mean_silence_f', 'median_silence_f', 'std_silence_f', 'max_silence_f', 'min_silence_f', \
            'percentil75_silence_f', 'percentil90_silence_f', 'percentil95_silence_f', 'percentil99_silence_f', \
            'DFLP_dispersion_f', 'EMA_trading_f', 'DMI_trading_f', 'aroonUp_trading_f', \
            'mean_activity_fc', 'median_activity_fc', 'std_activity_fc', 'max_activity_fc', 'min_activity_fc', \
            'percentil75_activity_fc', 'percentil90_activity_fc', 'percentil95_activity_fc', 'percentil99_activity_fc', \
            'EMA_trading_fc', 'DMI_trading_fc', 'aroonUp_trading_fc',
            'mean_activity_t', 'median_activity_t', 'std_activity_t', 'max_activity_t', 'min_activity_t', \
            'percentil75_activity_t', 'percentil90_activity_t', 'percentil95_activity_t', 'percentil99_activity_t', \
            'mean_silence_t', 'median_silence_t', 'std_silence_t', 'max_silence_t', 'min_silence_t', \
            'percentil75_silence_t', 'percentil90_silence_t', 'percentil95_silence_t', 'percentil99_silence_t']

    dataset = pd.DataFrame(data=data)
    dataset.columns = features + ['target']

    trys = 0  
    f1Score = { number: 0 for number in range(1,53)}
    conf_matrix = {number: (0,0,0,0) for number in range(1,53)}
    print("\nTeste para",comp,"componentes")

    for i in range(1,30):

        # test_size: what proportion of original data is used for test set
        train_dataset, test_dataset = train_test_split( dataset, test_size=0.2)

        train = train_dataset.drop('target', axis=1)
        train_label = train_dataset['target']

        test = test_dataset.drop('target', axis=1)
        test_label = test_dataset['target']

        ######################################################################
        ############### Standarize the values of the train data ##############
        ######################################################################

        scaler = StandardScaler()

        # Fit on training set only.
        scaler.fit(train)
        
        # Apply transform to both the training set and the test set.
        train = scaler.transform(train)
        test = scaler.transform(test)

        ######################################################################
        ############### Apply PCA to feature reduction #######################
        ######################################################################    

        # Feature Dimensionality Reduction
        pca = PCA(n_components=comp)
        pca.fit(train)

        train = pca.transform(train)
        test = pca.transform(test)

        ######################################################################
        ########### Test the features using the OC-SVM Model #################
        ######################################################################   

        score = []
        flag = True

        train_normal = train_label[train_label == 0]
        train_outliers = train_label[train_label == 1] 
        outlier_prop = len(train_outliers)/len(train_normal) if len(train_outliers)/len(train_normal) else 1

        gamma = [0.001, 0.01]
        kernel = ['poly', 'linear']
        nu = [0.001, 0.01, outlier_prop ]
        degree = [3]

        classifier = []
        comb = combinations(gamma, kernel, nu, degree)

        for index, cl in comb.items():
            exec(cl[1])

        test_label[test_label == 1] = -1
        test_label[test_label == 0] = 1
        test_label = np.array(test_label).reshape(-1,1)

        for index, cl in enumerate(classifier):
            print(color.BOLD+"Classificador:"+color.END,cl)
            cl.fit(train)

            # Predict
            predicted_label = cl.predict(test).reshape(-1,1)

            # Calculate the Score
            calc = calc_score(test_label, predicted_label)

            # if the calc is NaN
            if calc != 0: 
                
                # print(color.BOLD+"F1-Score:"+color.END,calc,"%")
                
                # Append Scores
                score.append(calc)

                if flag:
                    predicted = predicted_label
                    flag = False
                else:
                    # Join the predicted labels
                    predicted = np.concatenate((predicted, predicted_label), axis=1)

            else:
                # print(color.BOLD+"F1-Score:"+color.END,calc,"%")
                continue

        if score != []:
            ignore = remove_algorithms(score)
            f1, tn, fp, fn, tp = print_results(test_label, decide(predicted, ignore=ignore))
            f1Score[comp] += round(f1,3)
            conf_matrix[comp] = (conf_matrix[comp][0]+tn,conf_matrix[comp][1]+fp,conf_matrix[comp][2]+fn,conf_matrix[comp][3]+tp)
            print(conf_matrix[comp])    
            trys += 1            
    
    f1Score[comp] = f1Score[comp]/trys
    conf_matrix[comp] = (int(conf_matrix[comp][0]/trys),int(conf_matrix[comp][1]/trys),int(conf_matrix[comp][2]/trys),int(conf_matrix[comp][3]/trys))
    print("F1-Score:", f1Score[comp])
    print(color.BOLD+" Confusion matrix:"+color.END)
    print(pd.DataFrame([[conf_matrix[comp][0], conf_matrix[comp][1]], [conf_matrix[comp][2], conf_matrix[comp][3]]], ["Actual Normal", "Actual Anomaly"], ["Predicted Normal", "Predicted Anomaly"]))
        
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