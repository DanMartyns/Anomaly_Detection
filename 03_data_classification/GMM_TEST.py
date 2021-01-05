# Import libraries
import os
import math
import random
import struct
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from astropy.visualization import hist
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

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
    print("Accuracy:",acc)
    print("Precision:", precision)
    print("F1-Score:", f1_score)
    print("#####################################################")    
    return precision, f1_score

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

    train_files = []
    anomaly_test_files = []
    regular_test_files = []

    # divide filenames in normal or anomaly
    for f in args.files:
        if args.wildcard in f:
            train_files.append(f)
        else:
            anomaly_test_files.append(f)
    
    score = []
    flag = True
    
    # begin process of deciding test and train files
    ratio = 0.8
    split = math.floor(ratio*len(train_files))

    if split > 0:
        while len(train_files) > split:
            random.shuffle(train_files)
            regular_test_files.append(train_files.pop(0))

    test_files = anomaly_test_files + regular_test_files
    
    features = [ 'mean_activity', 'std_activity', 'mean_silence', 'std_silence', \
    'percentil_75', 'percentil_90', 'percentil_95', 'percentil_99', 'median', \
    'max', 'mean_active_frequencies', 'diff_first_last_one', 'DMI', 'AroonUP', \
    'diff_max_min', 'meanConseqAct', 'meanConseqActiveFreq', 'EMA']
    scaler = StandardScaler()

    ############################################################################
    ######## Read the train and test files and convert to a dataset ############
    ############################################################################
    
    dataset_train = pd.DataFrame(data=readFileToMatrix(train_files))
    dataset_test = pd.DataFrame(data=readFileToMatrix(test_files))

    dataset_train.columns = features + ['target']
    dataset_test.columns = features + ['target']

    Y_train = dataset_train['target']
    X_train = dataset_train.drop(['diff_max_min','meanConseqAct','max','EMA','DMI','target'], axis = 1)
    X_train = scaler.fit_transform(X_train)

    Y_test = dataset_test['target']
    X_test = dataset_test.drop(['diff_max_min','meanConseqAct','max','EMA','DMI','target'], axis = 1)
    X_test = scaler.fit_transform(X_test)

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

    dataset_all = pd.concat([dataset_train,dataset_test])
    dataset_all = scaler.fit_transform(dataset_all)

    n_estimators = np.arange(1, 20)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    clfs = [GaussianMixture(n, max_iter=5000, covariance_type=cv).fit(dataset_all) for n in n_estimators for cv in cv_types]
    bics = [clf.bic(dataset_all) for clf in clfs]
    aics = [clf.aic(dataset_all) for clf in clfs]

    # Bayesian Information Criterion
    plt.plot(n_estimators, bics, label='BIC')
    # Akaike Information Criterion
    plt.plot(n_estimators, aics, label='AIC')
    plt.legend()
    plt.show()

    clf = GaussianMixture(2, max_iter=5000, covariance_type='spherical').fit(X_train)

    plt.hist(X_train, 1000, density=True, alpha=0.5)
    plt.xlim(0,20)
    plt.show()

    true_normal = np.where(Y_test == 0)[0]
    true_outliers = np.where(Y_test == 1)[0]

    scores = clf.score_samples(X_test)
    thresh = np.quantile(scores, .1)
    print("Threshold:",thresh)
    
    do = np.where(scores < thresh)[0]
    values = X_train[do]
    
    plt.scatter(X_train[:,0], X_train[:,1])
    plt.scatter(values[:,0], values[:,1], color='r')
    plt.show()
    
    calc_score(true_normal, true_outliers, do)

    kde = KernelDensity(bandwidth=1).fit(X_train)
    scores_kde = kde.score_samples(X_test)
    thresh_kde = np.quantile(scores_kde, .1)
    print("Threshold:",thresh_kde)
    do_kde = np.where(scores_kde < thresh_kde)[0]
    values = X_train[do_kde]

    plt.scatter(X_train[:,0],X_train[:,1])
    plt.scatter(values[:,0], values[:,1], color='r')
    plt.show()

    calc_score(true_normal, true_outliers, do_kde)

if __name__ == '__main__':
    main()