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
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

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

    scaler = StandardScaler()

    dataset_clean = pd.DataFrame(data=readFileToMatrix(regular_files))
    dataset_dirty = pd.DataFrame(data=readFileToMatrix(anomaly_files))

    dataset_clean.columns = features + ['target']
    dataset_dirty.columns = features + ['target']

    Y_clean = dataset_clean['target']
    X_clean = dataset_clean.drop(['diff_max_min','meanConseqAct','max','EMA','DMI','target'], axis = 1)   

    Y_dirty = dataset_dirty['target']
    X_dirty = dataset_dirty.drop(['diff_max_min','meanConseqAct','max','EMA','DMI','target'], axis = 1)

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

    dataset_all = pd.concat([dataset_clean,dataset_dirty])
    dataset_all = scaler.fit_transform(dataset_all)
 
    n_estimators = np.arange(1, 20)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    clfs = [GaussianMixture(n, max_iter=5000, covariance_type='spherical').fit(dataset_all) for n in n_estimators]
    bics = [clf.bic(dataset_all) for clf in clfs]
    aics = [clf.aic(dataset_all) for clf in clfs]

    # # Bayesian Information Criterion
    # plt.plot(n_estimators, bics, label='BIC')
    # # Akaike Information Criterion
    # plt.plot(n_estimators, aics, label='AIC')
    # plt.legend()
    # plt.show()

    result = pd.DataFrame(columns=['Classifier', 'K-Fold', 'F1-Score']) 

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

            #######################################################################
            ############ Test the features using the Gaussian Mixture #############
            #######################################################################  

            clf = GaussianMixture(20, max_iter=5000, covariance_type='spherical').fit(X_train_clean)

            # plt.hist(X_train_clean, 1000, density=True, alpha=0.5)
            # plt.xlim(0,20)
            # plt.show()

            true_normal = np.where(cv_label == 1)[0]
            true_outliers = np.where(cv_label == -1)[0]

            scores = clf.score_samples(cv_test)
            thresh = np.quantile(scores, .1)
            # print("Threshold:",thresh)
            
            do = np.where(scores < thresh)[0]
            values = cv_test[do]
            
            # plt.scatter(cv_test[:,0],cv_test[:,1])
            # plt.scatter(values[:,0], values[:,1], color='r')
            # plt.show()           
           
            pr, f1_score = calc_score(true_normal, true_outliers, do)
            result = result.append({'Classifier': 'GMM', 'K-Fold': k, 'F1-Score': float("{:.2f}".format(f1_score*100))}, ignore_index=True)

            kde = KernelDensity(bandwidth=1).fit(X_train_clean)
            scores_kde = kde.score_samples(cv_test)
            thresh_kde = np.quantile(scores_kde, .1)
            # print("Threshold:",thresh_kde)
            do_kde = np.where(scores_kde < thresh_kde)[0]
            values = cv_test[do_kde]

            # plt.scatter(cv_test[:,0],cv_test[:,1])
            # plt.scatter(values[:,0], values[:,1], color='r')
            # plt.show()

            pr, f1_score = calc_score(true_normal, true_outliers, do_kde)
            result = result.append({'Classifier': 'KDE', 'K-Fold': k, 'F1-Score': float("{:.2f}".format(f1_score*100))}, ignore_index=True)

    dataframe = result.groupby(['Classifier','K-Fold']).agg({"F1-Score" : 'mean'})
    dataframe.columns = ['F1-Score_mean']
    dataframe = dataframe.reset_index()
    print("Dataframe Content:\n", dataframe) 

if __name__ == '__main__':
    main()