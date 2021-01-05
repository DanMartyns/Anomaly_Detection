import struct
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import seaborn as sns

def main() :
    global dataset

    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f'  , '--file', help='Filename', required=True)
    args = parser.parse_args()

    # open the file
    fb=open(args.file,'rb')

    try:

        matrix = []
        
        while True:
            # read each line of the file
            record=fb.read(128)

            # break when the line was empty
            if(len(record) < 128): break

            # unpack the line
            line = list(struct.unpack('=16d',record))

            matrix.append(line)

        matrix= np.array(matrix)
        
        dataset = pd.DataFrame({ 'mean_activity' : matrix[:,0],\
                                'std_activity' : matrix[:,1],\
                                'mean_silence' : matrix[:,2], \
                                'std_silence' : matrix[:,3], \
                                'percentil_75' : matrix[:,4], \
                                'percentil_90' : matrix[:,5], \
                                'percentil_95' : matrix[:,6], \
                                'percentil_99' : matrix[:,7], \
                                'median' : matrix[:,8], \
                                'max' : matrix[:,9], \
                                'active_frequencies' : matrix[:,10], \
                                'diff_first_last_one' : matrix[:,11], \
                                'imd' : matrix[:,12], \
                                'aroonUp' : matrix[:,13], \
                                'min-max' : matrix[:,14], \
                                'target' : matrix[:,15] })

        # print(dataset)
        # plt.figure(figsize=(10, 7))
        # sns.heatmap(dataset.corr(),
        #             annot = True,
        #             fmt = '.2f',
        #             cmap='Blues')
        # plt.title('Correlação entre variáveis do dataset de Network features')
        # plt.show()

        features = [ 'mean_activity', 'std_activity', 'mean_silence', 'std_silence', 'percentil_75', 'percentil_90', 'percentil_95', 'percentil_99', 'median', 'max', 'mean_active_frequencies', 'diff_first_last_one', 'Directional_Movement_Index', 'aroonUp', 'diff_max_min']

        dataset = pd.DataFrame(data=matrix)
        dataset.columns = features + ['target']

        dataset['target'][dataset['target'] == 0] = 'normal'
        dataset['target'][dataset['target'] == 1] = 'anomaly'

        print(dataset)

        # Separating out the features
        x = dataset.loc[:, features].values

        # Separating out the target
        y = dataset.loc[:,['target']].values

        # Make an instance of the Model
        pca = PCA(n_components=2)

        # Standardizing the features
        principalComponents = pca.fit_transform(x)

        principalDf = pd.DataFrame(data = principalComponents,columns=['principal component 1','principal component 2'])

        finalDf=pd.concat([principalDf,dataset[['target']]],axis=1)

        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(111)
        ax.set_title('2 component PCA', fontsize = 20)
        targets = [ 'normal', 'anomaly' ]
        colors = ['g', 'r']
        for target, color in zip(targets,colors):
            indicesToKeep = finalDf['target'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                    , finalDf.loc[indicesToKeep, 'principal component 2']
                    , c = color
                    , s = 5)
        ax.legend(targets)
        ax.grid()
        plt.ylim(-10,20)
        plt.xlim(-1.5,0.5)
        plt.show()
        
    except KeyboardInterrupt:
        print('>>> Interrupt received, stopping...')
    except Exception as e:
        print(e)
    finally:
        fb.close()

if __name__ == '__main__':

    main()