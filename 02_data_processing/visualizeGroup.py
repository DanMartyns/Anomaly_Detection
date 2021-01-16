import os
import sys
import struct
import copy
import math
import argparse
import numpy as np
import pandas as pd
from glob import glob
import itertools
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
np.set_printoptions(threshold=sys.maxsize)

import warnings
warnings.simplefilter('error')

# last EMA
last_EMA_f = 0
# last max
last_max_f = 0

n = 0

# features name
features = [ 'mean_activity_f', 'median_activity_f', 'std_activity_f', 'max_activity_f', 'min_activity_f', \
            'percentil75_activity_f', 'percentil90_activity_f', 'percentil95_activity_f', 'percentil99_activity_f', \
            'mean_silence_f', 'median_silence_f', 'std_silence_f', 'max_silence_f', 'min_silence_f', \
            'percentil75_silence_f', 'percentil90_silence_f', 'percentil95_silence_f', 'percentil99_silence_f', \
            'DFLP_dispersion_f', 'EMA_trading_f', 'DMI_trading_f', 'aroonUp_trading_f', \
            'mean_activity_fc', 'median_activity_fc', 'std_activity_fc', 'max_activity_fc', 'min_activity_fc', \
            'percentil75_activity_fc', 'percentil90_activity_fc', 'percentil95_activity_fc', 'percentil99_activity_fc', \
            'mean_activity_t', 'median_activity_t', 'std_activity_t', 'max_activity_t', 'min_activity_t', \
            'percentil75_activity_t', 'percentil90_activity_t', 'percentil95_activity_t', 'percentil99_activity_t', \
            'mean_silence_t', 'median_silence_t', 'std_silence_t', 'max_silence_t', 'min_silence_t', \
            'percentil75_silence_t', 'percentil90_silence_t', 'percentil95_silence_t', 'percentil99_silence_t']

"""
    Algorithm to pass this [1,0,1,1,1,0,1] to this [1,3,2]
"""
def group_list(lista):
    values = []
    
    for sample in lista:
        added = False
        l = []
        for value in sample:
            if l == [] and value == 1:
                l.append(value)
                added = True
            else:
                if value == 1:
                    if not added: 
                        l.append(value)
                        added = True 
                    else:
                        l[-1] += value
                elif value == 0:
                    added = False

        if l != []:
            values.append(round(np.mean(l),3))
        else:
            values.append(0)
        
    return values


def group_array(lista):
    result = []
    r = 0
    for value in lista:
        if value == 1: r += 1
        else:
            if r != 0:
                result.append(r)
                r = 0

    return result if result != [] else [0]

def window_analysis(observation_window, threshold, window_number, file):
    global last_EMA_f
    global last_max_f
    global n

    try:

        ######################################################################################
        ################## Apply the threshold and transcribe to 0s and 1s ###################
        ######################################################################################
        
        function = lambda x, threshold: 1 if x >= threshold else 0
        vfunc = np.vectorize(function)
        
        observation_window[:,2:-1] = vfunc(observation_window[:,2:-1], threshold)

        ######################################################################################
        ################################# Frequency/time #####################################
        ######################################################################################

        ob_window = []
        if ('anomaly' in file):
            ob_window = observation_window[ np.where(observation_window[:,-1] == 1) ][:,2:-1]
        else:
            ob_window = observation_window[:,2:-1]
        
        activity = ob_window
        silence = 1-activity
        
        ## Acitivity/Silence Features

        # Sum by lines
        sum_by_lines_a = activity.sum(axis=1)
        # print("Média:", np.mean(sum_by_lines_a))
        # print(sum_by_lines_a)
        # Mean Activity
        mean_activity = np.mean(sum_by_lines_a)
        # Median Activity
        median_activity = np.median(sum_by_lines_a)
        # Standard deviation Activity
        std_activity = np.std(sum_by_lines_a)
        # Percentil 75 activity
        percentil75_activity = np.percentile(sum_by_lines_a,75)
        # Percentil 90 activity
        percentil90_activity = np.percentile(sum_by_lines_a,90)
        # Percentil 95 activity
        percentil95_activity = np.percentile(sum_by_lines_a,95)
        # Percentil 99 activity
        percentil99_activity = np.percentile(sum_by_lines_a,99)
        # Min Activity
        min_activity = np.min(sum_by_lines_a)
        # Max Activity
        max_activity = np.max(sum_by_lines_a)

        # Sum by lines
        sum_by_lines = silence.sum(axis=1)

        # Mean Silence
        mean_silence = np.mean(sum_by_lines)
        # Median Silence
        median_silence = np.median(sum_by_lines)
        # Standard deviation Silence
        std_silence = np.std(sum_by_lines)
        # Percentil 75 silence
        percentil75_silence = np.percentile(sum_by_lines,75)
        # Percentil 90 silence
        percentil90_silence = np.percentile(sum_by_lines,90)
        # Percentil 95 silence
        percentil95_silence = np.percentile(sum_by_lines,95)
        # Percentil 99 silence
        percentil99_silence = np.percentile(sum_by_lines,99)
        # Min silence
        min_silence = np.min(sum_by_lines)
        # Max silence
        max_silence = np.max(sum_by_lines)

        dflp = []

        ## Dispersion Feature
        # Difference between the first and the last active position
        for sample in activity:
            indices = [i for i, x in enumerate(sample) if x == 1]
            if len(indices) > 1:
                dflp.append(indices[-1] - indices[0])
            else:
                dflp.append(0)
            
        DFLP = np.mean(dflp)

        ## Trading features
        
        # Aroon Up
        max_position = list(sum_by_lines_a).index(max(sum_by_lines_a))
        aroonUp_f = len(observation_window) - max_position

        # DMI
        DMI_f = 0
        if last_max_f != 0:
            UpMove = max_activity - last_max_f
            if UpMove > 0:
                DMI_f = UpMove

        last_max_f = max_activity

        # EMA
        # Exponential moving average = (Close - previous EMA) * (2 / n+1) + previous EMA
        # https://www.thebalance.com/simple-exponential-and-weighted-moving-averages-1031196
        sum_by_lines_a = observation_window[:,2:-1].sum(axis=1)
        mean_activity = np.mean(sum_by_lines_a)
        observation_windows_considered = 6
        k = 2/(observation_windows_considered + 1)
        EMA_f = 0
        if window_number == 1:
            EMA_f = round(mean_activity,3)
            last_EMA_f = EMA_f
        else:
            EMA_f = (mean_activity - last_EMA_f) * k + last_EMA_f
            EMA_f = round(EMA_f,3)
            last_EMA_f = EMA_f

        ######################################################################################
        ################################# Activity over time #################################
        ######################################################################################
        
        # Noise activates 2 frequencies, so it will be considered a moment of activity if an instant of time has more than two active frequencies
        function = lambda x: 1 if x > 3 else 0
        vfunc = np.vectorize(function)
        activity = observation_window[:,2:-1]
        activity = vfunc(activity.sum(axis=1))
        silence = 1 - activity

        if 'anomaly' in file and len(activity) > 2:
            # print("Activity:", activity)
            # print("Group act:", group_array(activity))
            tmp = copy.deepcopy(activity)
            for i in np.arange(0, len(activity)-10):
                if activity[i] == 1 and activity[i+10] == 1:
                    for x in np.arange(1,10):
                        tmp[i+x] = 1
            activity = tmp
            for i in np.arange(0, len(activity)-2):
                if activity[i] == 0 and activity[i+2]==0:
                    tmp[i+1] = 0

            # print('Temporar:',tmp)
            # print("Group tmp:", group_array(tmp),'\n')

        activity = group_array(activity)
        silence = group_array(silence)

        # if 'anomaly' in file and len(activity) > 2:
            # print(activity)
            # activity = sorted(activity, reverse=True)[:2]
        
        activity = np.array([x for x in activity if x != 1 and x!=2])

        # Mean Activity
        mean_activity_t = np.mean(activity) 
        # print("Mean Activity:", mean_activity_t)
        # Median Activity
        median_activity_t = np.median(activity)
        # Standard deviation Activity
        std_activity_t = np.std(activity)

        # Percentil 75 activity
        percentil75_activity_t = np.percentile(activity,75)
        # Percentil 90 activity
        percentil90_activity_t = np.percentile(activity,90)
        # Percentil 95 activity
        percentil95_activity_t = np.percentile(activity,95)
        # Percentil 99 activity
        percentil99_activity_t = np.percentile(activity,99)
        # Min Activity
        min_activity_t = np.min(activity)
        # Max Activity
        max_activity_t = np.max(activity)

        # Mean Silence
        mean_silence_t = np.mean(silence)
        # print("Mean Activity:", mean_activity_t.shape, "Mean Silence:", mean_silence_t.shape)
        # Median Silence
        median_silence_t = np.median(silence)
        # Standard deviation Silence
        std_silence_t = np.std(silence)
        # Percentil 75 silence
        percentil75_silence_t = np.percentile(silence,75)
        # Percentil 90 silence
        percentil90_silence_t = np.percentile(silence,90)
        # Percentil 95 silence
        percentil95_silence_t = np.percentile(silence,95)
        # Percentil 99 silence
        percentil99_silence_t = np.percentile(silence,99)
        # Min silence
        min_silence_t = np.min(silence)
        # Max silence
        max_silence_t = np.max(silence)

        ######################################################################################
        ############################# Consecutive Frequency/time #############################
        ######################################################################################    
        activity_fc = []
        function = lambda x: 1 if x > 2 else 0
        vfunc = np.vectorize(function)
        activity = observation_window[:,2:-1]
        activity = vfunc(activity.sum(axis=1))

        for ac, ob in zip(activity, ob_window):
            if ac == 1:
                activity_fc += group_array(ob)
        
        if activity_fc == []:
            activity_fc += [0]

        # print(activity_fc)
        ## Acitivity/Silence Features

        # Mean Activity
        mean_activity_fc = np.mean(activity_fc)
        # Median Activity
        median_activity_fc = np.median(activity_fc)
        # Standard deviation Activity
        std_activity_fc = np.std(activity_fc)

        # Percentil 75 activity
        percentil75_activity_fc = np.percentile(activity_fc,75)
        # Percentil 90 activity
        percentil90_activity_fc = np.percentile(activity_fc,90)
        # Percentil 95 activity
        percentil95_activity_fc = np.percentile(activity_fc,95)
        # Percentil 99 activity
        percentil99_activity_fc = np.percentile(activity_fc,99)
        # Min Activity
        min_activity_fc = np.min(activity_fc)
        # Max Activity
        max_activity_fc = np.max(activity_fc)

        target = np.max(observation_window[:, -1])

    except Exception as e:
        print(e)

    return [ mean_activity, median_activity, std_activity, max_activity, min_activity,
            percentil75_activity, percentil90_activity, percentil95_activity, percentil99_activity,
            mean_silence, median_silence, std_silence, max_silence, min_silence,
            percentil75_silence, percentil90_silence, percentil95_silence, percentil99_silence,
            DFLP, EMA_f, DMI_f, aroonUp_f,
            mean_activity_fc, median_activity_fc, std_activity_fc, max_activity_fc, min_activity_fc,
            percentil75_activity_fc, percentil90_activity_fc, percentil95_activity_fc, percentil99_activity_fc,
            mean_activity_t, median_activity_t, std_activity_t, max_activity_t, min_activity_t,
            percentil75_activity_t, percentil90_activity_t, percentil95_activity_t, percentil99_activity_t,
            mean_silence_t, median_silence_t, std_silence_t, max_silence_t, min_silence_t,
            percentil75_silence_t, percentil90_silence_t, percentil95_silence_t, percentil99_silence_t,
            target ]

def readFile_processFeatures(file, window_size, window_offset, threshold):

    # open the file
    fb=open(file,'rb')

    typeFile = ('normal' if 'normal' in file else 'anomaly')
    print("Type of File:", typeFile)

    print("Window Size:", window_size, "Window Offset:", window_offset)
    print("File size:", os.stat(file).st_size)
    
    # initialization of the observation window
    observation_window = []

    result = []

    try:
        
        # The number of the current Observation Window
        window_number = 1

        # The initialization of the number of samples
        sample = 0

        nr_anomalies = 0

        nr_samples = 0

        while True:
            # read each line of the file
            record=fb.read(696)

            # break when the line was empty
            if(len(record) < 696): break

            # unpack the line
            line = list(struct.unpack('=87d',record))
            
            nr_samples += 1
            if line[-1] == 1:
                nr_anomalies += 1
            
            # The observation window will have de size of window size
            if sample < window_size:
                # print("Line:",line)
                observation_window.append(line)
                    
                # increase the number of samples inside the window
                sample += 1
            else:
                
                # jump to the next position in the file
                fb.seek( 696*window_offset*window_number )
                
                # increase the number of windows
                window_number += 1
                
                # number of samples for each observation window
                sample = 0

                # print(observation_window)
                # analyse the window
                # remove the first 3 measurements
                if window_number >= 3:
                    data = window_analysis(np.array(observation_window), threshold, window_number, file)
                    if data != []:
                        data = [round(value,3) for value in data] + [typeFile]
                        result.append(data)
                               
                # reset the observation window
                observation_window = []
        print("Number of Anomalies:", nr_anomalies)
        print("Number of Samples:", nr_samples)
    except KeyboardInterrupt:
        print(">>> Interrupt received, stopping...")
    except Exception as e:
        print(e)
    finally:
        fb.close()

    df = pd.DataFrame(result, columns= features + ['target', 'file'])
    return df
def main() :
    
    # input arguments
    parser = argparse.ArgumentParser()
    # Read from an directory
    parser.add_argument("-d", "--directory", nargs='+')
    parser.add_argument('-t'  , '--threshold', type=float, help='the limit to consider a frequency active or not .', required=True)
    parser.add_argument('-ws', '--windowSize', type=int, help='the size of each observation window (seconds).', default=600)
    parser.add_argument('-wo', '--windowOffset', type=int, help='(seconds).', default=5)
    parser.add_argument('-p'  , '--port', help='port')
    args = parser.parse_args()

    files = []

    # get all filenames from directory
    if args.directory:
        for dir in args.directory:
            for r, d, f in os.walk(dir):
                for file in f:
                    if ".dat" in file:
                        files.append(os.path.join(r, file))


    files = sorted(files)

    print("Files:",*files, sep='\n')

    # register the results in dataframes
    frames = []
    for f in files:    
        print("File:", f)
        data = readFile_processFeatures(f, args.windowSize, args.windowOffset, args.threshold)
        frames.append(data)
        print("Nr Registers in file:", len(data))

    frames = pd.concat(frames)
    frames = frames.reset_index(drop=True)
    print("Frames:\n",frames)


    function = ['mean', 'median', 'std', 'max', 'min', 'percentil75', 'percentil90', 'percentil95', 'percentil99']
    base_idea = ['f', 't']

    l = [function, base_idea]

    combination = [(p[0]+'_activity_'+p[1], p[0]+'_silence_'+p[1]) for p in itertools.product(*l)] 

    comb = [ 'DFLP_dispersion_f', 'EMA_trading_f', 'DMI_trading_f', 'aroonUp_trading_f', \
            'mean_activity_fc', 'median_activity_fc', 'std_activity_fc', 'max_activity_fc', 'min_activity_fc', \
            'percentil75_activity_fc', 'percentil90_activity_fc', 'percentil95_activity_fc', 'percentil99_activity_fc' ]

    # DataFrame for each feature

    # Inicialize variables
    updateMenus = []
    data = []

    for feature_index, feature in enumerate(combination):
        print(feature)
        feature_activity, feature_silence = feature

        activity_clean   = list(frames[ (frames['file'] == 'normal') ][feature_activity])
        silence_clean    = list(frames[ (frames['file'] == 'normal') ][feature_silence ])
        activity_anomaly = list(frames[ (frames['file'] == 'anomaly') ][feature_activity]) 
        silence_anomaly  = list(frames[ (frames['file'] == 'anomaly') ][feature_silence ])
        
        maximum = max(len(activity_clean),len(silence_clean), len(activity_anomaly), len(silence_anomaly))

        dados = pd.DataFrame(data = dict({ 
                                    "normal activity"  : activity_clean  + [np.nan]* (maximum - len(activity_clean)),
                                    "normal silence"   : silence_clean   + [np.nan]* (maximum - len(silence_clean)), 
                                    "anomalous activity"  : activity_anomaly  + [np.nan]* (maximum - len(activity_anomaly)),
                                    "anomalous silence"   : silence_anomaly   + [np.nan]* (maximum - len(silence_anomaly)),    
                                                                                                                                        
                                }))
        
        # print(dados)

        # add traces
        data.append(go.Scatter(x = dados["normal activity"], y = dados["normal silence"], mode="markers", name="normal"))
        data.append(go.Scatter(x = dados["anomalous activity"], y = dados["anomalous silence"], mode="markers", name="anomaly"))
        

        label = ("Time" if feature_activity.split("_")[-1] == 't' else "Nr of Frequencies")+": "+feature_activity.split("_")[0].capitalize()
        updateMenus.append(
            dict(
                {
                    'label': label,
                    'method': 'update',
                    'args': [
                        {'visible': [False]*feature_index*2 + [True]*2 + [False]*((len(combination) - feature_index)*2 ) }
                    ]
                }  
            )
        )

    # for feature_index, feature in enumerate(comb):
    #     activity_clean   = list(frames[ (frames['file'] == 'normal') ][feature])
    #     activity_anomaly = list(frames[ (frames['file'] == 'anomaly') ][feature]) 

    #     if ('EMA' in feature) or ('DMI' in feature):
    #         data.append(go.Scatter(x=np.arange(0,len(activity_clean)), y=activity_clean, mode='markers', name='normal'))
    #         data.append(go.Scatter(x=np.arange(0,len(activity_anomaly)), y=activity_anomaly, mode='markers', name='anomaly'))
    #     else:
    #         data.append(go.Box(y=activity_clean, name='normal'))
    #         data.append(go.Box(y=activity_anomaly, name='anomaly'))

    #     l = [ 'DFLP', 'EMA', 'DMI', 'Aroon Up', 'Mean', 'Median', 'Std', 'Max', 'Min', 'Percentil 75', 'Percentil 90', 'Percentil 95', 'Percentil 99']

    #     label = ("Consecutive Active Frequencies" if feature.split("_")[-1] == 'fc' else ("Nr of Frequencies" if feature.split("_")[-1] == 'f' else 'Time'))+": "+l[feature_index]

    #     updateMenus.append(
    #         dict(
    #             {
    #                 'label': label,
    #                 'method': 'update',
    #                 'args': [
    #                     {'visible': [False]*feature_index*2 + [True]*2 + [False]*((len(comb) - feature_index)*2 ) }
    #                 ]
    #             }  
    #         )
    #     )        

    # update layout with buttons, and show the figure
    layout=go.Layout(updatemenus=list([dict(buttons=updateMenus,
                                            direction="down",
                                            pad={"r": 10, "t": 10},
                                            showactive=True,
                                            x=0.0,
                                            xanchor="left",
                                            y=1.15,
                                            yanchor="top")]),
                    xaxis_title="Activity time (seconds)",
                    yaxis_title="Silence time (seconds)",
                    # xaxis = dict(
                    #     tickmode = 'linear',
                    #     tick0 = 0,
                    #     dtick = 100
                    # ),
                    width = 600,
                    height = 600

                    )
    
    #defining figure and plotting
    fig = go.Figure(data,layout)

    fig.update_traces(
        line=dict(dash="dot", width=4),
        selector=dict(type="scatter", mode="lines"))

    fig.show()

if __name__ == '__main__':
	main()