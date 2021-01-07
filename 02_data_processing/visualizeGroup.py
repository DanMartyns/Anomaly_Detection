import os
import struct
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

# last EMA
last_EMA_f = 0
last_EMA_fc = 0
# last max
last_max_f = 0
last_max_fc = 0

# features name
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

def window_analysis(observation_window, threshold, window_number):
    global last_EMA_f
    global last_EMA_fc
    global last_max_f
    global last_max_fc

    ######################################################################################
    ################## Apply the threshold and transcribe to 0s and 1s ###################
    ######################################################################################
    # mean_samples = []
    # for i in observation_window[:,2:-1]:
    #     mean_samples.append(np.mean(i))
    
    function = lambda x, threshold: 1 if x >= threshold else 0
    vfunc = np.vectorize(function)
    
    observation_window[:,2:-1] = vfunc(observation_window[:,2:-1], threshold)
    # mean_samples = vfunc(mean_samples, threshold)
    # print(observation_window)
    # print(mean_samples)

    ######################################################################################
    ################################# Frequency/time #####################################
    ######################################################################################

    activity = observation_window[:,2:-1]
    # for x,y in zip(activity.sum(axis=1), observation_window[:,-1]):
    #     print(x, y)
    silence = 1-activity
    # for x in silence.sum(axis=1):
    #     print(x)

    ## Acitivity/Silence Features

    # Sum by lines
    sum_by_lines_a = activity.sum(axis=1)
    # print("Média:", np.mean(sum_by_lines_a))

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
    # EMA
    observation_windows_considered = 2
    k = 2/(observation_windows_considered + 1)
    EMA_f = 0
    if window_number == 1:
        EMA_f = round(mean_activity,3)
        last_EMA_f = EMA_f
    else:
        EMA_f = round(mean_activity * k + last_EMA_f * (1-k),3)
        last_EMA_f = EMA_f
    
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

    ######################################################################################
    ############################# Consecutive Frequency/time #############################
    ######################################################################################    

    activity_fc = group_list(activity)
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

    ## Trading features
    # EMA
    observation_windows_considered = 5
    k = 2/(observation_windows_considered + 1)
    EMA_fc = 0
    if window_number == 1:
        EMA_fc = mean_activity_fc
        last_EMA_fc = EMA_fc
    else:
        EMA_fc = mean_activity_fc * k + last_EMA_fc * (1-k)
        last_EMA_fc = EMA_fc

    # Aroon Up
    max_position = list(activity_fc).index(max(activity_fc))
    aroonUp_fc = len(observation_window) - max_position

    # DMI
    DMI_fc = 0
    if last_max_fc != 0:
        UpMove = max_activity_fc - last_max_fc
        if UpMove > 0:
            DMI_fc = UpMove

    last_max_fc = max_activity_fc

    ######################################################################################
    ################################# Activity over time #################################
    ######################################################################################
    
    # Noise activates 2 frequencies, so it will be considered a moment of activity if an instant of time has more than two active frequencies
    function = lambda x: 1 if x > 2 else 0
    vfunc = np.vectorize(function)
    activity = vfunc(activity.sum(axis=1))
    silence = 1 - activity
    # print("Silence:", silence)


    activity = group_array(activity)
    # print(activity)
    silence = group_array(silence)
    # print(silence)

    # Mean Activity
    mean_activity_t = np.mean(activity) 
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

    target = np.max(observation_window[:, -1])


    return [ mean_activity, median_activity, std_activity, max_activity, min_activity,
            percentil75_activity, percentil90_activity, percentil95_activity, percentil99_activity,
            mean_silence, median_silence, std_silence, max_silence, min_silence,
            percentil75_silence, percentil90_silence, percentil95_silence, percentil99_silence,
            DFLP, EMA_f, DMI_f, aroonUp_f,
            mean_activity_fc, median_activity_fc, std_activity_fc, max_activity_fc, min_activity_fc,
            percentil75_activity_fc, percentil90_activity_fc, percentil95_activity_fc, percentil99_activity_fc,
            EMA_fc, DMI_fc, aroonUp_fc,
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
                # if ('anomaly' in file and line[-1] == 1) or ('normal' in file):
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
                    data = window_analysis(np.array(observation_window), threshold, window_number)
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

    # update layout with buttons, and show the figure
    layout=go.Layout(updatemenus=list([dict(buttons=updateMenus)]),
                    xaxis_title="Activity (seconds)",
                    yaxis_title="Silence (seconds)",
                    width = 800,
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