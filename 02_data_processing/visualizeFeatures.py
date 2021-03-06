
# import the necessary packets
import os
import sys
import struct
import time
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from functools import reduce
from bitarray import bitarray
from datetime import datetime
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as plx
from plotly.subplots import make_subplots
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def waitforEnter(fstop=True):
    if fstop:
        if sys.version_info[0] == 2:
            input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")


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
    added = False
    l = []
    for value in lista:
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
        return [round(x,3) for x in l]
    else:
        return [0]

# last EMA
last_EMA_f = 0
last_EMA_fc = 0
# last max
last_max_f = 0
last_max_fc = 0

# the size of a observation window (number of samples)
window_size = 0

def window_analysis(observation_window, threshold, window_number):
    global last_EMA_f
    global last_EMA_fc
    global last_max_f
    global last_max_fc
    
    ######################################################################################
    ################## Apply the threshold and transcribe to 0s and 1s ###################
    ######################################################################################
    mean_samples = []
    for i in observation_window[:,2:-1]:
        mean_samples.append(np.mean(i))
    
    function = lambda x, threshold: 1 if x >= threshold else 0
    vfunc = np.vectorize(function)
    
    observation_window[:,2:-1] = vfunc(observation_window[:,2:-1], threshold)
    mean_samples = vfunc(mean_samples, threshold)

    ######################################################################################
    ################################# Frequency/time #####################################
    ######################################################################################

    activity = observation_window[:,2:-1]
    silence = 1-activity

    ## Acitivity/Silence Features

    # Sum by lines
    sum_by_lines_a = activity.sum(axis=1)

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
    
    # print("Activity:",mean_samples)
    activity = np.array(mean_samples)
    silence = 1 - activity
    # print("Silence:",silence)
    
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

    #print(silence)

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

    target = 0
    # if all samples are anomalous
    if all(b == 1 for b in observation_window[:,-1]):
        target = 1
    # if all samples are normal
    elif all(b == 0 for b in observation_window[:,-1]):
        target = 0
    else:
        target = 1


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
            
def main() :
    
    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f'  , '--file', help='Filename', required=True)
    parser.add_argument('-t'  , '--threshold', type=float, help='the limit to consider a frequency active or not .', required=True)
    parser.add_argument('-ws', '--windowSize', type=int, help='the size of each observation window ( number of samples ).', default=600)
    parser.add_argument('-wo', '--windowOffset', type=int, help='(number of samples).', default=5)
    parser.add_argument('-p'  , '--plot', action='store_true')
    args = parser.parse_args()

    # open the file
    fb=open(args.file,'rb')

    # open the file to write the datas
    out=open("02_data_processing/data/"+args.file.split('/')[-1],'wb')
  
    global window_size
    # The exact number of samples for each Observation Window
    window_size = args.windowSize

    # The number of offset samples
    window_offset = args.windowOffset

    # initialization of the observation window
    observation_window = []

    plot = []

    start = time.time()

    try:
        
        # The number of the current Observation Window
        window_number = 1

        # The initialization of the number of samples
        sample = 0

        while True:
            # read each line of the file
            record=fb.read(696)

            # break when the line was empty
            if(len(record) < 696): break

            # unpack the line
            line = list(struct.unpack('=87d',record))

            # The observation window will have de size of window size
            if sample < window_size:
                #if ('anomaly' in args.file and line[-1] == 1) or ('normal' in args.file):    
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

                # analyse the window
                # remove the first 3 measurements
                if window_number >= 3:
                    data = window_analysis(np.array(observation_window),args.threshold, window_number)
                    data = [round(value,3) for value in data]
                    plot.append(data)
                    
                
                # reset the observation window
                observation_window = []
            
        print("It took", time.time() - start, "seconds.")

        if args.plot:

            # print(plot)

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

            # Inicialize variables
            updateMenus = []
            data = []

            # Add traces to each feature
            timeSet = np.arange(0,len([x[0] for x in plot ]))
            
            for feature_index, feature in enumerate(features):
                data.append(go.Scatter(x = timeSet, y = [ x[feature_index] for x in plot ], mode="markers", marker=dict(color="DarkOrange"), name=feature))
                updateMenus.append(
                    dict(
                        {
                            'label': feature,
                            'method': 'update',
                            'args': [{'visible': [False]*feature_index + [True] + [False]*(len(features) - feature_index) }, {'title': feature}]
                        }  
                    )
                )

            # update layout with buttons, and show the figure
            layout=go.Layout(title = 'Results of each feature',
                            updatemenus=list([dict(buttons=updateMenus)]))
            
            #defining figure and plotting
            fig = go.Figure(data,layout)
            fig.show()
            
            # fig.update_layout(showlegend=False)
            # fig.update_layout(legend_title_text='Features')
            # fig.update_layout(
            #         font_family="Courier New",
            #         font_size=14,
            #         title={
            #             'y':0.9,
            #             'x':0.5,
            #             'xanchor': 'center',
            #             'yanchor': 'top'},
            #         autosize = False,
            #         width= 1800,
            #         height= 500
            # )

    except KeyboardInterrupt:
        print(">>> Interrupt received, stopping...")
    except Exception as e:
        print(e)
    finally:
        fb.close()
        out.close()

if __name__ == '__main__':
	main()
