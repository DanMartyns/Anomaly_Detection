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
        
        if len(ob_window) == 0: return []
        else:
            activity = ob_window
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

            # Remove isolate events and group aglomerations
            if 'anomaly' in file and len(activity) > 2:
                tmp = copy.deepcopy(activity)
                for i in np.arange(0, len(activity)-10):
                    if activity[i] == 1 and activity[i+10] == 1:
                        for x in np.arange(1,10):
                            tmp[i+x] = 1
                activity = tmp
                for i in np.arange(0, len(activity)-2):
                    if activity[i] == 0 and activity[i+2]==0:
                        tmp[i+1] = 0

            activity = group_array(activity)
            silence = group_array(silence)
            
            activity = np.array([x for x in activity if x != 1 and x!=2])

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

    except Exception as e:
        print(e)

def main() :
    
    # input arguments
    parser = argparse.ArgumentParser()
    # Read from an directory
    parser.add_argument('-f'  , '--file', help='Filename', required=True)
    parser.add_argument('-t'  , '--threshold', help='the limit to consider a frequency active or not .', required=True)
    parser.add_argument('-ws', '--windowSize', help='the size of each observation window (seconds).', default=600)
    parser.add_argument('-wo', '--windowOffset', help='(seconds).', default=5)
    args = parser.parse_args()

    file, window_size, window_offset, threshold = args.file, int(args.windowSize), int(args.windowOffset), float(args.threshold)
    
    # open the file
    fb=open(file,'rb')

    out=open("02_data_processing/data/"+args.file.split('/')[-1],'wb')
    
    # initialization of the observation window
    observation_window = []

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
            if(len(record) < 696): 
                data = window_analysis(np.array(observation_window), threshold, window_number, file)
                if data != []:
                    data = [round(value,3) for value in data]
                    out.write(struct.pack("=50d",*data))               
                break

            # unpack the line
            line = list(struct.unpack('=87d',record))
            
            nr_samples += 1
            if line[-1] == 1:
                nr_anomalies += 1
            
            # The observation window will have de size of window size
            if sample < window_size:
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
                    data = window_analysis(np.array(observation_window), threshold, window_number, file)
                    if data != []:
                        data = [round(value,3) for value in data]
                        out.write(struct.pack("=50d",*data))
                               
                # reset the observation window
                observation_window = []
        print("Number of Anomalies:", nr_anomalies)
        print("Number of Samples:", nr_samples)
        print("Number of Windows:", window_number)
    except KeyboardInterrupt:
        print(">>> Interrupt received, stopping...")
    except Exception as e:
        print(e)
    finally:
        fb.close()
        out.close()

if __name__ == '__main__':
	main()