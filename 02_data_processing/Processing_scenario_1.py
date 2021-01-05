
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
    mean_silence = np.mean(silence)
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
    max_position = list(sum_by_lines_a).index(max(sum_by_lines_a))
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
    parser.add_argument('-ws', '--windowSize', type=int, help='the size of each observation window ( number of samples ).', default=10)
    parser.add_argument('-wo', '--windowOffset', type=int, help='(number of samples).', default=2)
    parser.add_argument('-p'  , '--plot', action='store_true')
    args = parser.parse_args()

    # open the file
    fb=open(args.file,'rb')

    # open the file to write the datas
    out=open("02_data_processing/data/"+args.file.split('/')[-1],'wb')
    # out=open("data/"+args.file.split('/')[-1].replace('.dat','_scenario_1.dat'),'wb')
    
    global window_size
    # The exact number of samples for each Observation Window
    window_size = args.windowSize

    # The number of offset samples
    window_offset = args.windowOffset

    # initialization of the observation window
    observation_window = []

    plot = []

    start = time.time()

    #sample_1sec = []

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
                # if len(sample_1sec) < 81:
                #     sample_1sec.append(line)
                # else:
                # append the sample to the window
                observation_window.append(line)
                #sample_1sec = []
            
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
                    out.write(struct.pack("53d",*data))
                    plot.append(data)
                
                # reset the observation window
                observation_window = []

        print("It took", time.time() - start, "seconds.")

        if args.plot:

            timeSet = np.arange(0,len([x[43] for x in plot ]))
            data = {'obs':timeSet,'mean_silence':[x[43] for x in plot ]}
            df = pd.DataFrame(data=data)
            fig = plx.scatter(df, x="obs", y="mean_silence", labels={"obs" : "Observation Window", "mean_silence" : "Mean Silence"})
            #fig.update_yaxes(range=[-70,-40], dtick=10, row=1, col=1)
            fig.update_layout(
                    font_family="Courier New",
                    font_size=15,
                    title={
                        'text': "Anomaly",
                        'y':0.95,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'}
            )

            app.layout = html.Div(children=[

                dcc.Graph(
                    id='example-graph',
                    figure=fig
                )
            ])

            if __name__ == '__main__':
                app.run_server(debug=False, port=8051)

            # plt.figure(1, figsize=(10,8))
            # plt.subplots_adjust(hspace = 0.5)

            # ax = plt.subplot(4,4,1)
            # ax.title.set_text("Média de frequências ativas")
            # ax.plot([x[0] for x in plot ], color='b')
            
            # bx = plt.subplot(4,4,2)
            # bx.title.set_text("Mediana de frequências ativas") 
            # bx.plot([x[1] for x in plot ], color='b')

            # cx = plt.subplot(4,4,3)
            # cx.title.set_text("Std de frequências ativas")
            # cx.plot([x[2] for x in plot ], color='b')
            
            # dx = plt.subplot(4,4,4)
            # dx.title.set_text("Max de frequências ativas") 
            # dx.plot([x[3] for x in plot ], color='b')
            
            # ex = plt.subplot(4,4,5)
            # ex.title.set_text("Min de frequências ativas") 
            # ex.plot([x[4] for x in plot ], color='b')
            
            # fx = plt.subplot(4,4,6)
            # fx.title.set_text("Percentil 75 de frequências ativas") 
            # fx.plot([x[5] for x in plot ], color='b')
            
            # gx = plt.subplot(4,4,7)
            # gx.title.set_text("Percentil 90 de frequências ativas") 
            # gx.plot([x[6] for x in plot ], color='b')
            
            # hx = plt.subplot(4,4,8)
            # hx.title.set_text("Percentil 95 de frequências ativas") 
            # hx.plot([x[7] for x in plot ], color='b')
            
            # ix = plt.subplot(4,4,9)
            # ix.title.set_text("Percentil 99 de frequências ativas") 
            # ix.plot([x[8] for x in plot ], color='b')
            
            # jx = plt.subplot(4,4,10)
            # jx.title.set_text("Média de frequências inativas")
            # jx.plot([x[9] for x in plot ], color='b')
            
            # kx = plt.subplot(4,4,11)
            # kx.title.set_text("Mediana de frequências inativas") 
            # kx.plot([x[10] for x in plot ], color='b')

            # lx = plt.subplot(4,4,12)
            # lx.title.set_text("Std de frequências inativas")
            # lx.plot([x[11] for x in plot ], color='b')
            
            # mx = plt.subplot(4,4,13)
            # mx.title.set_text("Max de frequências inativas") 
            # mx.plot([x[12] for x in plot ], color='b')
            
            # nx = plt.subplot(4,4,14)
            # nx.title.set_text("Min de frequências inativas") 
            # nx.plot([x[13] for x in plot ], color='b')
            
            # ox = plt.subplot(4,4,15)
            # ox.title.set_text("Percentil 75 de frequências inativas") 
            # ox.plot([x[14] for x in plot ], color='b')
            
            # px = plt.subplot(4,4,16)
            # px.title.set_text("Percentil 90 de frequências inativas") 
            # px.plot([x[15] for x in plot ], color='b')

            # plt.show()
            # plt.figure(2, figsize=(10,8))
            # plt.subplots_adjust(hspace = 0.5)
            
            # qx = plt.subplot(4,4,1)
            # qx.title.set_text("Percentil 95 de frequências inativas") 
            # qx.plot([x[16] for x in plot ], color='b')

            # rx = plt.subplot(4,4,2)
            # rx.title.set_text("Percentil 99 de frequências inativas") 
            # rx.plot([x[17] for x in plot ], color='b')

            # sx = plt.subplot(4,4,3)
            # sx.title.set_text("Diferença entre a 1ª e a última posição ativa") 
            # sx.plot([x[18] for x in plot ], color='b')
            
            # tx = plt.subplot(4,4,4)
            # tx.title.set_text("EMA de frequências") 
            # tx.plot([x[19] for x in plot ], color='b')

            # ux = plt.subplot(4,4,5)
            # ux.title.set_text("DMI de frequências") 
            # ux.plot([x[20] for x in plot ], color='b')

            # xx = plt.subplot(4,4,6)
            # xx.title.set_text("Aroon Up de frequências") 
            # xx.plot([x[21] for x in plot ], color='b')

            # zx = plt.subplot(4,4,7)
            # zx.title.set_text("Média de frequências consecutivas") 
            # zx.plot([x[22] for x in plot ], color='b')

            # aa = plt.subplot(4,4,8)
            # aa.title.set_text("Mediana de frequências consecutivas") 
            # aa.plot([x[23] for x in plot ], color='b')

            # ba = plt.subplot(4,4,9)
            # ba.title.set_text("Std de frequências consecutivas") 
            # ba.plot([x[24] for x in plot ], color='b')  

            # ca = plt.subplot(4,4,10)
            # ca.title.set_text("Max de frequências consecutivas") 
            # ca.plot([x[25] for x in plot ], color='b')      

            # da = plt.subplot(4,4,11)
            # da.title.set_text("Min de frequências consecutivas") 
            # da.plot([x[26] for x in plot ], color='b')    

            # ea = plt.subplot(4,4,12)
            # ea.title.set_text("Percentil 75 de frequências consecutivas") 
            # ea.plot([x[27] for x in plot ], color='b')

            # fa = plt.subplot(4,4,13)
            # fa.title.set_text("Percentil 90 de frequências consecutivas") 
            # fa.plot([x[28] for x in plot ], color='b')

            # ga = plt.subplot(4,4,14)
            # ga.title.set_text("Percentil 95 de frequências consecutivas") 
            # ga.plot([x[29] for x in plot ], color='b')

            # ha = plt.subplot(4,4,15)
            # ha.title.set_text("Percentil 99 de frequências consecutivas") 
            # ha.plot([x[30] for x in plot ], color='b')

            # plt.show()
            # plt.figure(3, figsize=(10,8))
            # plt.subplots_adjust(hspace = 0.5)

            # ia = plt.subplot(4,4,1)
            # ia.title.set_text("EMA de frequências consecutivas") 
            # ia.plot([x[31] for x in plot ], color='b')

            # ja = plt.subplot(4,4,2)
            # ja.title.set_text("DMI de frequências consecutivas") 
            # ja.plot([x[32] for x in plot ], color='b')

            # ka = plt.subplot(4,4,3)
            # ka.title.set_text("Aroon Up de frequências consecutivas") 
            # ka.plot([x[33] for x in plot ], color='b')

            # la = plt.subplot(4,4,4)
            # la.title.set_text("Target") 
            # la.plot([x[34] for x in plot ], color='b')

            # plt.show()

            # global mean
            # bins = 20
            # df = pd.DataFrame(mean)
            # df.columns = ['RSSI (dBm)'] 
            # df['Time'] = np.arange(df['RSSI (dBm)'].count() )
            # df98 = calculaPercentil98(df)
            
            # # Gráfico de Dispersão
            # # Criando o ambiente do gráfico 
            # sns.set_style('white')
            # fig, tx = plt.subplots(1,1,figsize=(15,10))
            
            # # Insere curva KDE (Kernel Density Estimation)
            # # curva KDE (Kernel Density Estimation) que estima a probabilidade de um valor aleatório do conjunto de dados estar em cada intervalo de medição
            # g1 = sns.distplot(df98['RSSI (dBm)'], ax=tx, kde=True, hist=False)

            # # Insere histograma
            # # as barras verticais referem-se à quantidade de valores presentes em cada um dos intervalos. Foram definidos 20 intervalos através da variável bins, 
            # # a qual é passada como parâmetro na função distplot() da curva g2, o que cria 20 barras verticais.
            # tx_copy = tx.twinx()
            # g2 = sns.distplot(df98['RSSI (dBm)'], ax=tx_copy, kde=False, hist=True, bins=bins, norm_hist=False)

            # # Ajusta rótulos
            # g1.set_ylabel("Probabilidade")
            # g2.set_ylabel("Quantidade")
            # g1.xaxis.set_major_locator(ticker.MultipleLocator((df98['RSSI (dBm)'].max()-df98['RSSI (dBm)'].min())/bins))

            # plt.setp(tx.get_xticklabels(), rotation=45)
            # plt.show()

            # z = [ x for x in mean if x > args.threshold]
            # fig = plt.figure(figsize=(15,10))
            # plt.boxplot(z)
            # plt.show()
      
            

    except KeyboardInterrupt:
        print(">>> Interrupt received, stopping...")
    except Exception as e:
        print(e)
    finally:
        fb.close()
        out.close()

if __name__ == '__main__':
	main()
