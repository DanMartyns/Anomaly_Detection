#!/usr/bin/env python3
import math
import sys
import os
import struct
import subprocess
import threading
import argparse
import time
import pandas as pd
import numpy as np
from bitarray import bitarray
from sys import builtin_module_names
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
import matplotlib.colors as colors
import dash
import plotly.express as px
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

########################################

# The file was chaged to group by instance number and block number
# The data was saved in a new file

########################################

def waitforEnter(fstop=True):
    if fstop:
        if sys.version_info[0] == 2:
            raw_input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")

def main():

    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='filename')
    parser.add_argument('-a'  , '--auxiliar', help='Auxiliar File')
    args = parser.parse_args()

    # plt.ion()
    # plt.show()
    # fig=plt.figure()

    fb=open(args.file,'rb')

    blen=fb.read(36)
    data=struct.unpack('=IQQQQ',blen)
    nPow=int((data[0]-16)/4)
    Rlen=data[0]+4+16

    first_time=data[1]+data[2]/1e6

    fStep=(data[4]-data[3])/1e6/nPow


    print(nPow,Rlen,first_time,fStep)

    fb.seek(0)

    if args.auxiliar:
        
        fn=open(args.auxiliar,'r')

        timestamp = fn.readline()
        timestampBegin = timestamp.split(" ")[0]+" "+timestamp.split(" ")[1]
        timestampEnd = timestamp.split(" ")[2]+" "+timestamp.split(" ")[3]
        
        timeSize = len(timestamp)

    i = 0
    lines = 0
    ####
    while True:
            rec=fb.read(Rlen)
            lines+=1
            if(len(rec)<Rlen): break

            data=struct.unpack('=IQQQQ{}f'.format(nPow),rec)
                
            times=data[1]+data[2]/1e6
            
            # For some reason in the extracted data, the sample was supposed to have 
            # 20 lines, 5 frequencies and 5 power signals each, but the first sample
            # don't have this pattern, so we get the min and max frequency from the 
            # fifth sample, to garente that the min and max frequency is the correct one         
            if times>first_time:
                if i==5:
                    f1=(int)(data[3]/1e6) 
                    f2=(int)(lastf/1e6)
                    break
                first_time=last_time=data[1]+data[2]/1e6
                i+=1
            lastf=data[4]

    nf=(int)((f2-f1)/fStep)
    frq=np.arange(f1,f2,fStep)
    frq=np.arange(f1,2483,fStep)
    print(f1,f2)
    ####

    print("Start! {}, {}".format(Rlen,nPow))

    avgpow=np.array([])
    maximuns=np.array([])
    
    fb.seek(lines*Rlen)
    try:
        last_time=0
        nPow=5
        n=0
        nt=500
        t=np.arange(nt)
        S=-90*np.ones((nt,nf))
        sfA=-90*np.ones((1,nf))
        sample_number = 0
        tmp = []
        beginings = []

        while True:
            rec=fb.read(Rlen)
            if(len(rec)<Rlen): 
                break

            data=struct.unpack('=IQQQQ{}f'.format(nPow),rec)
            
            times=data[1]+data[2]/1e6
                
            if times>last_time:
                for index,e in enumerate(sfA.tolist()):
                    # Some measurements fail, which causes some values to come with the inf value
                    if any(np.isinf(e)):
                        e[:] = [x if not np.isinf(x) else -90 for x in e]
                        if any(np.isnan(e)):
                            e[:] = [x if not np.isnan(x) else -90 for x in e]
                    sfA[index] = e
                sample_number +=1
                if args.auxiliar:
                    if timestamp != '':                        
                        if datetime.fromtimestamp(times) >= datetime.strptime(timestampBegin,'%Y-%m-%d %H:%M:%S.%f') and datetime.fromtimestamp(times) <= datetime.strptime(timestampEnd.replace('\n',''),'%Y-%m-%d %H:%M:%S.%f') :
                            tmp.append(sample_number)                     
                        elif datetime.fromtimestamp(times) > datetime.strptime(timestampEnd.replace('\n',''),'%Y-%m-%d %H:%M:%S.%f'):
                            timestamp = fn.readline()
                            if len(timestamp) >= timeSize:
                                timestampBegin = timestamp.split(" ")[0]+" "+timestamp.split(" ")[1]
                                timestampEnd = timestamp.split(" ")[2]+" "+timestamp.split(" ")[3]
                            beginings.append(tmp[0])
                            beginings.append(tmp[-1])
                            tmp = []
                            print("Timestamp: ",timestamp)
                            print("Line: ",datetime.fromtimestamp(times))
		        # Change between mean or max to see results
                avgpow = np.append(avgpow,np.mean(sfA))
                maximuns = np.append(avgpow,np.max(sfA))
                #sfA = np.where(sfA > -55, 1, 0) 
                S=np.vstack((sfA,S))[:nt,:]  
                sfA=np.zeros((1,nf))
                n+=1
                last_time=times
                
            fidx=((int)(data[3]/1e6)-f1)
            sfA[0,fidx:fidx+nPow]=data[5:]
            sfA[0,fidx:fidx+nPow]=np.maximum(sfA[0,fidx:fidx+nPow],data[5:])

        # print(beginings)
        max_seconds = len(avgpow)/81
        avgpow = avgpow[5:]
        maximuns = maximuns[5:]
        mean_maximuns = np.mean(maximuns)
        print("mean_maximuns:", mean_maximuns)
        picos = [x for x in maximuns if x >= 0.95*mean_maximuns ] # all the values higher than the mean (k)
        #print(picos)
        thr = np.mean(picos) # mean of the higher values
        print('Threshold: ', thr)
        # ymin = thr
        # #ymin = np.min(avgpow)
        # ymax = np.max(avgpow)
        # print("Minimo:",ymin)
        # print("Maximo:",ymax)
        # plt.figure(2, figsize=(10, 8), dpi= 80, facecolor='w', edgecolor='k')
        # ax = plt.axes()
        
        # ax.set_xlabel('Time (seconds)')	
        # ax.set_ylabel('RSSI (dBm)')

        timeSet = np.arange(0,max_seconds, max_seconds/len(avgpow))
        
        # plt.plot(timeSet, avgpow, 'o', label='Activity')
        
        # plt.xlim([0, int(np.max(timeSet))])
        if args.auxiliar:
            print("Len(Beginnings): ", len(beginings))
            for i in beginings:
                plt.axvline(x=i, color='g', linestyle='-')  
        #####horizontal line
        # #plt.axhline(y=thr, color='r', linestyle='--', label='Threshold')
        # plt.legend(loc="upper left")
        # plt.title("Clean")
        # #plt.ylim([ymin,ymax])
        # plt.ylim([-75,-50])
        # #plt.ylim([-65,-20])
        # plt.show()
        # waitforEnter()

        data = {'Time':timeSet,'RSSI':avgpow}
        df = pd.DataFrame(data=data)
        fig = px.scatter(df, x="Time", y="RSSI", labels={"Time" : "Time (seconds)"})
        fig.update_yaxes(range=[-70,-40], dtick=2, row=1, col=1)
        fig.update_layout(
                font_family="Courier New",
                font_size=15,
                title={
                    'text': "Clean data",
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
            app.run_server(debug=True, port=8051)


    except KeyboardInterrupt:
        print(">>> Interrupt received, stopping...")
    finally:
        fb.close()

if __name__ == '__main__':
	main()
