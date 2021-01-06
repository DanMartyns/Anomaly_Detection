import os
import sys
import struct
import time
import argparse
import dateutil.relativedelta
from datetime import datetime
import numpy as np
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as plx
import functools 

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def main() :
    
    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f'  , '--file', help='Filename', required=True)
    parser.add_argument('-s'  , '--seconds', help='group in how much seconds', required=True)
    parser.add_argument('-pl'  , '--plot', help='plot aggregation result', action='store_true')
    parser.add_argument('-p'  , '--port', help='port')
    args = parser.parse_args()

    # open the file
    fb=open(args.file,'rb')

    w=open("data/32_15/"+args.file.split("/")[-1],'wb')

    start = time.time()

    maximuns = []
    mean = []
    
    try:
        n_sample = 0
        sampling_window = 0
        sample_sec = np.empty(shape=(0,85))

        while True:
            # read each line of the file
            record=fb.read(696)

            # break when the line was empty
            if(len(record) < 696): break

            # unpack the line
            line = np.array(list(struct.unpack('=87d',record)))

            
            if n_sample <= int(args.seconds)*81:
                sample_sec = np.vstack((sample_sec, line[2:]))
                n_sample += 1
            else:
                target = max(sample_sec[:,-1])
                s = np.mean(sample_sec[:,:-1], axis=0)
                result = np.concatenate(([ sampling_window, line[1]], s))
                result = np.concatenate((result, [target]))
                w.write(result)

                # See the results
                mean.append(np.mean(s))
                maximuns.append(max(s))

                # Reset Variables
                sample_sec = np.empty((0,85))
                sampling_window += 1
                n_sample = 0


        mean_maximuns = np.mean(maximuns[1:])
        picos = [x for x in maximuns if x >= 0.99*mean_maximuns ] # all the values higher than the mean (k)
        thr = np.mean(picos) # mean of the higher values
        # print("Threshold:",thr)

        if args.plot:
            fig = px.scatter(x=np.arange(0,len(mean[1:])), y=mean[1:], width=1200, height=800)     
            # fig.update_yaxes(range=[-80,-10])
            fig.update_layout(
                title="Spectrum activity ("+args.seconds+" second aggregation) ",
                xaxis_title="Time (seconds)",
                yaxis_title="RSSI (dBM)"
            )
            app.layout = html.Div(children=[

                dcc.Graph(
                    id='example-graph',
                    figure=fig
                )
            ])

            if __name__ == '__main__':
                app.run_server(debug=False, port=args.port)

        print("It took", time.time() - start, "seconds.") 

    except KeyboardInterrupt:
        print(">>> Interrupt received, stopping...")
    except Exception as e:
        print(e)
    finally:
        fb.close()

if __name__ == '__main__':
	main()
