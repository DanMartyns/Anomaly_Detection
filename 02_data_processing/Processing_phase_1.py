
# import the necessary packets
import os
import sys
import struct
import time
import argparse
from datetime import datetime
from functools import reduce
from bitarray import bitarray
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as colors

sample_number = 0
fn=None
timeSize = 0
tmp = []
beginings = []
timestamp = None
timestampBegin = None
timestampEnd = None
S = None
auxiliar = None

def to_binary(value, threshold):
    return 1 if value > threshold else 0

def nr_freq_with_activity(value, window_size):
    return 1 if value < window_size else 0

def waitforEnter(fstop=True):
    if fstop:
        if sys.version_info[0] == 2:
            input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")

# the size of a observation window (number of samples)
window_size = 0

def sampling_window(sample_window, threshold):

    global sample_number 
    sample_number += 1

    global beginings
    global timestamp
    global timestampBegin
    global timestampEnd
    global tmp

    if auxiliar:
        if timestamp != '':                        
            if datetime.fromtimestamp(sample_window[0][1]) >= datetime.strptime(timestampBegin,'%Y-%m-%d %H:%M:%S.%f') and datetime.fromtimestamp(sample_window[0][1]) <= datetime.strptime(timestampEnd.replace('\n',''),'%Y-%m-%d %H:%M:%S.%f') :
                tmp.append(sample_number)                        
            elif datetime.fromtimestamp(sample_window[0][1]) > datetime.strptime(timestampEnd.replace('\n',''),'%Y-%m-%d %H:%M:%S.%f'):
                timestamp = fn.readline()
                if len(timestamp) >= timeSize:
                    timestampBegin = timestamp.split(" ")[0]+" "+timestamp.split(" ")[1]
                    timestampEnd = timestamp.split(" ")[2]+" "+timestamp.split(" ")[3]
                print(tmp[0], tmp[-1])
                beginings.append(tmp[0])
                beginings.append(tmp[-1])
                tmp = []

    matrix = []
    
    # Convert all power signals of each frequency in 0/1 depending of a threshold to see if exists activity
    to_binary_vec = np.vectorize(to_binary)
    for array in sample_window:
        a = bitarray()
        a.extend(to_binary_vec(array[2:], threshold) )
        matrix.extend( [a] )

    # Reduce matrix of N elements to a single element, data from bitwise OR between elements
    sample = reduce(lambda x, y: x | y, matrix) 
    #print(sample)
    
    total_freq_active = sum(sample)

    # The number of active frequencies for each instant
    positions_of_each_active_freq = [ index for index, value in enumerate(sample) if value == 1]
    
    # Descriptive rules of the sample
    widthS = 0
    difusion_level = 0
    FrequenciesInARow = 0
    
    if len(positions_of_each_active_freq) >= 2:
        
        nr_zeros = 0
        temp = []
        
        # Calculation of the distance between the 1s, between the first and the last appearance of a 1.
        for value in sample:
            if value == 0:
                nr_zeros += 1
            else:
                if nr_zeros > 0:
                    temp.append(nr_zeros)
                    nr_zeros = 0

        if temp != []:
            # Average distance between 1s
            difusion_level = np.mean(temp)
        
        # width is the difference between the highest and lowest value in a data set.
        widthS = positions_of_each_active_freq[-1] - positions_of_each_active_freq[0]

    if len(positions_of_each_active_freq) > 0:
        nr_ones = 0
        temp = []

        # Calculation of the distance between the 1s, between the first and the last appearance of a 1.
        for value in sample:
            if value == 1:
                nr_ones += 1
            else:
                if nr_ones > 0:
                    temp.append(nr_ones)
                    nr_ones = 0

        if temp != []:
            FrequenciesInARow = np.mean(temp)

    # [0] - time    
    # [1] - mean of consecutive zeros 
    # [2] - mean of consecutive ones
    # [3] - width ( diference between the last active position and the first active position ) 
    # [4] - active frequencies total
    # [5] - activity/silence (1/0)
    # [time, consec_zeros, consec_ones, widthS, total_freq_active, activity/silence ]

    if widthS == 0:
        return [sample_window[0][1], difusion_level, 0, FrequenciesInARow, total_freq_active, 0]
    else:
        return [sample_window[0][1], difusion_level, FrequenciesInARow, widthS, total_freq_active, 1]

def main() :
    
    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f'  , '--file', help='Filename', required=True)
    parser.add_argument('-a'  , '--auxiliar', help='Auxiliar File')
    parser.add_argument('-t'  , '--threshold', type=int, help='the limit to consider a frequency active or not .', required=True)
    parser.add_argument('-p'  , '--plot', action='store_true')
    args = parser.parse_args()

    # open the file
    fb=open(args.file,'rb')

    # open the file to write the datas
    out=open("data/"+args.file.split('/')[-1].replace('.dat','_phase_1.dat'),'wb')

    if args.auxiliar:
        global auxiliar
        auxiliar = args.auxiliar

        global fn
        fn=open(args.auxiliar,'r')

        global timestamp
        timestamp = fn.readline()
        
        global timestampBegin
        timestampBegin = timestamp.split(" ")[0]+" "+timestamp.split(" ")[1]
        
        global timestampEnd
        timestampEnd = timestamp.split(" ")[2]+" "+timestamp.split(" ")[3]
        
        global timeSize
        timeSize = len(timestamp)

    try:
        
        # Metrics for each sample returned from sampling_window function
        data = []

        # variable to count until 5, to ignore the first 5 records
        i = 0
        while True:
            # read each line of the file
            record=fb.read(816)

            # break when the line was empty
            if(len(record) < 816): break

            # unpack the line
            line = list(struct.unpack('=102d',record))  

            if i > 5:
                s = sampling_window([line], args.threshold)
            
                data.append(s)

            #out.write(struct.pack("5d",*s))    
            i+=1 
        if args.auxiliar:
            print("Beginnings: ", beginings)

        if args.plot:        
            # x0 = []
            # x1 = []
            # x2 = []
            # for i in range(13):
            #     x0.append(stats.mode( list(filter(lambda value: value not in x0 , [x[0] for x in data ] )) )[0])
            #     x1.append(stats.mode( list(filter(lambda value: value not in x1 , [x[1] for x in data ] )) )[0])
            #     x2.append(stats.mode( list(filter(lambda value: value not in x2 , [x[2] for x in data ] )) )[0])
            
            # print(np.min([x[0] for x in data]), [value[0] for value in x0], max([x[0] for x in data]))
            # print(np.min([x[1] for x in data]), [value[0] for value in x1], max([x[1] for x in data]))
            # print(np.min([x[2] for x in data]), [value[0] for value in x2], max([x[2] for x in data]))

            # [time, consec_zeros, consec_ones, widthS, total_freq_active, activity/silence ]

            ax = plt.subplot(321)
            ax.title.set_text("Média de Frequências inativas consecutivas")
            ax.plot([x[1] for x in data ], color='b')
            ymin = np.min([x[1] for x in data ])
            ymax = np.max([x[1] for x in data ])
            ax.set_ylim([ymin-1,ymax+1])

            cx = plt.subplot(322)
            cx.title.set_text("Média de Frequências ativas consecutivas") 
            cx.plot([x[2] for x in data ], color='b')
            ymin = np.min([x[2] for x in data ])
            ymax = np.max([x[2] for x in data ])
            cx.set_ylim([ymin-1,ymax+1])

            bx = plt.subplot(323)
            bx.title.set_text("Diferença entre a posição da última Freq. Ativa e a 1ª")
            bx.plot([x[3] for x in data ], color='b')

            dx = plt.subplot(324)
            dx.title.set_text("Nr de Frequencias Ativas") 
            dx.plot([x[4] for x in data ], color='b')
            ymin = np.min([x[4] for x in data ])
            ymax = np.max([x[4] for x in data ])
            dx.set_ylim([ymin-1,ymax+1])

            dx = plt.subplot(325)
            dx.title.set_text("Actividade/Silence") 
            dx.plot([x[5] for x in data ], color='b')
            ymin = np.min([x[5] for x in data ])
            ymax = np.max([x[5] for x in data ])
            dx.set_ylim([ymin-1,ymax+1])

            if args.auxiliar:
                for x in range(len(beginings)):
                    ax.axvline(x=beginings[x], color='r')
                    bx.axvline(x=beginings[x], color='r')
                    cx.axvline(x=beginings[x], color='r')
                    dx.axvline(x=beginings[x], color='r')       
                    
            plt.show()

    except KeyboardInterrupt:
        print(">>> Interrupt received, stopping...")
    except Exception as e:
        print(e)
    finally:
        fb.close()
        if args.auxiliar:
            fn.close()
        out.close()

if __name__ == '__main__':
	main()
