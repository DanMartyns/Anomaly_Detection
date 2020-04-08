
# import the necessary packets
import os
import struct
import argparse
import statistics
import datetime
import numpy as np
import dateutil.relativedelta
from Metrics import Metrics
from Plot import Plot
import matplotlib.pyplot as plt

def autolabel(rects, ax):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')

epoch = datetime.datetime.utcfromtimestamp(0)

def unix_time_millis(dt):
    return (dt - epoch).total_seconds() * 1000.0

# initialize variables
c = Metrics()
p = Plot()


# the size of a observation window (number of samples)
window_size = 0

# value to define when is active or not
threshold = -65

def window_analysis(observation_window):
    global window_size

    # Features to be written to the output file (window_size samples, 14 channels, 4 metrics)
    # outMeasuresForEachSample = np.zeros( (sample, channel,metric) )
    # [0] - mean for each sample
    # [1] - variance for each sample
    # [2] - silence times
    outMeasuresForEachSample = np.zeros( (window_size,14,3) )

    for i, sample in enumerate(observation_window):
        # Split each instance in the 14 Wi-Fi channels
        channels = c.calculate("split_channels", sample[2:])
        
        # Calculate the metricss for each channel for each sample
        for index, channel in enumerate(channels):
            outMeasuresForEachSample[i, index, 0] = c.calculate('weighted_average', channel)
            outMeasuresForEachSample[i, index, 1] = np.var(channel)
            outMeasuresForEachSample[i, index, 2] = len(channel) - np.count_nonzero([int(x) for x in list(map(lambda x: x > threshold, channel))])

        # For each channel
        #print("For each channel: ",outMeasuresForEachSample[i, :,:], sep='\n')

        # N = 14
        # ind = np.arange(N)  # the x locations for the groups
        # width = 0.27        # the width of the bars

        # fig = plt.figure()
        # ax = fig.add_subplot(111)

        # yvals = outMeasuresForEachSample[i,:,0]
        # rects1 = ax.bar(ind, yvals, width, color='r')
        # zvals = outMeasuresForEachSample[i,:,1]
        # rects2 = ax.bar(ind+width, zvals, width, color='g')
        # kvals = outMeasuresForEachSample[i,:,2]
        # rects3 = ax.bar(ind+2*width, kvals, width, color='b')

        # ax.set_ylabel('Scores')
        # ax.set_xticks(ind+width)
        # ax.set_xticklabels( ('Channel 1','Channel 2','Channel 3','Channel 4','Channel 5','Channel 6','Channel 7','Channel 8','Channel 9','Channel 10','Channel 11','Channel 12','Channel 13','Channel 14') )

        # autolabel(rects1, ax)
        # autolabel(rects2, ax)
        # autolabel(rects3, ax)

        # plt.show()

    return outMeasuresForEachSample
    
def main() :

    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='Filename')
    parser.add_argument('-ws', '--windowSize', type=int, help='the size of each observation window ( number of samples ).', default=10)
    parser.add_argument('-wo', '--windowOffset', type=int, help='(number of samples).', default=2)
    args = parser.parse_args()

    global window_size
    # The exact number of samples for each Observation Window
    window_size = args.windowSize
    
    # The number of offset samples
    window_offset = args.windowOffset

    # initialization of the observation window
    observation_window = []
        
    # open the file
    fb=open(args.file,'rb')

    # open the file to write the results
    out=open("data/"+args.file.split('/')[-1],'wb')
    
    try:

        # The number of the current Observation Window
        window_number = 1

        # The initialization of the number of samples
        sample = 0

        while True:

            # read each line of the file
            record=fb.read(816)

            # break when the line was empty
            if(len(record) < 816): break

            # unpack the line
            line = list(struct.unpack('=102d',record))
        
            if sample < window_size:
                # append the sample to the window
                observation_window.append(line)
                # increase the number of samples inside the window
                sample += 1
            else:
                # jump to the next position in the file
                fb.seek( 816*window_offset*window_number )
                # increase the number of windows
                window_number += 1
                # number of samples for each observation window
                sample = 0
                # analyse the window
                # to remove the first 4 measurements
                if window_number >= 5:
                    data = window_analysis(observation_window) 
                    
                    # Each observation window will have X samples (X == observation window size)
                    # Each data_slice is a sample
                    for data_slice in data:
                        np.save(out, data_slice)
                
                # reset the observation window
                observation_window = []              

    except KeyboardInterrupt:
        print(">>> Interrupt received, stopping...")
    finally:
        fb.close()
        out.close()


if __name__ == '__main__':
	main()