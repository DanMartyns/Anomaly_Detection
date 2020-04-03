
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

    #print("Observation Window: ", *observation_window, sep='\n')
    #print('\n')

    # Features to be written to the output file (14 channels, 3 metrics)
    # [0] - mean for each sample
    # [1] - variance for each sample
    # [2] - standard deviation for each sample
    # [3] - activity time
    outMeasures = np.zeros( (14,4) )

    silences = np.zeros( (14, window_size) )

    # weight average for each channel
    weight_average = np.zeros((14,window_size))

    for i, sample in enumerate(observation_window):
        # Split each instance in the 14 Wi-Fi channels
        channels = c.calculate("split_channels", sample[2:])
        
        # Calculate the mean for each channel
        for index, channel in enumerate(channels):
            # weighted average for each channel (each channel was 10 weighted_average)
            # weighed_average[channel, sample] => result in 14 channels with 10 weighted_average
            #print("Channel {}".format(index+1), channel, sep='\n') 
            wa = c.calculate('weighted_average', channel)
            weight_average[index, i] = wa
            # Array of silence times for each channel
            silences[index, i] = 1 if wa > threshold else 0

    #print("Weight Average: ", weight_average)

    # mean for each sample using the weighted average of the channels
    outMeasures[:, 0] = weight_average.mean(axis=1)
    #print("Mean: ", outMeasures[:, 0], '\n')

    # variance for each sample using the weighted average of the channels
    outMeasures[:, 1] = weight_average.var(axis=1)
    #print("Variance: ", outMeasures[:, 1], '\n')

    # standard deviation for each sample using the weighted average of the channels
    outMeasures[:, 2] = weight_average.std(axis=1)
    #print("Standard Deviation: ", outMeasures[:, 2], '\n')

    # number of activities times for each channel
    outMeasures[:, 3] = [float(e) for e in np.count_nonzero(silences,axis=1)]
    #print("Silences: ", outMeasures[:, 3], '\n')

    return outMeasures
    
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
                    outMeasures = window_analysis(observation_window) 

                    #print(outMeasures, '\n')
                    #p.plot('plot_freq',weight_average)
                    for i in range(14):
                        out.write(struct.pack("=4d",*outMeasures[i]))
                
                # reset the observation window
                observation_window = []              

    except KeyboardInterrupt:
        print(">>> Interrupt received, stopping...")
    finally:
        fb.close()
        out.close()


if __name__ == '__main__':
	main()