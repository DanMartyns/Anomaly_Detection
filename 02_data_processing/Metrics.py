'''
Channel	F0 (MHz)	Frequency    North       Japan      Most of
                    Range (MHz)  America                world	
1	    2412	    2401–2423	    Yes	      Yes	    Yes
2	    2417	    2406–2428	    Yes	      Yes	    Yes
3	    2422	    2411–2433	    Yes	      Yes	    Yes
4	    2427	    2416–2438	    Yes	      Yes	    Yes
5	    2432	    2421–2443	    Yes	      Yes	    Yes
6	    2437	    2426–2448	    Yes	      Yes	    Yes
7	    2442	    2431–2453	    Yes	      Yes	    Yes
8	    2447	    2436–2458	    Yes	      Yes	    Yes
9	    2452	    2441–2463	    Yes	      Yes	    Yes
10	    2457	    2446–2468	    Yes       Yes	    Yes
11	    2462	    2451–2473	    Yes       Yes	    Yes
12	    2467	    2456–2478	    No	      Yes	    Yes
13	    2472	    2461–2483	    No     	  Yes	    Yes
14	    2484	    2473–2495	    No	    11b Only    No
'''
# Importing the statistics module 
import statistics
import datetime as dt

class Metrics(object):
    
    def calculate(self, method_name, argument):
        """Dispatch method"""
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid metric")(argument)
        # Call the method as we return it
        return method

    '''
        Input 
            - instance : list of arrays with 10 elements
        Return
            - array of channels. Each channel is an array. Each channel has a list of records 
    '''  
    def split_channels(self, sample):
        # initialize channels array
        channels = []
        
        # for each channel
        for c in range(11):
            # take the instance and divide it into channels
            channels.append([p for index, p in enumerate(sample) if index >= ( 0 + 5*c ) and index <= ( 22 + 5*c ) ])
        
        #channels.append([p for index, p in enumerate(sample) if index >= 72 and index <= 94 ])
        
        # Print the channels splitted
        # for c in range(14):
        #     print("Channel "+str(c+1))
        #     for index, i in enumerate(channels[c]):
        #         if c+1 < 14:
        #             print(2401+5*c+index,i)
        #         if c+1 == 14:
        #             print(2473+index, i)
        
        # return the channels divided
        return channels

    '''
        Input 
            - channel : list of arrays with 10 elements
        Return
            - mean of the power signals
    '''        
    def weighted_average(self, channel):
        wei_avg = []
        weight = []
        for ind, i in enumerate(reversed(channel[:11])):    
            wei_avg.insert(0, channel[ind] * (0.8/(2**ind)))
            weight.append(0.8/(2**ind))
        
        wei_avg.append(channel[11])
        weight.append(1)

        for ind, i in enumerate(channel[12:]): 
            wei_avg.append(channel[ind] * (0.8/(2**ind)))
            weight.append(0.8/(2**ind))

        # return the mean of the data
        return sum(wei_avg)/sum(weight)



