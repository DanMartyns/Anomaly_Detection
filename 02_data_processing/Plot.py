import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class Plot(object):
    
    def plot(self, method_name, argument):
        """Dispatch method"""
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid metric")(argument)
        # Call the method as we return it
        return method

    '''
        Input
            - channel : list of arrays with 10 elements
    '''
    def plot_freq(self,channels):
        
        x = np.arange(14)
        for c in channels:
            plt.plot(x, c)
        
        plt.show()

# use ggplot style for more sophisticated visuals
plt.style.use('ggplot')

def realtime_plot(x_vec,y1_data,line1,identifier='',pause_time=0.1):
    
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec,y1_data,'-o',alpha=0.8)        
        #update plot label/title
        plt.ylabel('Power Signal (MHz)')
        plt.xlabel('Time (miliseconds)')
        plt.title('{}'.format(identifier))
        plt.show()
    
    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)

    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    
    # return line so we can update it again in the next iteration
    return line1