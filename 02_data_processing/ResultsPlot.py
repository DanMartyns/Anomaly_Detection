import sys
import struct
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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
    args = parser.parse_args()

    fb=open(args.file,'rb')
    plt.ion()
    plt.show()
    fig=plt.figure(figsize=(8,8))
    plt.subplots_adjust(hspace = .15)
    try:
        n=0
        nt=10000
        avgpow=-90*np.ones((nt,1))
        vari=-90*np.ones((nt,1))
        silences=90*np.ones((nt,1))
        while True:

            record=fb.read(24)
            if(len(record) < 24): break

            # unpack the line
            line = list(struct.unpack('=3d',record))

            avgpow=np.vstack((line[0],avgpow))[:nt,:]
            vari=np.vstack((line[1],avgpow))[:nt,:]
            silences=np.vstack((line[2],avgpow))[:nt,:]

            n+=1
            if n%200==0:
                
                plt.subplot(311)  
                plt.plot(avgpow, 'r')
                plt.ylim(-80, -55)
                plt.xlabel('Observation Window')
                plt.ylabel('Mean/Observation Window')
                
                plt.subplot(312)  
                plt.plot(vari,'g')
                plt.ylim(-80, -55)
                plt.xlabel('Observation Window')
                plt.ylabel('Var/Observation Window')                

                plt.subplot(313)  
                plt.plot(silences)
                plt.ylim(0, 100)
                plt.xlabel('Observation Window')
                plt.ylabel('Silences/Observation Window')

                plt.draw()
                fig.canvas.flush_events()
                plt.clf()
   
        waitforEnter()

    except KeyboardInterrupt:
        print(">>> Interrupt received, stopping...")
    except Exception as e:
        print(e)
    finally:
        fb.close()

if __name__ == '__main__':
	main()