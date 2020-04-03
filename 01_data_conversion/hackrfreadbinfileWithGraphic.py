#!/usr/bin/env python3
import math
import sys
import os
import struct
import subprocess
import threading
import argparse
import time
import numpy as np
from sys import builtin_module_names
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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
    args = parser.parse_args()

    plt.ion()
    plt.show()
    fig=plt.figure()
    fb=open(args.file,'rb')

    blen=fb.read(36)
    data=struct.unpack('=IQQQQ',blen)
    nPow=int((data[0]-16)/4)
    Rlen=data[0]+4+16

    first_time=data[1]+data[2]/1e6

    fStep=(data[4]-data[3])/1e6/nPow


    print(nPow,Rlen,first_time,fStep)

    fb.seek(0)

    ####
    while True:
            rec=fb.read(Rlen)
            if(len(rec)<Rlen): break

            data=struct.unpack('=IQQQQ{}f'.format(nPow),rec)
                
            time=data[1]+data[2]/1e6
            
            if time>first_time:
                f1=(int)(data[3]/1e6)
                f2=(int)(lastf/1e6)
                break
            
            lastf=data[4]
            
    nf=(int)((f2-f1)/fStep)
    frq=np.arange(f1,f2,fStep)
    print(f1,f2)
    ####

    print("Start! {}, {}".format(Rlen,nPow))

    avgpow=np.array([])
    

    try:
        last_time=0
        nPow=5
        n=0
        nt=500
        t=np.arange(nt)
        S=-90*np.ones((nt,nf))
        sfA=-90*np.ones((1,nf))

        while True:
            rec=fb.read(Rlen)
            if(len(rec)<Rlen): 
                break

            data=struct.unpack('=IQQQQ{}f'.format(nPow),rec)
            
            time=data[1]+data[2]/1e6
            fidx=((int)(data[3]/1e6)-f1)

            sfA[0,fidx:fidx+nPow]=data[5:]
            sfA[0,fidx:fidx+nPow]=np.maximum(sfA[0,fidx:fidx+nPow],data[5:])
            
            if time>last_time:
                avgpow=np.append(avgpow,np.mean(sfA))
                S=np.vstack((sfA,S))[:nt,:]  
                sfA=np.zeros((1,nf))
                n+=1
                last_time=time
                
            if n%50==0:
                plt.pcolormesh(frq,t,S,norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=-90, vmax=-50))
                plt.draw()
                fig.canvas.flush_events()
                plt.clf()
                
        plt.figure(2)
        plt.plot(avgpow)   
        waitforEnter()

    except KeyboardInterrupt:
        print(">>> Interrupt received, stopping...")
    finally:
        fb.close()

if __name__ == '__main__':
	main()