#!/usr/bin/env python3
import math
import sys
import os
import math
import struct
import subprocess
import threading
import argparse
import time
import numpy as np
from sys import builtin_module_names
from datetime import datetime

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
    parser.add_argument('-fi', '--filter', help='auxiliar file to filter the main file')
    args = parser.parse_args()

    if args.filter:
        filter = open(args.filter,'r')
        one_char = filter.read(1)
        filter.seek(0)
        if one_char:
            a = filter.readline()
            size_of_line = len(a)
        

    f = args.file.split("/")[-1]
    fb=open(args.file,'rb')
    f = f.replace("bin","dat")
    w = open("data/"+f, 'wb')

    blen=fb.read(36)
    data=struct.unpack('=IQQQQ',blen)
    nPow=int((data[0]-16)/4)
    Rlen=data[0]+4+16

    first_time=last_time=data[1]+data[2]/1e6

    fStep=(data[4]-data[3])/1e6/nPow

    print(nPow,Rlen,first_time,fStep)

    fb.seek(0)

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
        
            
    nf=(int)(((f2-f1)/fStep))
    frq=np.arange(f1,f2,fStep)
    print(f1,f2)
    ####

    print("Start! {}, {}".format(Rlen,nPow))
    avgpow=np.array([])
    
    fb.seek(lines*Rlen)
    try:
        nPow=5
        n=0
        nt=500
        t=np.arange(nt)
        S=-90*np.ones((nt,nf))
        sfA=-90*np.ones((1,nf))
        ignored = False
    
        while True:
            rec=fb.read(Rlen)
            
            if(len(rec)<Rlen): 
                break

            data=struct.unpack('=IQQQQ{}f'.format(nPow),rec)
            
            # We want get only the data lines who belongs to between begin and end timestamps who're registered in the filter file
            # if the filter flag is True
            if args.filter:
                # Check if file is empty by reading first character in it
                if one_char:          
                    while True:
                        # get the timestamp from the main file
                        et = time.mktime(datetime.fromtimestamp(data[1]).timetuple())

                        if len(a) < size_of_line:
                            ignored=True
                            break
                        # get the timestamps from the filter file   
                        epoch_time = a.replace('\n','').split(" ")
                        
                        # convert them to datatime types
                        epoch_time_begin = time.mktime(datetime.strptime(epoch_time[0]+" "+epoch_time[1], '%Y-%m-%d %H:%M:%S.%f').timetuple())
                        epoch_time_end = time.mktime(datetime.strptime(epoch_time[2]+" "+epoch_time[3], '%Y-%m-%d %H:%M:%S.%f').timetuple())
                        
                        if et < epoch_time_begin:
                            #print("IGNORADO |","Et: ", et, "Begin: ",epoch_time_begin,"End: ", epoch_time_end)
                            ignored = True
                            break
                        
                        # if the et timestamp is greater than epoch_time_end 
                        elif et > epoch_time_end:
                            #print("FORA DO INTERVALO |","Et: ", et, "Begin: ",datetime.strptime(epoch_time[0]+" "+epoch_time[1], '%Y-%m-%d %H:%M:%S.%f'), epoch_time_begin,"End: ",datetime.strptime(epoch_time[2]+" "+epoch_time[3], '%Y-%m-%d %H:%M:%S.%f'), epoch_time_end)                                            
                            a = filter.readline()                        

                        elif (et > epoch_time_begin and et < epoch_time_end) or (et == epoch_time_begin) or (et == epoch_time_end):
                            #print("PERTENCE AO INTERVALO |","Et: ",et, "Begin: ", epoch_time_begin, "End: ",epoch_time_end) 
                            break
  
            if ignored:
                ignored = False
                continue
            else:
                times=data[1]+data[2]/1e6
                
                if times>last_time:
                    final = [n, times]                    
                    for e in sfA.tolist():
                        # Some measurements fail, which causes some values to come with the inf value
                        if any(np.isinf(e)):
                            e[:] = [x if not np.isinf(x) else -90 for x in e]
                        final += e

                    if len(final) == 82:
                        w.write(struct.pack("=82d",*final)) 
                    avgpow=np.append(avgpow,np.mean(sfA))
                    S=np.vstack((sfA,S))[:nt,:]  
                    sfA=np.zeros((1,nf))
                    n+=1
                    last_time=times

                fidx=((int)(data[3]/1e6)-f1)
                sfA[0,fidx:fidx+nPow]=data[5:]
                sfA[0,fidx:fidx+nPow]=np.maximum(sfA[0,fidx:fidx+nPow],data[5:])            
    finally:
        fb.close()
        if args.filter:
            filter.close()
        w.close()

if __name__ == '__main__':
	main()