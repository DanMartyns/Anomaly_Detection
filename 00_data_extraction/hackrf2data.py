#!/usr/bin/env python3
import math
import os
import struct
import subprocess
import threading
import time
import argparse
import numpy as np
from sys import builtin_module_names
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from datetime import datetime

f1=2401
f2=2473#2483

fstr="{}:{}".format(f1,f2)

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--wildcard', help='wildcard for the file')
parser.add_argument('-l', '--lna', help='RX LNA (IF) gain, 0-40dB, 8dB steps')
parser.add_argument('-g', '--vga', help='RX VGA (baseband) gain, 0-62dB, 2dB steps')
args = parser.parse_args()

label = args.wildcard 
l = args.lna
g = args.vga

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H_%M_%S")
bfilename = label+"_"+dt_string+".bin"

ON_POSIX = 'posix' in builtin_module_names
# my_hackrf_sweep -f 2412:2484 -B -l 8 -g 20 -r power1.bin

cmdpipe = subprocess.Popen(["./my_hackrf_sweep","-f {}".format(fstr),'-B',' -l '+l+' -g '+g, "-r {}".format(bfilename)],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    close_fds=ON_POSIX)


print("Start!")
start = time.time()

try:
    while True:
        rec=cmdpipe.stderr.readline(1024).decode('utf-8').strip()
        print(rec)
except KeyboardInterrupt:
    print(">>> Interrupt received, stopping...")
except TimeoutError:
    print(">>> Interrupt received (Timeout), stopping...")    
finally:
    cmdpipe.terminate()
    print("Finished in : ",time.time()-start," seconds")

