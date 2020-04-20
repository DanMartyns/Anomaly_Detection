import bluetooth
import time
from datetime import datetime

bd_addr = "68:EC:C5:2C:A8:7A"
port = 1

sock=bluetooth.BluetoothSocket( bluetooth.RFCOMM )
sock.connect((bd_addr, port))

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H_%M_%S")
bfilename = "vazio_"+dt_string+".txt"

fb = open('sample.pdf', 'rb')
f  = open(bfilename,'a')
counter = 1
content = fb.read()


while True:
    try:
        try:
            sock.getpeername()
            still_connected = True
        except:
            still_connected = False
        
        if still_connected:
            start = datetime.now()
            sock.send(content)
            data = sock.recv(1024)
            print(str(start)+" "+len(data))
            f.write(str(start)+" "+str(datetime.now())+"\n")
            counter=((counter+1))
    except:
        pass
sock.close()
fb.close()
f.close()
