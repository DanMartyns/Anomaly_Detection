import sys
import bluetooth
import time

bd_addr = '68:EC:C5:2C:A8:7A'
port = 1
socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
socket.connect((bd_addr, port))

fb = open('sample_5MB.pdf','rb')
data = fb.read()

while True:
    socket.send(data)
    time.sleep(30)
