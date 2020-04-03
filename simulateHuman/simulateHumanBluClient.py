import bluetooth
import time

bd_addr = "68:EC:C5:2C:A8:7A"
port = 1

sock=bluetooth.BluetoothSocket( bluetooth.RFCOMM )
sock.connect((bd_addr, port))

fb=open('sample.pdf', 'rb')
counter = 1

while True:
    print("Will be send data ",counter)
    sock.send(fb.read())
    print("Sended data")
    counter=((counter+1))
    time.sleep(30)
sock.close()
