import bluetooth
import argparse
import time
import sys
import os
import threading
from datetime import datetime

server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
server_sock.bind(("", bluetooth.PORT_ANY))
server_sock.listen(1)

port = server_sock.getsockname()[1]

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--uuid', help='exclusive service identifier')
parser.add_argument('-n', '--name', help='service name')
parser.add_argument('-d', '--directory', help='Output directory')
parser.add_argument('-t','--type', help='Type of file')
args = parser.parse_args()

uuid = str(args.uuid) #"94f39d29-7d6d-437d-973b-fba39e49d4ee"

bluetooth.advertise_service(server_sock, args.name, service_id=uuid,
                            service_classes=[uuid, bluetooth.SERIAL_PORT_CLASS],
                            profiles=[bluetooth.SERIAL_PORT_PROFILE],
                            protocols=[bluetooth.OBEX_UUID]
                            )

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H_%M_%S")
bfilename = args.type+"_"+dt_string+".txt"
f = open("../00_data_extraction/data/"+args.directory+"/"+bfilename,"a")


print("Waiting for connection on RFCOMM channel", port)

#Function for handling connections. This will be used to create threads
def clientthread(conn, start):
	try:
		size = 0
		while True:
			#Receiving from client
			data = conn.recv(2048)
			size += len(data)
	except Exception as e:
		print(e)
	finally:
		conn.close()
		print("Data Received at: ",str(start)+" "+str(datetime.now())+"Size: "+str(size))
		f.write(str(start)+" "+str(datetime.now())+'\n')
		f.flush()

threads = []
#now keep talking with the client
while True:
	#wait to accept a connection - blocking call
	conn, addr = server_sock.accept()
	print('Connected with ' + addr[0] + ':' + str(addr[1]))
	#start new thread takes 1st argument as a function name to be run, second is the tuple of arguments to the function.
	start = datetime.now()
	new_thread = threading.Thread(target = clientthread , args =(conn, start, ))
	new_thread.daemon = True
	new_thread.start()
	threads.append(new_thread)

	for thread in threads:
		thread.join()
	
server_sock.close()
