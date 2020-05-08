import bluetooth
import argparse
from datetime import datetime

server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
server_sock.bind(("", bluetooth.PORT_ANY))
server_sock.listen(1)

port = server_sock.getsockname()[1]

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--uuid', help='exclusive service identifier')
parser.add_argument('-d', '--directory', help='Output directory')
parser.add_argument('-n', '--name', help='service name')
args = parser.parse_args()

uuid = str(args.uuid) #"94f39d29-7d6d-437d-973b-fba39e49d4ee"

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H_%M_%S")
bfilename = "vazio_"+dt_string+".txt"
f  = open('../00_data_extraction/data/'+args.directory+'/'+bfilename,'a')

bluetooth.advertise_service(server_sock, args.name, service_id=uuid,
                            service_classes=[uuid, bluetooth.SERIAL_PORT_CLASS],
                            profiles=[bluetooth.SERIAL_PORT_PROFILE],
                            # protocols=[bluetooth.OBEX_UUID]
                            )

print("Waiting for connection on RFCOMM channel", port)

while True:

	client_sock, client_info = server_sock.accept()
	print("Accepted connection from", client_info)

	start = datetime.now()
	try:
		while True:
			data = client_sock.recv(1024)
			if not data:
				break
	except OSError:
		pass
	finally:
		f.write(str(start)+" "+str(datetime.now())+"\n")
		f.flush()
		print(str(start)+" "+str(datetime.now())+"\n")

print("Disconnected.")
client_sock.close()
server_sock.close()
print("All done.")
