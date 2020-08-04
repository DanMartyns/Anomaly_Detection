import sys
import time
import bluetooth
import argparse
import threading

addr = None

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--uuid', help='exclusive service identifier')
parser.add_argument('-s','--server', help='Server Address')
args = parser.parse_args()

if not args.uuid:
    print("No device specified. Searching all nearby bluetooth devices for "
          "the SampleServer service...")
else:
    addr = args.server
    print("Searching for SampleServer on {}...".format(addr))

fb = open('sample_18MB.pdf','rb')
data = fb.read()

#Function for handling connections. This will be used to create threads
def clientthread(sock, frame):
    sock.send(data)
    time.sleep(0.1)

dataframes = [data[i:i+1024] for i in range(0,len(data)-1,1024)]
print(dataframes)

start = None
threads = []

while True:
    try:
        # search for the SampleServer service
        uuid = args.uuid
        service_matches = bluetooth.find_service(uuid=uuid, address=addr)

        if len(service_matches) == 0:
            print("Couldn't find the SampleServer service.")
            sys.exit(0)

        first_match = service_matches[0]
        port = first_match["port"]
        name = first_match["name"]
        host = first_match["host"]

        print("Connecting to \"{}\" on {}".format(name, host))
        
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        sock.connect((host, port))

        print("A enviar documento")
        print("Tamanho do dataframe: ",len(dataframes))
        
        for frame in dataframes:
            try:
                new_thread = threading.Thread(target = clientthread , args =(sock, frame, ))
                new_thread.start()
                threads.append(new_thread)
            except:
                pass

        for thread in threads:
            print("À espera do join")
            thread.join()

    except Exception as e:
        print("Something happend :",e)
    finally:
        print("Documento enviado")
        sock.close()



