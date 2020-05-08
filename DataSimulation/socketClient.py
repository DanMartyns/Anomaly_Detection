import sys
import time
import bluetooth

addr = None

if len(sys.argv) < 2:
    print("No device specified. Searching all nearby bluetooth devices for "
          "the SampleServer service...")
else:
    addr = sys.argv[1]
    print("Searching for SampleServer on {}...".format(addr))

fb = open('sample_5MB.pdf','rb')
data = fb.read()

while True:
    try:
        # search for the SampleServer service
        uuid = sys.argv[2] #"94f39d29-7d6d-437d-973b-fba39e49d4ee"
        service_matches = bluetooth.find_service(uuid=uuid, address=addr)

        if len(service_matches) == 0:
            print("Couldn't find the SampleServer service.")
            sys.exit(0)

        first_match = service_matches[0]
        port = first_match["port"]
        name = first_match["name"]
        host = first_match["host"]

        print("Connecting to \"{}\" on {}".format(name, host))

        # Create the client socket
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        sock.connect((host, port))

        try:
            print("A enviar documento")
            sock.send(data)
            print("Documento enviado")
        except:
            pass
    except:
        pass
    time.sleep(60)

sock.close()
