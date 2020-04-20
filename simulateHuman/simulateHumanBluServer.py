import bluetooth

server_sock=bluetooth.BluetoothSocket( bluetooth.RFCOMM )

port = 1
server_sock.bind(("",port))
server_sock.listen(1)
client_sock,address = server_sock.accept()
print("Accepted connection from host", address[0]," and the port", address[1] )

still_connected = False
while True:
    try:
        while True:
            try:
                client_sock.getpeername()
                still_connected = True
            except:
                still_connected = False

            data = client_sock.recv(1024)
            print("Data Length: ", len(data))

            if still_connected:
                client_sock.send(data)
    except: 
        pass
    else:
        break
client_sock.close()
server_sock.close()
