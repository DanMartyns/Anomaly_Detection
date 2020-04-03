import bluetooth

server_sock=bluetooth.BluetoothSocket( bluetooth.RFCOMM )

port = 1
server_sock.bind(("",port))
server_sock.listen(1)
client_sock,address = server_sock.accept()
print("Accepted connection from host", address[0]," and the port", address[1] )

while True:
    data = client_sock.recv(1024)
    print("Data Length: ", len(data))

client_sock.close()
server_sock.close()