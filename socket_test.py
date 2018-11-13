import socket
import time
    
def startTcpServer():
    host = "127.0.0.1"
    port = 5000

    mainsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if mainsock == -1:
        print("create sock fail!")
    print("create sock success!")
    mainsock.setblocking(1)
    mainsock.bind((host, port))
    
    mainsock.listen(1)  # 只能同时连接一个
    connectedsock, address = mainsock.accept()
    connectedsock.setblocking(1)
    print("connection from ", str(address))
    return connectedsock

def processTcpServer():
    while True:
        data = connectedsock.recv(1024)  # 接收buffer的大小
        data = data.decode("ascii")
        if not data:
            break

        print("from connected user : ")
        print( data, len(data))

        time.sleep(1)

        data = str(data).upper()
        
        print("sending data : " + data)
        connectedsock.send(bytes("test: %s" % data,encoding="ascii"))
    connectedsock.close()


def udpServer():
    host = "127.0.0.1"
    port = 5001

    mainsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    mainsock.bind((host, port))

    print("Server started")
    while True:
        data, addr = mainsock.recvfrom(1024)
        print("message from {}, is {}".format(addr),str(data))
        mainsock.sendto(data, addr)
    mainsock.close()

