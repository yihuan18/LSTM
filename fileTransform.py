def file2binary(file_name):
    filename = file_name
    f = open(filename,"rb+")    
    data = f.read()
    f.close()
    binary_seq = []
    for i in data:
        #print(i)
        binary = bin(i)
        binary = binary[2:]
        for _ in range(8-len(binary)):
            binary_seq.append(0)
        for char in binary:
            binary_seq.append(int(char))
    return binary_seq


def file2bytes(file_name):
    filename = file_name
    #读文本文件
    f = open(filename,"r",encoding="ascii")  #读文本文件
    data = f.read()
    f.close()

    #存储字节流
    data_byte = []          #存放读取的字节流
    for i in data:
        if i == '1':
            data_byte.append([1.0])
        else:
            data_byte.append([0.0])
    return data_byte