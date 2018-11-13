import fileTransform
import os

# 遍历目录
def traverse(directory):
    filename = os.listdir(directory)
    file_path = []
    for fl in filename:
        file_path.append(os.path.join(directory,fl))
    return file_path

#将两个序列进行异或运算
def NOR(seq1,seq2):
    result = []
    if (len(seq1) != len(seq2)):
        print("seq1 len : " + str(len(seq1)) + " , seq2 len : " + str(len(seq2)))
        print("False!!!")
        return
    for i in range(len(seq1)):
        if(seq1[i] == seq2[i]):
            result.append(0.0)
        else:
            result.append(1.0)
    return result


#生成 bitwise 的输入 
def generateInput(seedfile):
    file_name = seedfile + "/original"
    seed_binary = fileTransform.file2binary(file_name)

    fileType1 = traverse(seedfile + "/no_use") 
    fileType2 = traverse(seedfile + "/use")

    #无效变异
    binary_X = []
    binary_y = []
    for files in fileType1:
        temp_binary = fileTransform.file2binary(files)

        if(len(temp_binary) < len(seed_binary)):
            for _ in range(len(seed_binary) - len(temp_binary)):
                temp_binary.append(0)
        elif(len(temp_binary) > len(seed_binary)):
            temp_binary = temp_binary[:len(seed_binary)]

        if(len(temp_binary) == len(seed_binary)):
            temp_binary = NOR(temp_binary, seed_binary)
            binary_X.append([temp_binary])
            binary_y.append([0.0])

    train_length = int(len(binary_X)*0.9)

    binary_X_train = binary_X[0:train_length]
    binary_X_test = binary_X[train_length:]

    binary_y_train = binary_y[0:train_length]
    binary_y_test = binary_y[train_length:]

    #有效变异
    binary_X_2 = []
    binary_y_2 = []
    for files in fileType2:
        temp_binary = fileTransform.file2binary(files)

        if(len(temp_binary) < len(seed_binary)):
            for _ in range(len(seed_binary) - len(temp_binary)):
                temp_binary.append(0)
        elif(len(temp_binary) > len(seed_binary)):
            temp_binary = temp_binary[:len(seed_binary)]

        if(len(temp_binary) == len(seed_binary)):
            temp_binary = NOR(temp_binary, seed_binary)
            binary_X_2.append([temp_binary])
            binary_y_2.append([1.0])

    train_length = int(len(binary_X_2)*0.9)

    binary_X_2_train = binary_X_2[0:train_length]
    binary_X_2_test = binary_X_2[train_length:]

    binary_y_2_train = binary_y_2[0:train_length]
    binary_y_2_test = binary_y_2[train_length:]

    #获取训练数据和测试数据
    binary_X_train += binary_X_2_train
    binary_y_train += binary_y_2_train
    binary_X_test += binary_X_2_test
    binary_y_test += binary_y_2_test

    #print(len(binary_X_train))
    return binary_X_train,binary_y_train,binary_X_test,binary_y_test

def generateInput_bytes_old(seedfile):
    file_name = seedfile + "/original"
    seed_bytes = fileTransform.file2bytes(file_name)

    fileType1 = traverse(seedfile + "/no_use") 
    fileType2 = traverse(seedfile + "/use")

    #无效变异
    bytes_X = []
    bytes_y = []
    for files in fileType1:
        temp_bytes = fileTransform.file2bytes(files)
        
        if(len(temp_bytes) < len(seed_bytes)):
            for _ in range(len(seed_bytes) - len(temp_bytes)):
                temp_bytes.append(0)
        elif(len(temp_bytes) > len(seed_bytes)):
            temp_bytes = temp_bytes[:len(seed_bytes)]

        if(len(temp_bytes) == len(seed_bytes)):
            temp_bytes = NOR(temp_bytes, seed_bytes)
            bytes_X.append([temp_bytes])
            bytes_y.append([0.0])

    train_length = int(len(bytes_X)*0.9)

    X_train = bytes_X[0:train_length]
    X_test = bytes_X[train_length:]

    y_train = bytes_y[0:train_length]
    y_test = bytes_y[train_length:]

    #有效变异
    bytes_X1 = []
    bytes_y1 = []
    for files in fileType2:
        temp_bytes = fileTransform.file2bytes(files)

        if(len(temp_bytes) < len(seed_bytes)):
            for _ in range(len(seed_bytes) - len(temp_bytes)):
                temp_bytes.append(0)
        elif(len(temp_bytes) > len(seed_bytes)):
            temp_bytes = temp_bytes[:len(seed_bytes)]

        if(len(temp_bytes) == len(seed_bytes)):
            temp_bytes = NOR(temp_bytes, seed_bytes)
            bytes_X1.append([temp_bytes])
            bytes_y1.append([1.0])

    train_length = int(len(bytes_X1)*0.5)

    X_train1 = bytes_X1[0:train_length]
    X_test1 = bytes_X1[train_length:]

    y_train1 = bytes_y1[0:train_length]
    y_test1 = bytes_y1[train_length:]

    #获取训练数据和测试数据
    X_train += X_train1
    X_test += X_test1
    y_train += y_train1
    y_test += y_test1
    return X_train, y_train, X_test, y_test

def generateInput_bytes(seedfile , dimension):
    fileType1 = traverse(seedfile + "/useless") 
    fileType2 = traverse(seedfile + "/useful")

    #无效变异
    bytes_X = []
    bytes_y = []
    bytes_length=[]
    for files in fileType1:
        temp_bytes = fileTransform.file2bytes(files)
        #bytes_length.append(len(temp_bytes))
        if(len(temp_bytes) < dimension):
            bytes_length.append(len(temp_bytes))
            for _ in range(dimension - len(temp_bytes)):
                temp_bytes.append([0.0])
        elif(len(temp_bytes) >= dimension):
            bytes_length.append(dimension)
            temp_bytes = temp_bytes[:dimension]

        bytes_X.append(temp_bytes)
        bytes_y.append([0.0])

    train_length = int(len(bytes_X)*0.9)

    X_train = bytes_X[0:train_length]
    X_test = bytes_X[train_length:]

    y_train = bytes_y[0:train_length]
    y_test = bytes_y[train_length:]

    X_train_length = bytes_length[0:train_length]
    X_test_length = bytes_length[train_length:]

    #有效变异
    bytes_X1 = []
    bytes_y1 = []
    bytes_length1=[]
    for files in fileType2:
        temp_bytes = fileTransform.file2bytes(files)
        #bytes_length1.append(len(temp_bytes))
        if(len(temp_bytes) < dimension):
            bytes_length1.append(len(temp_bytes))
            for _ in range(dimension - len(temp_bytes)):
                temp_bytes.append([0.0])
        elif(len(temp_bytes) >= dimension):
            bytes_length1.append(dimension)
            temp_bytes = temp_bytes[:dimension]

        bytes_X1.append(temp_bytes)
        bytes_y1.append([1.0])

    train_length = int(len(bytes_X1)*0.9)

    X_train1 = bytes_X1[0:train_length]
    X_test1 = bytes_X1[train_length:]

    y_train1 = bytes_y1[0:train_length]
    y_test1 = bytes_y1[train_length:]

    X_train1_length = bytes_length1[0:train_length]
    X_test1_length = bytes_length1[train_length:]

    #获取训练数据和测试数据
    X_train += X_train1
    X_test += X_test1
    X_train_length += X_train1_length
    X_test_length += X_test1_length
    y_train += y_train1
    y_test += y_test1
    #print(X_train)
    return X_train, X_train_length, y_train, X_test, X_test_length, y_test