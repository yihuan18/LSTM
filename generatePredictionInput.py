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

def generateInput(dir):
    file_name = dir + "/original"
    seed_binary = fileTransform.file2binary(file_name)

    mutationFiles = traverse(dir + "/mutations")

    binary_X = []
    binary_y = []
    for files in mutationFiles:
        temp_binary = fileTransform.file2binary(files)
        if(len(temp_binary) < len(seed_binary)):
            for _ in range(len(seed_binary) - len(temp_binary)):
                temp_binary.append(0)
        elif(len(temp_binary) > len(seed_binary)):
            temp_binary = temp_binary[:len(seed_binary)]

        temp_binary = NOR(temp_binary, seed_binary)
        binary_X.append([temp_binary])
        binary_y.append([0.0])

    #print(len(binary_X_train))
    return binary_X, binary_y, mutationFiles

def generateInput_bytes(dir):
    file_name = dir + "/original"
    seed_bytes = fileTransform.file2bytes(file_name)

    mutationFiles = traverse(dir + "/mutations")

    bytes_X = []
    bytes_y = []
    for files in mutationFiles:
        temp_bytes = fileTransform.file2bytes(files)
        if(len(temp_bytes) < len(seed_bytes)):
            for _ in range(len(seed_bytes) - len(temp_bytes)):
                temp_bytes.append(0)
        elif(len(temp_bytes) > len(seed_bytes)):
            temp_bytes = temp_bytes[:len(seed_bytes)]

        temp_bytes = NOR(temp_bytes, seed_bytes)
        bytes_X.append([temp_bytes])
        bytes_y.append([0.0])

    return bytes_X, bytes_y, mutationFiles