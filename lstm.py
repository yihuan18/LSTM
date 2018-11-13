#coding:utf-8
import tensorflow as tf
import numpy as np
#用于绘图
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

import generateInput
import os
import socket_test
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


batch_size = 32
lstm_hidden_size = 30
layer_num = 2
model_loaded = 0


# 模型建模
def lstm_model(X,X_length,y,is_training):
    
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size) for _ in range(layer_num)])
    
    outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=X, sequence_length=X_length, dtype=tf.float32)
    X_length_temp = (sess.run(X_length))[0]
    print("X_length_temp : ",X_length_temp)
    output = outputs[:, X_length_temp - 1, :]  #最后一个时刻的输出结果
    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)
    
    #如果是预测
    if not is_training:
        return predictions, None, None

    #计算损失函数
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)

    #创建模型优化器并得到优化步骤
    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), optimizer = "Adagrad", learning_rate = 0.1)
    return predictions, loss, train_op

# 训练模型
def train(sess, train_X, train_y):
    #将训练数据以数据集的方式提供给计算图
    ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    ds = ds.repeat().shuffle(1000).batch(batch_size)
    X, y = ds.make_one_shot_iterator().get_next()

    #调用模型，得到预测结果，损失函数，和训练操作
    with tf.variable_scope("model"):
        predictions, loss, train_op = lstm_model(X, y, True)
    
    saver = tf.train.Saver()
    #初始化变量
    sess.run(tf.global_variables_initializer())
    training_steps = len(X_train)
    for i in range(training_steps):
        _, l =  sess.run([train_op, loss])
        if i%100 == 0:
            print("train step : " +  str(i) + ", loss : " + str(l))
    #####################yihuan add 20180612 save model
    saver.save(sess,"Model/lstm_model.ckpt")

    #取出所有参与训练的参数
    params = tf.trainable_variables()
    print("Trainable variables:------------------------")

    #循环列出参数
    for idx, v in enumerate(params):
     print("  param {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))

#  为训练的模型生成热力图--bitwise
def generateHeatmap(train_X):
    dimention = len(train_X[0][0])
    vector_x = []
    vector_y = []
    for i in range(dimention):
        temp = []
        for j in range(dimention):
            temp.append(0.0)
        temp[i] = 1
        vector_x.append([temp])
        vector_y.append([0.0])

    ds = tf.data.Dataset.from_tensor_slices((vector_x, vector_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()

    #调用模型，得到预测结果，损失函数，和训练操作
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        prediction, _, _ = lstm_model(X, [0], False)
    
    #########yihuan add 20180612 reload model
    saver = tf.train.Saver()
    saver.restore(sess, "./Model/lstm_model.ckpt")

    #将预测结果存入一个数组
    predictions = []
    labels = []
    probability = []
    count = 0
    #print("testing_steps : " + str(testing_steps))
    for i in range(dimention):
        p, l = sess.run([prediction, y])
        if p >= 0.5:
            p = 1
        elif p < 0.5:
            p = 0
        if p == 1:
            count = count + 1
        if i%8 == 0:
            probability.append(float(count)/8)
            count = 0
            print(probability)
        if i%100 == 0:
            print("testing step : " +  str(i) + ", label : " + str(l)+ ", prediction : " + str(p))
        predictions.append(p)
        labels.append(l)
    return predictions

#  为训练的模型生成热力图--bytewise
def generateHeatmap_bytes(dimention):
    vector_x = []
    vector_y = []
    for i in range(dimention):
        temp = []
        for j in range(dimention):
            temp.append(0.0)
        temp[i] = 1
        vector_x.append([temp])
        vector_y.append([0.0])

    ds = tf.data.Dataset.from_tensor_slices((vector_x, vector_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()

    #调用模型，得到预测结果，损失函数，和训练操作
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        prediction, _, _ = lstm_model(X, [0], False)
    
    #########yihuan add 20180612 reload model
    saver = tf.train.Saver()
    saver.restore(sess, "./Model/lstm_model.ckpt")

    #将预测结果存入一个数组
    predictions = []
    probability = []
    max_probability = 0.5
    min_probability = 0.5
    #print("testing_steps : " + str(testing_steps))
    for i in range(dimention):
        p, _ = sess.run([prediction, y])
        #获取最大和最小的预测值
        if p > max_probability:
            max_probability = float(p)
        elif p < min_probability:
            min_probability = float(p)
        if i%100 == 0:
            print("testing step : " +  str(i) + ", prediction : " + str(p))
        predictions.append(p)
    
    #归一化
    big_prob = []
    for i in predictions:
        temp = float(i)
        regulize_i = (temp - min_probability) / (max_probability - min_probability)
        probability.append(regulize_i)
        if regulize_i >= 0.5:
            big_prob.append(regulize_i)
    print("probablity的维度为：" + str(len(probability)))
    print("probablility : ")
    print(probability)
    print("超过0.5的probablity的个数为：" + str(len(big_prob)))
    print("big_prob : ")
    print(big_prob)
    return probability

def run_eval(sess, test_X, test_X_length, test_y):
    #将训练数据以数据集的方式提供给计算图
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_X_length, test_y))
    #ds = ds.repeat(None).batch(1)
    ds = ds.repeat(None).batch(1)
    X, X_length, y = ds.make_one_shot_iterator().get_next()

    #调用模型，得到预测结果，损失函数，和训练操作
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        prediction, _, _ = lstm_model(X, X_length, [0.0], False)
    
    load_model(sess)
    sess.run(tf.local_variables_initializer())
    #将预测结果存入一个数组
    predictions = []
    labels = []
    testing_steps = len(test_X)
    print("test_steps is" , testing_steps)
    for i in range(testing_steps):
        p, l = sess.run([prediction, y])
        if i%10 == 0:
            print("testing step : " +  str(i) + ", label : " + str(l)+ ", prediction : " + str(p))
        predictions.append(p)
        labels.append(l)

    predictions.append(p)
    labels.append(l)
    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print("Mean Square Error is: %f" % rmse)
    return p


def load_model(sess):
    saver = tf.train.Saver()
    saver.restore(sess, "./Model/lstm_model.ckpt")

def train_test_process(sess):
    #####################yihuan load data process#########################
    X_train, y_train, X_test, y_test = generateInput.generateInput_bytes("seed3")
    print("X_test")
    print(X_test)
    print(y_test)

    print("维度：")
    print(len(X_test[0][0]))
    #####################yihuan train process#####################
    training_steps = len(X_train)
    print("training_steps : " + str(training_steps))
    train(sess, X_train, y_train)

    #####################yihuan test process########################
    testing_steps = len(X_test)
    print("testing_steps : " + str(testing_steps))
    run_eval(sess, X_test, y_test)

    #####################yihuan generate heat map #############
    dimention = len(X_train[0][0])
    prob = generateHeatmap_bytes(dimention)
    #prob_100 = prob[0:50]
    #plt.plot(prob)
    #plt.plot(prob_100)
    #plt.show()


seed_dimension = 70
conSock = socket_test.startTcpServer()
while True:
            #### 接收数据 ######
            tf.reset_default_graph()
            graph = tf.Graph()
            with graph.as_default() as g:
                with tf.Session(graph=g) as sess:
                    data = conSock.recv(2048)  # 接收buffer的大小
                    data = data.decode("ascii")
                    recvdata_len = []
                    
                    print("recv data len : ",recvdata_len)
                
                    #for i in range(100):
                        #print(data[i])
                    ################################处理数据 ###################################
                    x_test = []
                    y_test = []
                    for i in range(len(data)):
                        if data[i] == '1':
                            x_test.append([1.0])
                        else:
                            x_test.append([0.0])

                    ##################################匹配种子长度######################################
                    if(len(data) < seed_dimension):
                        recvdata_len.append(len(data))
                        for i in range(seed_dimension - len(data)):
                            x_test.append([0.0])
                    elif len(data) >= seed_dimension:
                        recvdata_len.append(seed_dimension)
                        x_test = x_test[:seed_dimension]

                    byte_x_test = []
                    byte_y_test = [] 
                    byte_x_test.append(x_test)
                    byte_y_test.append([0.0])

                    ################################### 预测数据 ######################################
                    start_time = int(round(time.time() * 1000))

                    print("start to predict")
                    prediction = run_eval(sess,byte_x_test,recvdata_len,byte_y_test)
                    print("prediction result is : ", prediction)
                    #       send to client if useful
                    if prediction[0][0] < 0.5:
                        sendbuf = bytes('0',encoding="ascii")
                    else:
                        sendbuf = bytes('1',encoding="ascii")
                    
                    end_time = int(round(time.time() * 1000))
                    print("Time needed for predicting one case : ",end_time - start_time)

                    print("sending data")
                    length = conSock.send(sendbuf)
                    print("sending length : " + str(length))
            gc.collect()

        
    

    
    
