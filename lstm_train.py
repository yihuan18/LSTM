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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


batch_size = 1
lstm_hidden_size = 30
layer_num = 2


# 模型建模
def lstm_model(X,X_length,y,is_training):
    
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size) for _ in range(layer_num)])
    X_length_temp = (sess.run(X_length))[0]
    print("[lstm]in lstm model x length : " , X_length, X_length_temp)
    #print(sess.run(X))
    outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=X, sequence_length=X_length, dtype=tf.float32)
    print("[lstm]outputs : ", outputs)
    output = outputs[:, X_length_temp-1, :]  #最后一个时刻的输出结果
    print("[lstm]output : ", output)
    
    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)
    print("[lstm]predictions : ", predictions)
    #如果是预测
    if not is_training:
        return predictions, None, None

    #计算损失函数
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)

    #创建模型优化器并得到优化步骤
    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), optimizer = "Adagrad", learning_rate = 0.1)
    return predictions, loss, train_op

# 训练模型
def train(sess, train_X, train_X_length, train_y):
    #将训练数据以数据集的方式提供给计算图
    print("[train]train_X shape : ", np.array(train_X).shape)
    print("[train]train_X_length shape : ", np.array(train_X_length).shape)
    print("[train]train_y shape : ", np.array(train_y).shape)

    ds = tf.data.Dataset.from_tensor_slices((train_X, train_X_length, train_y))
    ds = ds.repeat(None).shuffle(30000).batch(batch_size)
    X, X_length, y = ds.make_one_shot_iterator().get_next()

    print("[train]X dimension is : ", X.get_shape())
    print("[train]but valid bytes length is : ", X_length.get_shape(), sess.run(X_length))

    #调用模型，得到预测结果，损失函数，和训练操作
    with tf.variable_scope("model"):
        predictions, loss, train_op = lstm_model(X, X_length, y, True)
    
    saver = tf.train.Saver()
    #初始化变量
    sess.run(tf.global_variables_initializer())
    training_steps = len(train_X)
    print("[train]training_steps:", training_steps)
    for i in range(training_steps):
        _, l =  sess.run([train_op, loss])
        #if i%100 == 0:
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
    #print("test_X_length shape : ", np.array(test_X_length).shape)
    
    
    print("[run_eval]test_X shape : ", np.array(test_X).shape)
    print("[run_eval]test_X_length shape : ", np.array(test_X_length).shape)
    print("[run_eval]test_y shape : ", np.array(test_y).shape)
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_X_length, test_y))
    ds = ds.repeat(None).batch(1)
    X, X_length, y = ds.make_one_shot_iterator().get_next()
    print("[run_eval]X dimension is : ", X.get_shape())
    print("[run_eval]but valid bytes length is : ", X_length.get_shape(), sess.run(X_length))
    #调用模型，得到预测结果，损失函数，和训练操作
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        prediction, _, _ = lstm_model(X, X_length, [0.0], False)
    
    load_model(sess)
    sess.run(tf.local_variables_initializer())
    
    #将预测结果存入一个数组
    predictions = []
    labels = []
    testing_steps = len(test_X)
    print("[run_eval]test_steps is" , testing_steps)
    for i in range(testing_steps):
        p, l = sess.run([prediction, y])
        #if i%10 == 0:
        print("[run_eval]testing step : " +  str(i) + ", label : " + str(l)+ ", prediction : " + str(p))
        predictions.append(p)
        labels.append(l)

    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print("[run_eval]Mean Square Error is: %f" % rmse)


def load_model(sess):
    saver = tf.train.Saver()
    saver.restore(sess, "./Model/lstm_model.ckpt")

def train_test_process(sess):
    #####################yihuan load data process#########################
    X_train, X_train_length, y_train, X_test, X_test_length, y_test = generateInput.generateInput_bytes("mutation_file",70)
    print("[main]X_train_length : ", X_train_length)
    print("[main]X_test_length : ",X_test_length)
    print("[main]维度：",len(X_test[0][0]))

    #####################yihuan train process#####################
    training_steps = len(X_train)
    print("[main]training_steps : " + str(training_steps))
    train(sess, X_train, X_train_length, y_train)

    #####################yihuan test process########################
    testing_steps = len(X_test)
    print("[main]testing_steps : " , testing_steps)
    run_eval(sess, X_test, X_test_length, y_test)

with tf.Session() as sess:
    train_test_process(sess)


        
    

    
    
