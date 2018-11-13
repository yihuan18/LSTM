#coding:utf-8
import tensorflow as tf
import numpy as np
import generateInput
import generatePredictionInput
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


batch_size = 32
lstm_hidden_size = 30
layer_num = 2

training_examples = 10000
testing_examples = 1000

def lstm_model(X,y,is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size) for _ in range(layer_num)])
  
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = outputs[:, -1, :]  #最后一个时刻的输出结果
    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)

    #如果是预测
    if not is_training:
        return predictions, None, None

    #计算损失函数
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)

    #创建模型优化器并得到优化步骤
    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), optimizer = "Adagrad", learning_rate = 0.1)
    return predictions, loss, train_op


def run_eval(sess, test_X, test_y):
    #将训练数据以数据集的方式提供给计算图
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()

    print("维度 ： " + str(len(test_X[1][0])))
    #调用模型，得到预测结果，损失函数，和训练操作
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        prediction, _, _ = lstm_model(X, [0], False)
    
    #########yihuan add 20180612 reload model
    saver = tf.train.Saver()
    saver.restore(sess, "./Model/lstm_model.ckpt")

    #将预测结果存入一个数组
    predictions = []
    labels = []
    print("testing_steps : " + str(testing_steps))
    for i in range(testing_steps):
        p, l = sess.run([prediction, y])
        if p >= 0.5:
            p = 1
        elif p < 0.5:
            p = 0
        if i%100 == 0:
            print("testing step : " +  str(i) + ", label : " + str(l)+ ", prediction : " + str(p))
        predictions.append(p)
        labels.append(l)
    return predictions
    #predictions = np.array(predictions).squeeze()
    #labels = np.array(labels).squeeze()
    #rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    #print("Mean Square Error is: %f" % rmse)


with tf.Session() as sess:
    #####################yihuan load data process##############
    #binary_X_train,binary_y_train,binary_X_test,binary_y_test = generateInput.generateInput("seed2")
   
    X_test, y_test, mutationFiles = generatePredictionInput.generateInput_bytes("test_data")
    
    #######################yihuan test process###############
    testing_steps = len(X_test)
    print("testing_steps : " + str(testing_steps))
    print("开始查模型")
    predictions = run_eval(sess, X_test, y_test)

    print("变异文件 ： ")
    print(mutationFiles)
    print("对应预测值 ：(0代表有效1代表无效)")
    print(predictions)
