import tensorflow as tf
import numpy as np
# 创建输入数据
# batch_size=2,time_step=2,embedding_size=1,rnn_size=64
X = np.random.randn(2, 2, 1)
print(X)
# 第二个example长度为1
X[1,1:] = 0
X_lengths = [2, 1]
print(X)
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=64)

outputs, last_states = tf.nn.dynamic_rnn(cell=cell, dtype=tf.float64, sequence_length=X_lengths, inputs=X)
output = tf.reshape(outputs, [-1, 2])

result = tf.contrib.learn.run_n({"outputs": outputs, "last_states": last_states}, n=1, feed_dict=None)

#print(result[0])

assert result[0]["outputs"].shape == (2, 2, 64)

# 第二个example中的outputs超过1步(即第2步)的值应该为0
assert (result[0]["outputs"][1,1,:] == np.zeros(cell.output_size)).all()