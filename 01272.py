import numpy as np
import matplotlib.pyplot as plt
files = np.load("/Users/wangsen/ai/13/homework/NO4-0127-TensorFlow&跨年作业/第一次作业-TensorFlow作业/homework.npz")
X = files['X']
label = files['d']
len = X.shape[0]


label_one_hot = []
for x1, x2 in X:
    if x1 > 0 and x2 > 0:
        label_one_hot.append([1, 0])
    elif x1 < 0 and x2 < 0:
        label_one_hot.append([1, 0])
    else:
        label_one_hot.append([0, 1])
label_one_hot = np.array(label_one_hot)

import tensorflow as tf
import tensorflow.contrib.slim as slim
x = tf.placeholder(tf.float32, [None, 2], name="input_x")
d = tf.placeholder(tf.float32, [None, 2], name="input_y")
# 对于sigmoid激活函数而言，效果可能并不理想
net = slim.fully_connected(x, 4, activation_fn=tf.nn.relu, 
                              scope='full1', reuse=False)
net = slim.fully_connected(net, 4, activation_fn=tf.nn.relu, 
                              scope='full4', reuse=False)
y = slim.fully_connected(net, 2, activation_fn=None, 
                              scope='full5', reuse=False)
# loss = tf.reduce_mean(tf.square(y-d))
loss = tf.reduce_mean(-d*tf.log(tf.nn.softmax(y)))
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(d, 1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.01)
gradient = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
train_step = optimizer.apply_gradients(gradient)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
l = []
a = []
for itr in range(1000):
    idx = np.random.randint(0, 2000, 20)
    inx = X[idx]
    ind = label_one_hot[idx]
    if itr%10==0:
        _accuracy = sess.run(accuracy,feed_dict={d:label_one_hot,x:X})
        print("itr:{} accuracy:{}".format(itr,_accuracy))
        _loss = sess.run(loss,feed_dict={d:ind,x:inx})
        l.append(_loss)
        a.append(_accuracy)
    sess.run(train_step,feed_dict={d:ind,x:inx})

predict = sess.run(tf.argmax(y, 1),feed_dict={x:[[0.2,0.2]]})
print("[02,0,2] predict %d" % predict)
plt.plot(l)
plt.plot(a)
plt.show()
