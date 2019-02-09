import numpy as  np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim

X = np.linspace(-2,5,1000).reshape(-1,1)
d = np.square(X)+1
d = d.reshape(-1,1)
x = tf.placeholder(dtype=tf.float32,shape=[None,1],name="input_X")
y = tf.placeholder(dtype=tf.float32,shape=[None,1],name="input_Y")
# 定义网络
net = slim.fully_connected(x,2,activation_fn=tf.nn.relu)
net = slim.fully_connected(net,2,activation_fn=tf.nn.relu)
net = slim.fully_connected(net,1,activation_fn=None)
loss = tf.reduce_mean(tf.square(net-y))
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
l = []
for itr in range(10000):
    idx = np.random.randint(0,1000,512) 
    inx = X[idx]
    iny = d[idx]
    sess.run(train_step,feed_dict={x:inx,y:iny})
    if itr%100==0:
        print("step:{}".format(itr))
        #print()
        l_var  = sess.run(loss,feed_dict={x:X,y:d})
        l.append(l_var)
plt_x = np.linspace(-2,5,200).reshape(-1,1)
plt_y = sess.run(net,feed_dict={x:plt_x})
plt.plot(plt_x,plt_y,color='#FF0000')
plt.scatter(X,d)
plt.show()
plt.plot(l)
plt.show()
