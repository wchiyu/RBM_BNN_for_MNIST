import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def sample_prob(probs):
    return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

mnist=input_data.read_data_sets("D:/MNIST/",one_hot=True)
train_x=mnist.train.images
train_y=mnist.train.labels
test_x=mnist.test.images
test_y=mnist.test.labels

alpha = 1.0
batchsize = 100

X=tf.placeholder(tf.float32,[None,784])
Y=tf.placeholder(tf.float32,[None,10])

'''
rbm_w = tf.placeholder("float", [784, 500])
rbm_vb = tf.placeholder("float", [784])
rbm_hb = tf.placeholder("float", [500])
'''

rbm_w=tf.Variable(tf.random_normal([784,500],stddev=0.1),name="rbm_w")
rbm_vb=tf.Variable(tf.zeros([1,784]),name="rbm_vb")
rbm_hb=tf.Variable(tf.zeros([1,500]),name="rbm_hb")

h0 = sample_prob(tf.nn.sigmoid(tf.matmul(X, rbm_w) + rbm_hb))
v1 = sample_prob(tf.nn.sigmoid(tf.matmul(h0, tf.transpose(rbm_w)) + rbm_vb))
h1 = tf.nn.sigmoid(tf.matmul(v1, rbm_w) + rbm_hb)
w_positive_grad = tf.matmul(tf.transpose(X), h0)
w_negative_grad = tf.matmul(tf.transpose(v1), h1)
update_w=alpha * (w_positive_grad - w_negative_grad) / tf.to_float(tf.shape(X)[0])
update_vb=alpha * tf.reduce_mean(X - v1, 0)
update_hb=alpha * tf.reduce_mean(h0 - h1, 0)

assign1=tf.assign(rbm_w,rbm_w+update_w)
assign2=tf.assign(rbm_vb,rbm_vb+update_vb)
assign3=tf.assign(rbm_hb,rbm_hb+update_hb)

h_sample = sample_prob(tf.nn.sigmoid(tf.matmul(X, rbm_w) + rbm_hb))
v_sample = sample_prob(tf.nn.sigmoid(tf.matmul(h_sample, tf.transpose(rbm_w)) + rbm_vb))
err = X - v_sample
err_sum = tf.reduce_mean(err * err)

w2=tf.Variable(tf.random_normal([500,10],stddev=0.1),name="w2")
b2=tf.Variable(tf.zeros([1,10]),name="b2")
nn_h1=tf.nn.relu(tf.matmul(X,rbm_w)+rbm_hb,name="nn_h1")
y1=tf.nn.softmax(tf.matmul(nn_h1,w2)+b2,name="y1")

parameters_list=[rbm_w,rbm_hb,w2,b2]
cost=-tf.reduce_mean(Y*tf.log(y1))
optimizer=tf.train.AdamOptimizer(0.01).minimize(cost,var_list=parameters_list)
corrent_predict=tf.equal(tf.arg_max(y1,1),tf.arg_max(Y,1))
accuracy=tf.reduce_mean(tf.cast(corrent_predict,"float"))

with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(err_sum,feed_dict={X:train_x,Y:train_y}))
    for i in range(3):
        for start,end in zip(range(0,len(train_x),batchsize),range(batchsize,len(train_x),batchsize)):
            batch_x=train_x[start:end,:]
            batch_y=train_y[start:end,:]
            sess.run([assign1,assign2,assign3],feed_dict={X:batch_x,Y:batch_y})
            if start % 10000 == 0:
                print(sess.run(err_sum,feed_dict={X:train_x,Y:train_y}))
            
    for j in range(1000):
        batch_x,batch_y=mnist.train.next_batch(100)
        sess.run(optimizer,feed_dict={X:batch_x,Y:batch_y})
        if j%100==0:
            co,acc=sess.run([cost,accuracy],feed_dict={X:test_x,Y:test_y})
            print("%f---------%f%%"%(co,acc*100))
            














































