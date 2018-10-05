import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 

LR = 0.1
REAL_PARAMS = [1.2,2.5]
INIT_PARAMS = [5,4],
				[5,1],
			     [2,4.5][2]
x= np.linspace(-1,1,200,dtype = np.float32)

y_fun = lambda a, b: a*x +b
#remenmber to define the same dtype and shape when restore
# W = tf.Variable([[1,2,3],[3,4,5]], dtype = tf.float32, name = 'weights')
# b = tf.Variable([[1,2,3]], dtype = tf.float32, name = 'biases')

# init = tf.initialize_all_variables()

# saver = tf.train.Saver()

# with tf.Session() as sess :
# 	sess.run(init)

# 	save_path = saver.save(sess, "my_net/save_net.ckpt")
# 	print("Save to path: ", save_path)



#restore variables
W = tf.Variable(np.arange(6).reshape((2,3)),dtype = tf.float32, name = "weights")
biases = tf.Variable(np.arange(3).reshape((1,3)),dtype = tf.float32, name = "biases")

#not need to init step

saver = tf.train.Saver()
with tf.Session() as sess:
	saver.restore(sess, "my_net/save_net.ckpt")
	print("weights: ", sess.run(W))
	print("biases: ", sess.run(biases))