import tensorflow as tf
import numpy as np
x = tf.zeros([1,32,32,3])
# x = tf.constant([[[[1,2,3,4]]]])
print(x.shape)
# y = tf.depth_to_space(x,2)
a = tf.space_to_depth(x,2)
with tf.Session() as sess:
#     z = sess.run(y)
    b = sess.run(a)
    
# print(z)
# print(z.shape)
print(b)
print(b.shape)
