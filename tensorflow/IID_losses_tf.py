import sys
import tensorflow as tf

import numpy as np
from scipy.special import softmax
from generate_clusterings import genclust
  

def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
  k = x_out.shape[-1]
  # skip assertions

  # joint probability
  p = tf.reduce_sum(tf.expand_dims(x_out, 2) * tf.expand_dims(x_tf_out, 1), 0)
  p = (p + tf.transpose(p)) / 2 # symmetry
  p /= tf.reduce_sum(p) # normalize

  pi = tf.broadcast_to(tf.reshape(tf.reduce_sum(p, axis=1), (k, 1)), (k, k))
  pj = tf.broadcast_to(tf.reshape(tf.reduce_sum(p, axis=0), (1, k)), (k, k))

  p = tf.clip_by_value(p, EPS, 1e9)
  pi = tf.clip_by_value(pi, EPS, 1e9)
  pj = tf.clip_by_value(pj, EPS, 1e9)

  iloss = tf.reduce_sum( - p * (tf.math.log(p) \
                                - lamb * tf.math.log(pi) \
                                - lamb * tf.math.log(pj) ))

  return iloss

def test_IID_loss():
  b = 64
  k = 10

  print('Independent vectors:')
  x_out    = genclust(b=b, k=k, seed=100)
  x_tf_out = genclust(b=b, k=k, seed=200)
  loss = IID_loss(x_out, x_tf_out)
  print(loss.numpy())

  print('The same vector:')
  x_out    = genclust(b=b, k=k, seed=100)
  x_tf_out = genclust(b=b, k=k, seed=100)
  loss = IID_loss(x_out, x_tf_out)
  print(loss.numpy())

if __name__ == '__main__':
  tf.enable_eager_execution()
  test_IID_loss()
  
