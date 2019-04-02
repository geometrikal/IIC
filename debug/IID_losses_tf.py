import sys
import tensorflow as tf

import numpy as np
from scipy.special import softmax
def genclust(b = 64, k = 10, y=None, seed=0, noise_scale=1., add_noise=True):
  np.random.seed(seed)
  if y is None:
    # Generate class assignment if none is given
    y = np.random.choice(range(k), b, replace=True)
    y = np.eye(k)[y]  # onehot
  else:
    assert y.shape[-1] == k
  # Noise it up
  if add_noise:
    noise = np.random.randn(*y.shape) * noise_scale
    y += noise
  return softmax(y, axis=-1)
  

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

  print('Seed 1, 2')
  x_out    = genclust(b=b, k=k, seed=1)
  x_tf_out = genclust(b=b, k=k, seed=2)
  loss = IID_loss(x_out, x_tf_out)
  print(loss.numpy())

  print('Seed 1, 1, noise ~ N(0,1)')
  x_out    = genclust(b=b, k=k, seed=1)
  x_tf_out = genclust(b=b, k=k, seed=1)
  loss = IID_loss(x_out, x_tf_out)
  print(loss.numpy())

  print('Seed 1, 1, noise ~ N(0,3)')
  x_out    = genclust(b=b, k=k, seed=1, noise_scale=3.)
  loss = IID_loss(x_out, x_out)
  print(loss.numpy())

  print('Seed 1, 1, No noise')
  x_out    = genclust(b=b, k=k, seed=1, add_noise=False)
  loss = IID_loss(x_out, x_out)
  print(loss.numpy())

if __name__ == '__main__':
  tf.enable_eager_execution()
  test_IID_loss()
  