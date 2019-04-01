import tensorflow as tf
import sys

"""

Xu Ji, et al. Invariant Information Clustering for Unsupervised Image Classification and Segmentation
arXiv:1807.06653v3 (2019)

Implements Eq. 1, 2 and 3 from the paper describing the calculation of the joint probability function with a 
clustering objective, and the mutual information between two vectors.

Direct port of the snippet in the paper (Figure 4) plus the pytorch reference implementation.

See IIC/code/utils/cluster/IID_losses.py
"""
def IIC(z, zp, lam=1.):
  """
  z and zp are one-hot shaped (batch, classes)
  """
  b = z.shape[0] # batch dimension
  c = z.shape[1] # clusters dimension

  p = tf.reduce_sum(tf.expand_dims(z, 2) * tf.expand_dims(zp, 1), 0)
  p = (p + tf.transpose(p)) / 2 # symmetry
  p /= tf.reduce_sum(p) # normalize

  assert p.shape == (c,c)

  pi = tf.broadcast_to(tf.reshape(tf.reduce_sum(p, axis=1), (c,1)), (c,c))
  pj = tf.broadcast_to(tf.reshape(tf.reduce_sum(p, axis=0), (1,c)), (c,c))

  p = tf.clip_by_value(p,   sys.float_info.epsilon, 1e10) # clip 0's
  pi = tf.clip_by_value(pi, sys.float_info.epsilon, 1e10) 
  pj = tf.clip_by_value(pj, sys.float_info.epsilon, 1e10) 
  
  # return tf.reduce_sum( p * ( lam*tf.math.log(pi) + lam*tf.math.log(pj) - tf.math.log(p) ) )
  return tf.reduce_sum( - p * ( tf.math.log(p) - lam*tf.math.log(pi) - lam*tf.math.log(pj)) ) 

