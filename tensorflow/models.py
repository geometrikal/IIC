import tensorflow as tf
import numpy as np

from IID_losses_tf import IID_loss
from mnist_draw import convex_combo

from tensorflow.keras.layers import (
  Conv2D, Dense, GlobalMaxPooling2D, MaxPooling2D, Dropout, Flatten, 
  BatchNormalization, Lambda
)

class VGGModel(tf.keras.Model):
  def __init__(self, k=10, heads=5, aux_overcluster=True):
    super(VGGModel, self).__init__()
    self.trunk = []

    self.trunk += [Conv2D(64, (5,5), (2,2), activation=tf.nn.relu,
                             kernel_initializer='he_normal', name='c11')]
    self.trunk += [Conv2D(64, (3,3), (1,1), activation=tf.nn.relu,
                             kernel_initializer='he_normal',
                             padding='SAME', name='c12')]
    self.trunk += [Conv2D(256, (2,2), (2,2), activation=tf.nn.relu,
                             kernel_initializer='he_normal', name='c21')]
    self.trunk += [Conv2D(256, (2,2), (1,1), activation=tf.nn.relu,
                             kernel_initializer='he_normal',
                             padding='SAME', name='c22')]
    self.trunk += [Conv2D(512, (2,2), (2,2), activation=tf.nn.relu,
                             kernel_initializer='he_normal')]
    self.trunk += [Conv2D(512, (2,2), (1,1), activation=tf.nn.relu,
                             kernel_initializer='he_normal',
                             padding='SAME')]
    self.trunk += [Flatten()]
    # self.trunk += [Dropout(0.5)]
    self.trunk += [Dense(1024, activation=tf.nn.relu,
                            kernel_initializer='he_normal')]
    # self.trunk += [Dropout(0.5)]
    self.trunk += [Dense(512, activation=tf.nn.relu,
                            kernel_initializer='he_normal')]

    self.mainheads = []
    for j in range(heads):
      self.mainheads += [Dense(k, activation=tf.nn.softmax, name='k{}'.format(j),
                               kernel_initializer='he_normal')]

    if aux_overcluster:
      self.auxheads = []
      for j in range(heads):
        self.auxheads += [Dense(k*3, activation=tf.nn.softmax, name='aux{}'.format(j),
                                kernel_initializer='he_normal')]

  def call(self, x, head='main', verbose=False):
    for l in self.trunk:
      x = l(x)
      if verbose:
        print(l.name, ':' , x.shape)
    if head=='main':
      rets = [l(x) for l in self.mainheads]
    elif head=='aux':
      rets = [l(x) for l in self.auxheads]
    return rets


class ResNetModel(tf.keras.Model):
  def __init__(self, k=10, heads=5, aux_overcluster=True):
    super(ResNetModel, self).__init__()
    self.trunk, self.blocks, self.downsamplers = [], [], []
    self.clusterer = [] 

    self.trunk += [Conv2D(32, (5,5), (2,2), activation=tf.nn.relu,
                             kernel_initializer='he_normal', 
                             padding='SAME',
                             name='c11')]
    self.blocks += [self._make_residual_block(2, 32, 1)]

    self.trunk += [Conv2D(32, (3,3), (2,2), activation=tf.nn.relu,
                             kernel_initializer='he_normal', 
                             padding='SAME',
                             name='c21')]
    self.blocks += [self._make_residual_block(2, 32, 2)]

    self.trunk += [Conv2D(64, (3,3), (2,2), activation=tf.nn.relu,
                             kernel_initializer='he_normal', 
                             padding='SAME',
                             name='c31')]
    self.blocks.append(self._make_residual_block(2, 64, 3))

    self.clusterer += [Flatten()]
    # self.trunk += [Dropout(0.5)]
    self.clusterer += [Dense(512, activation=tf.nn.relu,
                            kernel_initializer='he_normal')]
    # self.trunk += [Dropout(0.5)]
    # self.clusterer += [Dense(1024, activation=tf.nn.relu,
    #                         kernel_initializer='he_normal')]

    self.mainheads = []
    for j in range(1):
      self.mainheads += [Dense(k, activation=tf.nn.softmax, 
                               name='k{}'.format(j),
                               kernel_initializer='he_normal')]

    if aux_overcluster:
      self.auxheads = []
      for j in range(heads):
        self.auxheads += [Dense(k*3, activation=tf.nn.softmax, 
                                name='aux{}'.format(j),
                                kernel_initializer='he_normal')]

  def call(self, x, head='main', verbose=False):
    for trunk_layer, res_block in zip(self.trunk, self.blocks):
      # xds = ds(x)
      x = trunk_layer(x)# + xds
      x_ = self._call_residual_block(x, res_block)
      x = tf.concat([x, x_], axis=-1)
      if verbose:
        print('res', x.shape)

    for l in self.clusterer:
      x = l(x)

    if head=='main':
      rets = [l(x) for l in self.mainheads]
    elif head=='aux':
      rets = [l(x) for l in self.auxheads]
    return rets

  def _call_residual_block(self, z_in, block):
    z_out = z_in 
    # print('callres', z_out.shape)
    for l in block:
      z_out = l(z_out)
      # print('callres loop', z_out.shape)
    return z_in + z_out

  def _make_residual_block(self, nblocks, filters, blocknum,
                           convargs={
                             'strides': (1,1),
                             'kernel_size': (3,3),
                             'activation': tf.nn.relu,
                             'kernel_initializer': 'he_normal',
                             'padding': 'SAME'}):
    layers = []
    for k in range(nblocks):
      layers.append(
        Conv2D(filters, name='rc{}-{}'.format(blocknum, k), **convargs))
      # layers.append(BatchNormalization())
    return layers
