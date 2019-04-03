import tensorflow as tf
import numpy as np

from IID_losses_tf import IID_loss
from mnist_draw import convex_combo

from tensorflow.keras.layers import (
  Conv2D, Dense, GlobalMaxPooling2D, MaxPooling2D, Dropout, Flatten
)

from matplotlib import pyplot as plt
plt.style.use('seaborn-whitegrid')

from data import get_iterator

class ClusterModel(tf.keras.Model):
  def __init__(self, k=10, heads=5, aux_overcluster=True):
    super(ClusterModel, self).__init__()
    self.mylayers = []
    # self.mylayers += [Dense(1024, activation=tf.nn.relu)]
    # self.mylayers += [Dropout(0.5)]
    # self.mylayers += [Dense(512, activation=tf.nn.relu)]
    # self.mylayers += [Dropout(0.5)]

    self.mylayers += [MaxPooling2D((2,2), (2,2))]
    self.mylayers += [Conv2D(64, (5,5), (2,2), activation=tf.nn.relu)]
    self.mylayers += [Conv2D(128, (2,2), (1,1), activation=tf.nn.relu)]
    # self.mylayers += [Conv2D(256, (3,3), (1,1), activation=tf.nn.relu)]
    self.mylayers += [Conv2D(256, (2,2), (1,1), activation=tf.nn.relu)]
    self.mylayers += [Flatten()]
    # self.mylayers += [Dropout(0.5)]
    self.mylayers += [Dense(512, activation=tf.nn.relu)]

    self.mainheads = []
    for j in range(heads):
      self.mainheads += [Dense(k, activation=tf.nn.softmax, name='k{}'.format(j))]

    if aux_overcluster:
      self.auxheads = []
      for j in range(heads):
        self.auxheads += [Dense(k*3, activation=tf.nn.softmax, name='aux{}'.format(j))]

  def call(self, x, head='main', verbose=False):
    for l in self.mylayers:
      x = l(x)
      if verbose:
        print(l.name, ':' , x.shape)
    if head=='main':
      rets = [l(x) for l in self.mainheads]
    elif head=='aux':
      rets = [l(x) for l in self.auxheads]
    return rets

def chain_crop_resize(x):
  x = tf.expand_dims(x, 0)
  x = tf.image.random_crop(x, (x.shape[0], 21, 21, x.shape[-1]))
  x = tf.image.resize_image_with_pad(x, 28, 28)
  x = tf.squeeze(x, axis=0)
  return x

def perturb_x(x, stddev=0.15):
  # Perturb with additive noise
  noise = tf.random.normal(stddev=stddev, shape=x.shape)
  # x = tf.map_fn(tf.image.random_flip_left_right, x, parallel_iterations=4)
  x = tf.map_fn(chain_crop_resize, x, parallel_iterations=4)
  x = tf.clip_by_value(x + noise, 0, 1)
  return x

def generate_mnist(x_src, y_src, batch_size=32, repeat=4, perturb=True):
  n_batches = x_src.shape[0] / batch_size
  indices = np.random.choice(np.arange(x_src.shape[0]), int(n_batches*batch_size), replace=False)
  for idx in np.split(indices, n_batches):
    x_batch = tf.concat([x_src[idx, ...]] * repeat, axis=0)
    x_batch = tf.expand_dims(tf.cast(x_batch, tf.float32), -1)
    y_batch = np.concatenate([y_src[idx]] * repeat)

    if perturb:
      x_perturb = perturb_x(x_batch)
      yield x_batch, x_perturb, y_batch
      # yield tf.reshape(x_batch, (-1, 28*28)), tf.reshape(x_perturb, (-1, 28*28)), y_batch
    else:
      yield x_batch, y_batch
      # yield tf.reshape(x_batch, (-1, 28*28)), y_batch

def main():
  k = 10
  epochs = 30
  batchsize = 96
  mnist = tf.keras.datasets.mnist
  (x_train, y_train),(x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  print('x_train:', x_train.shape)

  # mnist_generator = generate_mnist(x_train, y_train, batchsize=batchsize)
  mnist_iterator = get_iterator(x_train, y_train, batchsize=batchsize)
  x_batch, x_perturb, y_batch = next(mnist_iterator)

  print('xbatch', x_batch.shape, 'xperturb', x_perturb.shape, 'ybatch', y_batch.shape)

  model = ClusterModel(k=k)
  print('x_batch:', x_batch.shape)
  z = model(x_batch, head='main', verbose=True)
  for z_ in z:
    print('z:', z_.shape)
  z = model(x_batch, head='aux')
  for z_ in z:
    print('z:', z_.shape)
  model.summary()

  optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  plt.figure(figsize=(3,3), dpi=300)
  ax = plt.gca()
  ax.set_xlim([-1,1])
  ax.set_ylim([-1,1])
  for e in range(epochs):
    # mnist_generator = generate_mnist(x_train, y_train, batchsize=batchsize)

    mnist_iterator = get_iterator(x_train, y_train, batchsize=batchsize)
    for k, (x_batch, x_perturb, y_batch) in enumerate(mnist_iterator):
      if k % 2 == 0:
        trainhead = 'main'
      else:
        trainhead = 'aux'

      with tf.GradientTape() as tape:
        z = model(x_batch, head=trainhead)
        zp = model(x_perturb, head=trainhead)

        losses = [IID_loss(z_, zp_) for z_, zp_ in zip(z, zp)]
        loss = tf.reduce_mean(losses)
        grads = tape.gradient(loss, model.trainable_variables)

      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      if k % 100 == 0:
        zmax = tf.argmax(z[0], axis=-1).numpy()
        zpmax = tf.argmax(zp[0], axis=-1).numpy()
        print(zmax, np.unique(zmax))
        print(zpmax, np.unique(zpmax))
        print(y_batch)
        print('e: {} k: {} loss={} acc={}'.format(e, k, loss.numpy(), (zmax==zpmax).mean()))

    # Each epoch
    ztest = []
    ylabel = []
    mnist_iterator = get_iterator(x_test, y_test, batchsize=batchsize)
    # for j, (x_batch, y_batch) in enumerate(generate_mnist(x_train, y_train, repeat=1, batch_size=batch_size, perturb=False)):
    for j, (x_batch, x_perturb, y_batch) in enumerate(mnist_iterator):
      ztest.append(model(x_batch, head='main')[0])
      ylabel.append(y_batch)

    ztest = np.concatenate(ztest, axis=0)
    ylabel = np.concatenate(ylabel)
    print('ztest', ztest.shape)
    print('ylabel', ylabel.shape)
    convex_combo(ztest, ylabel, ax, 'pointcloud/{}_{}.png'.format(e, k))

if __name__ == '__main__':
  tf.enable_eager_execution()
  main()
