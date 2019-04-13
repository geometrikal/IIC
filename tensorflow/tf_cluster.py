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
from models import VGGModel, ResNetModel

def main():
  k = 10
  repeat = 4
  epochs = 30
  batchsize = 396
  mnist = tf.keras.datasets.mnist
  (x_train, y_train),(x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  print('x_train:', x_train.shape)

  mnist_iterator = get_iterator(x_train, y_train, batchsize=batchsize)
  x_batch, x_perturb, y_batch = next(mnist_iterator)

  print('xbatch', x_batch.shape, 'xperturb', x_perturb.shape, 'ybatch', y_batch.shape)

  model = ResNetModel(k=k)
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
  main_losses = []
  aux_losses = []
  for e in range(epochs):
    # mnist_generator = generate_mnist(x_train, y_train, batchsize=batchsize)

    mnist_iterator = get_iterator(x_train, y_train, batchsize=batchsize, repeat=repeat)
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

      if k % 2 == 0:
        main_losses.append(loss.numpy())
      else:
        aux_losses.append(loss.numpy())

      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      if k % 100 == 0:
        print('e: {} k: {} loss={}'.format(e, k, loss.numpy()))
        for i in range(5):
          zmax = tf.argmax(z[i], axis=-1).numpy()
          zpmax = tf.argmax(zp[i], axis=-1).numpy()
          acc = (zmax == zpmax).mean()
          print('\tacc={}'.format(acc), np.unique(zmax), np.unique(zpmax))

    # Each epoch
    ztest = {r: [] for r in range(5)}
    ylabel = []
    mnist_iterator = get_iterator(x_train, y_train, batchsize=batchsize, repeat=1)
    for j, (x_batch, x_perturb, y_batch) in enumerate(mnist_iterator):
      for i,h in enumerate(model(x_batch, head='main')):
        ztest[i].append(h)
      ylabel.append(y_batch)

    # ztest = np.concatenate(ztest, axis=0)
    ylabel = np.concatenate(ylabel)
    print('ylabel', ylabel.shape)
    for r in range(5):
      ztest[r] = np.concatenate(ztest[r], axis=0)
      print('ztest', ztest[r].shape)
      convex_combo(ztest[r], ylabel, ax, 'pointcloud/{}_{}.png'.format(e, r))

  with open('losses_main.txt', 'w+') as f:
    for l in main_losses:
      f.write('{}\n'.format(l))  

  with open('losses_aux.txt', 'w+') as f:
    for l in aux_losses:
      f.write('{}\n'.format(l))  

if __name__ == '__main__':
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  tf.enable_eager_execution(config=config)
  main()

