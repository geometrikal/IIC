import tensorflow as tf
import numpy as np

from IID_losses_tf import IID_loss

from tensorflow.keras.layers import (
  Conv2D, Dense, MaxPooling2D
)

class ClusterModel(tf.keras.Model):
  def __init__(self, k=10):
    super(ClusterModel, self).__init__()
    self.mylayers = []
    self.mylayers += [Dense(1024, activation=tf.nn.relu)]
    self.mylayers += [Dense(512, activation=tf.nn.relu)]

    self.mylayers += [Dense(k, activation=tf.nn.softmax, name='k')]

  def call(self, x):
    for l in self.mylayers:
      x = l(x)

    return x

def perturb_x(x_in, stddev=0.2):
  # Perturb with additive noise
  noise = tf.random.normal(stddev=stddev, shape=x_in.shape)
  x_in = tf.clip_by_value(x_in + noise, 0, 1)
  return x_in

def generate_mnist(x_src, y_src, batch_size=32, repeat=3):
  n_batches = x_src.shape[0] / batch_size
  indices = np.random.choice(np.arange(x_src.shape[0]), int(n_batches*batch_size), replace=False)
  for idx in np.split(indices, n_batches):
    x_batch = tf.concat([x_src[idx, ...]] * repeat, axis=0)
    x_batch = tf.cast(x_batch, tf.float32)
    x_perturb = perturb_x(x_batch)
    yield x_batch, x_perturb

def main():
  k = 10
  epochs = 10
  batch_size = 32
  mnist = tf.keras.datasets.mnist
  (x_train, y_train),(x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
  x_train = x_train.reshape(-1, 28*28)
  x_test = x_test.reshape(-1, 28*28)
  print('x_train:', x_train.shape)

  model = ClusterModel(k=k)
  mnist_generator = generate_mnist(x_train, y_train, batch_size=batch_size)
  x_batch, x_perturb = next(mnist_generator)
  print('x_batch:', x_batch.shape)
  z = model(x_batch)
  print('z:', z.shape)
  model.summary()

  optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  for e in range(epochs):
    mnist_generator = generate_mnist(x_train, y_train, batch_size=batch_size)

    for k, (x_batch, x_perturb) in enumerate(mnist_generator):
      with tf.GradientTape() as tape:
        z = model(x_batch)
        zp = model(x_perturb)

        loss = IID_loss(z, zp)
        grads = tape.gradient(loss, model.trainable_variables)

      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      if k % 50 == 0:
        zmax = tf.argmax(z, axis=-1).numpy()
        zpmax = tf.argmax(zp, axis=-1).numpy()
        print(zmax)
        print(zpmax)
        print('e: {} k: {} loss={} acc={}'.format(e, k, loss.numpy(), (zmax==zpmax).mean()))

if __name__ == '__main__':
  tf.enable_eager_execution()
  main()
