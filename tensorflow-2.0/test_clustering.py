import tensorflow as tf
import tensorflow.keras as K

import cv2
from utils import losses
import MulticoreTSNE

from matplotlib import pyplot as plt

from tensorflow.keras.layers import (Conv2D, Dense, Activation,
  GlobalAveragePooling2D, GlobalMaxPooling2D , Dropout, MaxPooling2D,
  Flatten)

import numpy as np
tf.enable_eager_execution()

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.expand_dims(x_train, -1).numpy()
# y_train = y_train.numpy()
x_test = tf.expand_dims(x_test, -1).numpy()
# y_test = y_test.numpy()
print('x_train', x_train.shape)
print('x_test', x_test.shape)

def generate_training(x_train, y_train, batch_size, n_batches=None, mult=4):
  # xbatch = np.zeros((batch_size, 28, 28, 1), dtype=np.uint8)
  if n_batches is None:
    ## The whole thing, once
    n_batches = x_train.shape[0] // batch_size

  for k in range(n_batches):
    idx = np.random.choice(range(x_train.shape[0]), batch_size, replace=False)
    xbatch = x_train[idx, ...]
    ybatch = y_train[idx]

    xbatch = np.concatenate([xbatch]*mult, axis=0)
    ybatch = np.concatenate([ybatch]*mult, axis=0)
    yield tf.cast(xbatch / 255., tf.float32), ybatch

def convex_combo(clstr, label, ax, saveto):
  plt.cla()
  ax.set_xlim([-1,1])
  ax.set_ylim([-1,1])
  def get_coord(probs, num_classes=10):
    # computes coordinate for 1 sample based on probability distribution over c
    coords_total = np.zeros(2, dtype=np.float32)
    probs_sum = probs.sum()
    fst_angle = 0.
    for c in range(num_classes):
      # compute x, y coordinates
      coords = np.ones(2) * 2 * np.pi * (float(c) / num_classes) + fst_angle
      coords[0] = np.sin(coords[0])
      coords[1] = np.cos(coords[1])
      coords_total += (probs[c] / probs_sum) * coords
    return coords_total

  xy = np.stack([get_coord(c) for c in clstr], axis=1)
  print(xy.shape)
  x = xy[0,:]
  y = xy[1,:]

  for k in range(10):
    ix = label == k
    ax.scatter(x[ix], y[ix], s=1, alpha=0.3)

  plt.savefig(saveto, bbox_inches='tight')

class ClusteringNHead(tf.keras.Model):
  def __init__(self, k_gt=10, n_subheads=5, n_cocluster=5, max_cocluster=30):
    super(ClusteringNHead, self).__init__()

    self.conv2d_1 = Conv2D(64, (7,7), (3,3), activation=tf.nn.relu,  name='conv1')
    self.conv2d_2 = Conv2D(128, (3,3), (2,2), activation=tf.nn.relu,  name='conv2')
    self.conv2d_21 = Conv2D(128, (3,3), (1,1), activation=tf.nn.relu,  name='conv2', padding='SAME')
    self.conv2d_3 = Conv2D(512, (2,2), (1,1), activation=tf.nn.relu, name='conv3')
    # self.pooling  = GlobalMaxPooling2D()
    self.squish   = Flatten()

    self.fc2      = Dense(512, activation=tf.nn.relu, name='fc')

    self.target = Dense(k_gt, activation=tf.nn.softmax, name='target')
    self.subheads = []
    for k in range(n_subheads):
      self.subheads.append(Dense(k_gt, activation=tf.nn.softmax, name='sh{}'.format(k)))

    # self.cocluster = []
    # for k in range(n_cocluster):
    #   self.cocluster.append(Dense(max_cocluster, activation=tf.nn.softmax, name='coclust'))
    # self.auxtargets = []
    # for k in np.linspace(10, max_cocluster, n_heads, dtype=np.int):
    #   self.auxtargets.append(Dense(k, activation=tf.nn.softmax))

  def call(self, x, verbose=False):
    net = self.conv2d_1(x)
    if verbose: print(net.shape)
    net = self.conv2d_2(net)
    if verbose: print(net.shape)
    net = self.conv2d_21(net)
    if verbose: print(net.shape)
    net = self.conv2d_3(net)
    if verbose: print(net.shape)
    net = self.squish(net)
    if verbose: print(net.shape)

    net = self.fc2(net)
    if verbose: print(net.shape)

    target = self.target(net)
    subhead_targets = []
    for l in self.subheads:
      subhead_targets.append(l(net))
    cocluster = []
    # for l in self.cocluster:
    #   cocluster.append(l(net))

    return target, subhead_targets, cocluster

# model = K.models.Sequential([
#   Conv2D(32, (7,7), (3,3), input_shape=x_train.shape[1:]),
#   Activation('relu'),
#   Conv2D(64, (5,5), (2,2)),
#   Activation('relu'),
#   Conv2D(128, (2,2), (1,1), padding='SAME'),
#   Activation('relu'),
#   GlobalAveragePooling2D(),
#   Dense(10),
#   Activation('softmax')
# ])

def perturb_images(imgs, stddev=0.15):
  b = imgs.shape[0]
  x_perturb = tf.identity(imgs)
  # x_perturb = tf.image.random_crop(x_perturb, [b, 22, 22, 1])
  # x_perturb = tf.image.resize_image_with_crop_or_pad(x_perturb, 28, 28)

  noise = tf.random.normal(stddev=stddev, shape=imgs.shape)
  x_perturb += noise
  x_perturb = tf.clip_by_value(x_perturb, 0, 1)

  return x_perturb

b = 128
# train_gen = generate_training(x_train)
iterator = generate_training(x_train, y_train, b, n_batches=25000)

x_batch, y_batch = next(iterator)
print('x_batch gen', x_batch.shape)
print('y_batch gen', y_batch.shape)
x_perturb = perturb_images(x_batch)
print('x_perturb', x_perturb.shape)

# Debug the image perturbations
for k in range(10):
  img_o = np.squeeze(x_batch[k, ...]) * 255.
  img_p = np.squeeze(x_perturb[k, ...]) * 255.
  cv2.imwrite('img/{}_o.png'.format(k), img_o)
  cv2.imwrite('img/{}_p.png'.format(k), img_p)

model = ClusteringNHead()
z_batch, z_sh, z_coclust = model(x_batch, verbose=True)
print('z_batch', z_batch.shape)
print('z_subheads', [z_sh_.shape for z_sh_ in z_sh])
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

model.summary()
plt.figure(figsize=(3,3), dpi=300)
ax = plt.gca()
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
trn_switch = -1
for k, (x_batch, y_batch) in enumerate(iterator):
  x_perturb = perturb_images(x_batch)
  with tf.GradientTape() as tape:
    z_batch, z_sh, z_aux = model(x_batch)
    z_perturb_batch, z_p_sh, z_p_aux = model(x_perturb)

    iloss = losses.IIC(z_batch, z_perturb_batch, lam=1)
    subhead_loss = tf.reduce_mean([losses.IIC(zsh_, zpsh_, lam=0.01) for zsh_, zpsh_ in zip(z_sh, z_p_sh)])
    # aux_loss = tf.reduce_mean([losses.IIC(auxz_, auxzp_, lam=1.0) for auxz_, auxzp_ in zip(z_aux, z_p_aux)])
    # aux_loss = losses.IIC(z_aux, z_p_aux, lam=1.)

    loss = tf.reduce_mean([iloss, subhead_loss])
    # if trn_switch==-1:
    #   loss = tf.reduce_mean([iloss, subhead_loss])
    # else:
    #   loss = aux_loss

    grads = tape.gradient(loss, model.trainable_variables)

  if k % 50 == 0:
    zmax = tf.argmax(z_batch, axis=-1)
    zpmax = tf.argmax(z_perturb_batch, axis=-1)
    print(z_batch[0,:])
    print(np.unique(zmax))
    print(np.unique(zpmax))
    print(y_batch)
    print('k:', k, 'Loss', loss.numpy(), 'acc', np.mean((zmax.numpy() == zpmax.numpy())))
    # print([np.mean(g) for g in grads if g is not None])

    # for m, v in enumerate(model.trainable_variables):
    #   print(v.name)
    #   print(grads[m].numpy().ravel()[:20])

    print()

  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  if k % 1000 == 0:
    trn_switch *= -1
    test_clstr = []
    test_label = []
    test_iterator = generate_training(x_test, y_test, b, n_batches=None)
    for x_batch, y_batch in test_iterator:
      test_clstr.append(model(x_batch)[0])
      test_label.append(y_batch)

    test_clstr = tf.concat(test_clstr, axis=0).numpy()
    test_label = np.concatenate(test_label)
    print(test_clstr.shape)
    print(test_label.shape)

    # z = MulticoreTSNE.MulticoreTSNE(n_jobs=-1).fit_transform(test_clstr.numpy())
    convex_combo(test_clstr, test_label, ax=ax, saveto='pointcloud/{:06d}.png'.format(k))
