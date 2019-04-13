import tensorflow as tf
import numpy as np

def get_iterator(x, y, batchsize=32, repeat=4):
  generator = lambda: _make_generator(x, y, repeat=repeat)
  dataset = _make_tf_dataset(generator, batchsize)
  iterator = dataset.make_one_shot_iterator()
  return iterator

def _make_tf_dataset(generator, batchsize=32):

  def mapped_fn(x, y):
    def perturb(x):
      xp = tf.image.random_crop(x, [1, 22, 22, 1])
      xp = tf.image.resize_image_with_pad(xp, 28, 28)
      noise = tf.random.normal(mean=1., stddev=0.1, shape=(28, 28, 1))
      xp = tf.clip_by_value(xp * noise, 0, 1)
      # xp = tf.image.random_flip_left_right(xp)
      # xp = tf.image.random_flip_up_down(xp)

      # Rotation
      # nrot = tf.round(tf.random.uniform(shape=(1,), minval=0, maxval=4))
      # xp = tf.image.rot90(xp, k=tf.squeeze(tf.cast(nrot, tf.int32)))

      xp = tf.squeeze(xp, 0)
      return xp

    ## add noise to x also
    # noise = tf.random.normal(stddev=0.15, shape=(28,28,1))
    # x = tf.clip_by_value(x + noise, 0, 1)
    xp = perturb(tf.expand_dims(x, 0))
    return x, xp, y

  dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.uint8),
                                           output_shapes=((28,28,1), ()))
  dataset = dataset.map(mapped_fn, num_parallel_calls=8)
  dataset = dataset.prefetch(4096)
  dataset = dataset.batch(batchsize)
  return dataset 

def _make_generator(x, y, repeat=1):
  assert x.shape[0] == len(y)
  idx = np.arange(x.shape[0])
  np.random.shuffle(idx)
  # apply shuffled repeats yielded sequentially
  idx = idx.repeat(repeat)
  for k in idx:
    x_ = np.expand_dims(x[k,...], -1)
    # x_ = np.expand_dims(x_, 0)
    y_ = y[k]

    yield x_, y_
