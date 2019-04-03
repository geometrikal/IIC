import tensorflow as tf
import numpy as np

def get_iterator(x, y, batchsize=32, repeat=4):
  generator = lambda: _make_generator(x,y)
  dataset = _make_tf_dataset(generator, batchsize, repeat)
  
  iterator = dataset.make_one_shot_iterator()

  return iterator

def _make_tf_dataset(generator, batchsize=32, repeat=4):
  # perturb the input ; skip the repeat parameter for now
  def mapped_fn(x, y):
    print('mapped fn', x.shape, y.shape)
    xp = tf.expand_dims(x, 0)
    xp = tf.image.random_crop(xp, [1, 21, 21, 1])
    xp = tf.image.resize_image_with_pad(xp, 28, 28)
    noise = tf.random.normal(stddev=0.2, shape=(28, 28, 1))
    xp = tf.clip_by_value(xp + noise, 0, 1)
    xp = tf.squeeze(xp, 0)
    return x, xp, y

  dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32),
                                           output_shapes=((28,28,1), ()))
  dataset = dataset.map(mapped_fn, num_parallel_calls=8)
  dataset = dataset.prefetch(3096)
  dataset = dataset.batch(batchsize)
  return dataset 

def _make_generator(x, y):
  assert x.shape[0] == len(y)
  idx = np.arange(x.shape[0])
  np.random.shuffle(idx)
  for k in idx:
    x_ = np.expand_dims(x[k,...], -1)
    # x_ = np.expand_dims(x_, 0)
    y_ = y[k]

    yield x_, y_