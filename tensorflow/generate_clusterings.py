import numpy as np
from scipy.special import softmax

def genclust(b = 64, k = 10, y=None, seed=0):
  np.random.seed(seed)

  # Generate class assignment if none is given
  if y is None:
    y = np.random.choice(range(k), b, replace=True)
    y = np.eye(k)[y]  # onehot
  else:
    assert y.shape[-1] == k

  # Noise it up
  noise = np.around(np.random.randn(*y.shape), decimals=5)
  y *= noise

  return softmax(y, axis=-1)

if __name__ == '__main__':
  b = 64
  k = 10

  y_11 = genclust(b=b, k=k, y=None, seed=1)
  y_12 = genclust(b=b, k=k, y=None, seed=1)
  y_2 = genclust(b=b, k=k, y=None, seed=2)

  y_11 = np.argmax(y_11, axis=-1)
  y_12 = np.argmax(y_12, axis=-1)
  y_2  = np.argmax(y_2, axis=-1)

  assert (y_11 == y_12).mean() == 1
  assert (y_11 == y_2).mean() < 1
