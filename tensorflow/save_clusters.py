import os
import cv2
import numpy as np

def save_images(images, clusters, dst, n=10):
  """ Save images labelled by cluster prediction

  images - np array [b, h, w, c]
  clusters - np array [b, c] (softmaxed)
  dst - str, destination
  n - int number of images per cluster to search

  """
  assert images.shape[0] == clusters.shape[0]
  images = np.array(images)
  clusters = np.argmax(clusters, axis=-1)

  if not os.path.exists(dst):
    print('creating {}'.format(dst))
    os.makedirs(dst)

  for c in np.unique(clusters):
    idx = np.squeeze(np.argwhere(clusters == c))
    print(c, idx)

    choices = np.random.choice(idx, n)
    for choice in choices:
      img = images[choice, ...]
      imgdst = os.path.join(dst, 'c{:02d}_im{:04d}_{}.jpg'.format(c, choice, np.datetime64('now')))
      print('\t', imgdst, img.shape)
      cv2.imwrite(imgdst, img[:,:,::-1]*255)