import cv2
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')

palette = sns.color_palette('hls', 10)
def convex_combo(clstr, label, ax, saveto):
  plt.cla()
  ax.set_xlim([-1.1,1.1])
  ax.set_ylim([-1.1,1.1])
  def get_coord(probs, num_classes=10):
    # computes coordinate for 1 sample based on probability distribution over c
    coords_total = np.zeros(2, dtype=np.float32)
    probs_sum = np.sum(probs)
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
  print('x', x.shape)
  print('y', y.shape)
  print('label', label.shape)

  sel = np.random.binomial(1,0.15,size=len(x))
  print('sel', sel.shape)

  x     =     np.squeeze(x[np.argwhere(sel)])
  y     =     np.squeeze(y[np.argwhere(sel)])
  label = np.squeeze(label[np.argwhere(sel)])
  print('selected', sel.sum(), len(x))
  print('x', x.shape)
  print('y', y.shape)
  print('label', label.shape)

  x = x + np.random.normal(0, 0.05, size=len(x))
  y = y + np.random.normal(0, 0.05, size=len(y))

  for k in range(10):
    ix = np.squeeze(label == k)
    print('\t{}:{}'.format(k, ix.sum()))
    ax.scatter(x[ix], y[ix], s=1, alpha=0.5, c=[palette[k]] * ix.sum(), label='{}'.format(k))

  plt.legend(bbox_to_anchor=(1,1))
  plt.savefig(saveto, bbox_inches='tight')
