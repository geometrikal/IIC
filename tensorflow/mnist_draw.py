import cv2
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')

palette = sns.color_palette('hls', 10)
def convex_combo(clstr, label, ax, saveto):
  plt.cla()
  ax.set_xlim([-1,1])
  ax.set_ylim([-1,1])
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

  for k in range(10):
    ix = label == k
    ax.scatter(x[ix], y[ix], s=1, alpha=0.3, c=palette[k])

  plt.savefig(saveto, bbox_inches='tight')
