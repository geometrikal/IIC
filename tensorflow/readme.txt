Core components needed to perform IIC, implemented in tensorflow with heavy reference to the pytorch implimentations.

package requirements:
scipy >= 1.2 (scipy introduced a softmax function I use)
tensorflow (version with eager execution; tested 1.12 & 1.13 (CPU))
opencv-python

To test tensorflow and pytorch loss functions:

```
(pytorch) $ python IID_losses.py 
Independent vectors:
-8.633034206059919e-05
The same vector:
-0.0036585953542896178

(tf1.13) $ python IID_losses_tf.py 
Independent vectors:
-8.633034206018125e-05
The same vector:
-0.0036585953542899734
```


To run clustering:
This trains an MNIST MLP with the IIC objective.

Only some additive noise is used for the image perturb.

It wants a directory called `pointcloud` for drawing the mnist point clouds.

```
python tf_cluster.py
```
