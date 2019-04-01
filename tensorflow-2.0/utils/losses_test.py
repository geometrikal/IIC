import tensorflow as tf


tf.enable_eager_execution()

def main():
  # Test IIC loss
  from losses import IIC

  b = 8
  c = 10

  z = tf.nn.softmax(tf.random.normal((b,c)), axis=-1)
  zp = tf.nn.softmax(tf.random.normal((b,c)), axis=-1)
  print(z)
  print(zp)

  iic = IIC(z, zp)
  
  print(iic)

if __name__ == '__main__':
  main()