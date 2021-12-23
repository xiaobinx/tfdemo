import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


a = tf.range(5)
b = tf.Variable(a)  # 可求导
b.name  #  isinstance(b, tf.Variable)
b.trainable  #  True
isinstance(b, tf.Tensor)  #  False
isinstance(b, tf.Variable)  #  True
tf.is_tensor(b)
b.numpy()
print(a)
print(b)
print(b.name)
print(b.trainable)
print(isinstance(b, tf.Tensor))
print(isinstance(b, tf.Variable))
print(tf.is_tensor(b))