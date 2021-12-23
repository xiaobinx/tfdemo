import enum
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x, y), (x_val, y_val) = datasets.mnist.load_data()
# print('datasets: ', x.shape, y.shape, x_val.shape, y_val.shape)
# datasets:  (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255
y = tf.convert_to_tensor(y, dtype=tf.uint8)
y = tf.one_hot(y, depth=10)
# print(x.shape, y.shape)
# (60000, 28, 28) (60000, 10)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(200)
# for step, (x, y) in enumerate(train_dataset):
#     print(step, x.shape, y.shape)
#     # (28, 28) tf.Tensor([0. 0. 0. 0. 0. 0. 1. 0. 0. 0.], shape=(10,), dtype=float32)
#     x = tf.reshape(x, (-1, 28*28))
#     print(step, x.shape, y.shape)
#     # (200, 784)(200, 10)
#     if (step > 2):
#         break

model = keras.Sequential([
    layers.Dense(510, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10),
])

optimizer = optimizers.SGD(learning_rate=0.001)


def train_epoch(epoch):
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            x = tf.reshape(x, (-1, 28*28))
            out = model(x)
            loss = tf.reduce_sum(tf.square(out-y)) / x.shape[0]

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                print(epoch, step, loss.numpy())


def train():
    for epoch in range(30):
        train_epoch(epoch)


if __name__ == '__main__':
    train()
