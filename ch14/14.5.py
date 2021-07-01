""" 14.5 - Keras APIを使って一般的なアーキテクチャの実装を単純化する """
import tensorflow as tf
from tensorflow.keras import activations, metrics, optimizers

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=16, activation='relu'))
model.add(tf.keras.layers.Dense(units=32, activation='relu'))

model.build(input_shape=(None, 4))
model.summary()

for v in model.variables:
    print('{:20s}'.format(v.name), v.trainable, v.shape)
    
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(
    units=16,
    activation=tf.keras.activations.relu,
    kernel_initializer=tf.keras.initializers.glorot_uniform(),
    bias_initializer=tf.keras.initializers.constant(2.0)
    ))
model.add(tf.keras.layers.Dense(
    units=32,
    activation=tf.keras.activations.sigmoid,
    kernel_regularizer=tf.keras.regularizers.l1))
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), # 確率的勾配降下法
    loss=tf.keras.losses.BinaryCrossentropy(), # 交差エントロピー
    metrics=[tf.keras.metrics.Accuracy(),
             tf.keras.metrics.Precision(),
             tf.keras.metrics.Recall(),])

""" 14.5.1 - XOR分類問題を解く """
import matplotlib.pyplot as plt
import numpy as np

# データの作成
tf.random.set_seed(1)
np.random.seed(1)

x = np.random.uniform(low=-1, high=1, size=(200, 2))
y = np.ones(len(x))
y[x[:, 0] * x[:, 1] < 0] = 0

x_train = x[:100, :]
y_train = y[:100]
x_valid = x[100:, :]
y_valid = y[100:]

plt.plot(x[y==0, 0], x[y==0, 1], 'o', alpha=0.75, markersize=10)
plt.plot(x[y==1, 0], x[y==1, 1], '<', alpha=0.75, markersize=10)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.show()

# モデルの作成
# 隠れ層のない線形モデルを作成する
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=(2,), activation='sigmoid'))
model.summary()
model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

hist = model.fit(x_train, y_train, 
                 validation_data=(x_valid, y_valid),
                 epochs=200, batch_size=2, verbose=0)

from mlxtend.plotting import plot_decision_regions
history = hist.history

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 3, 1)
ax.plot(history['loss'], lw=3)
ax.plot(history['val_loss'], lw=3)
plt.legend(['Train loss', 'Validation loss'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(1, 3, 2)
ax.plot(history['binary_accuracy'], lw=3)
ax.plot(history['val_binary_accuracy'], lw=3)
plt.legend(['Train Acc.', 'Validation Acc'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(1, 3, 3)
plot_decision_regions(x_valid, y_valid.astype(np.integer), clf=model)
ax.set_xlabel(r'$x_1$', size=15)
ax.xaxis.set_label_coords(1, -0.025)
ax.set_ylabel(r'$x_2$', size=15)
ax.yaxis.set_label_coords(-0.025, 1)

plt.show()

tf.random.set_seed(1)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=4, input_shape=(2, ), activation='relu'))
model.add(tf.keras.layers.Dense(units=4, activation='relu'))
model.add(tf.keras.layers.Dense(units=4, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])


hist = model.fit(x_train, y_train, 
                 validation_data=(x_valid, y_valid),
                 epochs=200, batch_size=2, verbose=0)

history = hist.history

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 3, 1)
ax.plot(history['loss'], lw=3)
ax.plot(history['val_loss'], lw=3)
plt.legend(['Train loss', 'Validation loss'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(1, 3, 2)
ax.plot(history['binary_accuracy'], lw=3)
ax.plot(history['val_binary_accuracy'], lw=3)
plt.legend(['Train Acc.', 'Validation Acc'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(1, 3, 3)
plot_decision_regions(x_valid, y_valid.astype(np.integer), clf=model)
ax.set_xlabel(r'$x_1$', size=15)
ax.xaxis.set_label_coords(1, -0.025)
ax.set_ylabel(r'$x_2$', size=15)
ax.yaxis.set_label_coords(-0.025, 1)

plt.show()
