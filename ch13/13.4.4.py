""" 13.4.4 - Irisデータセットの花の品種を分類する多層パーセプトロンを構築する """

# データセットを取得
from os import name
from tensorflow.python.ops.gen_array_ops import size
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

iris, iris_info = tfds.load('iris', with_info=True)
print(iris_info)

tf.random.set_seed(1)
ds_orig = iris['train']

# 訓練データとテストデータが混じらないようにシャッフルする
ds_orig = ds_orig.shuffle(150, reshuffle_each_iteration=False)
ds_train_orig = ds_orig.take(100)
ds_test = ds_orig.skip(100)

# 特徴量とラベルにタプルに分ける
ds_train_orig = ds_train_orig.map(lambda x: (x['features'], x['label']))
ds_test = ds_test.map(lambda x: (x['features'], x['label']))

# tf.keras.sequentialクラスを使って層を積み上げることでニューラルネットワークが構築できる
iris_model = tf.keras.Sequential([
    # Dense層: 全結合層または線形層とも呼ばれる
    # input_shape: 4つの入力層を受け取る
    tf.keras.layers.Dense(16, activation='sigmoid', name='fc1', input_shape=(4, )),
    # 多クラス分類をサポートするソフトマックス関数を利用
    tf.keras.layers.Dense(3, name='fc2', activation='softmax')
])
iris_model.summary()

# 損失関数、オプティマイザ、評価指標を指定する
iris_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

num_epochs = 100
training_size = 100
batch_size = 2
steps_per_epoch = np.ceil(training_size / batch_size)

ds_train = ds_train_orig.shuffle(buffer_size=training_size)
ds_train = ds_train.repeat()
print(ds_train)
ds_train = ds_train.batch(batch_size=batch_size)
ds_train = ds_train.prefetch(buffer_size=1000)

history = iris_model.fit(ds_train, epochs=num_epochs, steps_per_epoch=steps_per_epoch, verbose=0)

hist = history.history
fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(1, 2, 1)
ax.plot(hist['loss'], lw=3)
ax.set_title('Training loss', size=15)
ax.set_xlabel('Epoch', size=15)

ax = fig.add_subplot(1, 2, 2)
ax.plot(hist['accuracy'], lw=3)
ax.set_title('Training accuracy', size=15)
ax.set_xlabel('Epoch', size=15)

plt.tight_layout()
plt.show()

""" 13.4.5 - 訓練したモデルをテストデータセットで評価する"""
results = iris_model.evaluate(ds_test.batch(50), verbose=0)
print('Test Loss: {:.4f} Test Acc.: {:.4f}'.format(*results))