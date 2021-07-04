""" 15.3 - TensorFlowを使ってディープ畳み込みニューラルネットワークを実装する """

import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.ops.gen_math_ops import mod
import tensorflow_datasets as tfds

## データを準備する
mnist_bldr = tfds.builder('mnist')
mnist_bldr.download_and_prepare()
datasets = mnist_bldr.as_dataset(shuffle_files=False)

## 訓練データセットとテストデータセットを取得
mnist_train_org = datasets['train']
mnist_test_org = datasets['test']

## 訓練データセットから検証データセットを作る
BUFFER_SIZE = 10000
BATCH_SIZE = 64
NUM_EPOCHS = 20

mnist_train = mnist_train_org.map(
    lambda item: (tf.cast(item['image'], tf.float32) / 255.0,
                  tf.cast(item['label'], tf.int32)))

mnist_test = mnist_test_org.map(
    lambda item: (tf.cast(item['image'], tf.float32) / 255.0,
                  tf.cast(item['label'], tf.int32)))

tf.random.set_seed(1)
mnist_train = mnist_train.shuffle(buffer_size=BUFFER_SIZE, reshuffle_each_iteration=False)
mnist_valid = mnist_train.take(10000).batch(BATCH_SIZE)
mnist_train = mnist_train.skip(10000).batch(BATCH_SIZE)

# KerasのCNNの構築
model = tf.keras.Sequential()

# 畳み込み
model.add(tf.keras.layers.Conv2D(filters=32,  # フィルタ数
                                 kernel_size=(5, 5), # カーネルサイズ
                                 strides=(1,1), # ストライド
                                 padding='same', # パディング
                                 data_format='channels_last', # データフォーマット
                                 name='Conv_1',
                                 activation='relu'))

# Maxプーリング
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), name='pool_1'))

# 畳み込み
model.add(tf.keras.layers.Conv2D(filters=64,  # フィルタ数
                                 kernel_size=(5, 5), # カーネルサイズ
                                 strides=(1,1), # ストライド
                                 padding='same', # パディング
                                 name='Conv_2',
                                 activation='relu'))

# Maxプーリング
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), name='pool_2'))


# 出力サイズを計算
# 1次元目はバッチ次元のため省略してもよい
print(model.compute_output_shape(input_shape=(16, 28, 28, 1)))

# 全結合層（前までの出力を平坦化）
model.add(tf.keras.layers.Flatten())
print(model.compute_output_shape(input_shape=(16, 28, 28, 1)))

model.add(tf.keras.layers.Dense(units=1024, name='fc_1', activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.5))

# ソフトマックスで所属確率を取得
model.add(tf.keras.layers.Dense(units=10, name='fc_2', activation='softmax'))
print(model.compute_output_shape(input_shape=(16, 28, 28, 1)))

tf.random.set_seed(1)
model.build(input_shape=(None, 28, 28, 1))
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(mnist_train,
                    epochs=NUM_EPOCHS,
                    validation_data=mnist_valid,
                    shuffle=True)