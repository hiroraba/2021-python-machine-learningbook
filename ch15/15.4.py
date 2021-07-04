""" 15.4 - CNNを使って顔画像から性別を予測する """

import tensorflow as tf
import tensorflow_datasets as tfds

celeba_bldr = tfds.builder('celeb_a')
celeba_bldr.download_and_prepare()
celeba = celeba_bldr.as_dataset(shuffle_files=False)

celeba_train = celeba['train'] # 訓練データ
celeba_valid = celeba['validation'] # 検証データ
celeba_test = celeba['test'] # テストデータ

celeba_train = celeba_train.take(160000) 
celeba_valid = celeba_valid.take(1000)

# イテレータごとに画像をランダムに加工する(データセットが限られた状況において、過学習を抑制することができる)
def preprocess(example, size=(64, 64), mode='train'):
    image = example['image']
    label = example['attributes']['Male']
    if mode == 'train':
        image_cropped = tf.image.random_crop(
            image, size=(178, 178, 3))
        image_resized = tf.image.resize(
            image_cropped, size=size)
        image_flip = tf.image.random_flip_left_right(
            image_resized)
        return (image_flip/255.0, tf.cast(label, tf.int32))
    
    else: # 訓練データでない場合はランダムでなく中央のくり抜き
        image_cropped = tf.image.crop_to_bounding_box(
            image, offset_height=20, offset_width=0,
            target_height=178, target_width=178)
        image_resized = tf.image.resize(
            image_cropped, size=size)
        return (image_resized/255.0, tf.cast(label, tf.int32))

# データの準備
import numpy as np
BATCH_SIZE = 32
BUFFER_SIZE = 10000
IMAGE_SIZE = (64,64)
steps_per_epoch = np.ceil(160000/BATCH_SIZE)

ds_train = celeba_train.map(lambda x: preprocess(x, size=IMAGE_SIZE, mode='train'))
ds_train = ds_train.shuffle(BUFFER_SIZE).repeat()
ds_train = ds_train.batch(BATCH_SIZE)

ds_valid = celeba_valid.map(lambda x: preprocess(x, size=IMAGE_SIZE, mode='eval'))
ds_valid = ds_valid.batch(BATCH_SIZE)

# モデルの作成
model = tf.keras.Sequential([
    # 1つ目の畳込み、プーリング、ドロップアウト層
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(rate=0.5),
    # 2つ目の畳込み、プーリング、ドロップアウト層
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(rate=0.5),
    # 3つ目の畳込み、プーリング層
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    # 4つ目の畳込み層
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
])

# この層をそのまま全結合すると、入力ユニットが(8*8*256)16384個になる
# model.add(tf.keras.layers.Flatten())
# print(model.compute_output_shape(input_shape=(None, 64, 64, 3)))
# それぞれの特徴量の平均を取り、入力ユニットを256に削減する
model.add(tf.keras.layers.GlobalAveragePooling2D())
print(model.compute_output_shape(input_shape=(None, 64, 64, 3)))

# 最後に出力ユニットを1つだけ出力する
model.add(tf.keras.layers.Dense(1, activation=None))

tf.random.set_seed(1)
model.build(input_shape=(None, 64, 64, 3))
print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(ds_train, 
                    validation_data=ds_valid,
                    epochs=30,
                    steps_per_epoch=steps_per_epoch)

hist = history.history

import matplotlib.pyplot as plt
x_arr = np.arange(len(hist['loss'])) + 1
fig = plt.figure(figsize=(12, 4))

ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist['loss'], '-o', label='Train Loss')
ax.plot(x_arr, hist['val_loss'], '--<', label='Validation Loss')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')

ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist['accuracy'], '-o', label='Train acc.')
ax.plot(x_arr, hist['val_accuracy'], '--<', label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')

plt.show()