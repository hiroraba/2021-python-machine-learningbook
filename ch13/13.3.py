""" 13.3 - tf.dataを使った入力パイプラインの構築: Tensorflow Dataset API"""
import enum
from posixpath import basename
from sys import path
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import size

a = [1.2, 3.4, 7.5, 4.1, 5.0, 1.0]

# 値のリストからデータセットを作成
ds = tf.data.Dataset.from_tensor_slices(a)
print(ds)

# データセットを処理するにはfor文で処理
for item in ds:
    print(item)
    
# サイズが3のバッチを作成したい場合
ds_batch = ds.batch(3)

for i, elem in enumerate(ds_batch, 1):
    print('batch {}:'.format(i), elem.numpy())
    
# テンソルを結合する
tf.random.set_seed(1)

t_x = tf.random.uniform([4, 3], dtype=tf.float32) # 特徴量
t_y = tf.range(4) # クラスラベル

ds_x = tf.data.Dataset.from_tensor_slices(t_x)
ds_y = tf.data.Dataset.from_tensor_slices(t_y)

ds_joint = tf.data.Dataset.zip((ds_x, ds_y))
for example in ds_joint:
    print(' x:', example[0].numpy(), ' y:', example[1].numpy())


# datasetに変換しない方法
ds_joint = tf.data.Dataset.from_tensor_slices((t_x, t_y))
for example in ds_joint:
    print(' x:', example[0].numpy(), ' y:', example[1].numpy())

# 特徴量をスケーリングする
ds_trans = ds_joint.map(lambda x, y: (x*2 - 1.0, y))
for example in ds_trans:
    print(' x:', example[0].numpy(), ' y:', example[1].numpy())

# データをシャッフルする
tf.random.set_seed(1)
# buffer_size: シャッフルする要素の個数
ds = ds_joint.shuffle(buffer_size=len(t_x))
print('shuffle:')
for example in ds:
    print(' x:', example[0].numpy(), ' y:', example[1].numpy())

# 結合データをバッチに分割する
ds = ds_joint.batch(batch_size=3, drop_remainder=False)
batch_x, batch_y = next(iter(ds))
print('Batch-X:\n', batch_x.numpy())

""" 13.3.4 - ローカルディスク上のファイルからデータセットを作成する """
import pathlib

# 画像のファイルリストを作成する
imgdir_path = pathlib.Path('cat_dog_images')
file_list = sorted([str(path) for path in imgdir_path.glob('*.jpg')])
print(file_list)

# 画像を可視化する
import matplotlib.pyplot as plt
import os

fig = plt.figure(figsize=(10, 5))
for i, file in enumerate(file_list):
    img_raw = tf.io.read_file(file)
    img = tf.image.decode_image(img_raw)
    print('Image shape:', img.shape)
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(img)
    ax.set_title(os.path.basename(file), size=15)
    
plt.tight_layout()
plt.show()

# 犬画像に1,猫画像に0を割り当てる
labels = [1 if 'dog' in os.path.basename(file) else 0 for file in file_list]

ds_files_labels = tf.data.Dataset.from_tensor_slices((file_list, labels))
for item in ds_files_labels:
    print(item[0].numpy(), item[1].numpy())
    
# データセットのサイズが異なるので、80 * 120のサイズに変換する

def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image /= 255.0
    return image, label
    
img_width, img_height = 120, 80
ds_images_labels = ds_files_labels.map(load_and_preprocess)

fig = plt.figure(figsize=(10, 5))

for i, example in enumerate(ds_images_labels):
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(example[0])
    ax.set_title('{}'.format(example[1].numpy()), size=15)
    
plt.tight_layout()
plt.show()

""" 13.3.5 - tensorflow_datasetsライブラリからデータセットを取り出す """
import tensorflow_datasets as tfds

# データセットの数をプリント
print(len(tfds.list_builders()))

# 先頭5つをプリント
print(tfds.list_builders()[:5])

# セレブデータセットを読み込み
celeba_bldr = tfds.builder('celeb_a')

# 全体の特徴量をプリント
print(celeba_bldr.info.features)

# データセットをダウンロード
# celeba_bldr.download_and_prepare()

# データセットをインスタンス化
# datasets = celeba_bldr.as_dataset(shuffle_files=False)
# print(datasets.keys)

# MNISTデータセットを取得
mnist, mnist_info = tfds.load('mnist', with_info=True, shuffle_files=False)
print(mnist_info)

ds_train = mnist['train']
# 特徴量とラベルに分ける
ds_train = ds_train.map(lambda item: (item['image'], item['label']))
# バッチに分割
ds_train = ds_train.batch(10)
# 1つ目のバッチを取り出す
batch = next(iter(ds_train))
print(batch[0].shape, batch[1])

fig = plt.figure(figsize=(15, 6))

for i, (image, label) in enumerate(zip(batch[0], batch[1])):
    ax = fig.add_subplot(2, 5, i+1)
    ax.imshow(image[:, :, 0], cmap='gray_r')
    ax.set_title('{}'.format(label), size=15)
    
plt.show()