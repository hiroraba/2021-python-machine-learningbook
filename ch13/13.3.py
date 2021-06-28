""" 13.3 - tf.dataを使った入力パイプラインの構築: Tensorflow Dataset API"""
import enum
import tensorflow as tf

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
