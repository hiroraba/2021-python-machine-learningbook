from numpy.core.fromnumeric import transpose
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.array_ops import zeros

# numpyからテンソルを作成
# テンソル: 入力と出力を参照するシンボリックハンドル
a = np.array([1, 2, 3], dtype=np.int32)
b = [4, 5, 6]

t_a = tf.convert_to_tensor(a)
t_b = tf.convert_to_tensor(b)

t_ones = tf.ones((2, 3))
print(t_ones.shape)
print(t_ones.numpy())

# 転置
t = tf.random.uniform(shape=(3, 5))
t_tr = tf.transpose(t)
print(t.shape, ' --> ', t_tr.shape)

# 形状変更
t = tf.zeros((30,))
t_reshape = tf.reshape(t, shape=(5, 6))
print(t_reshape.shape)

# 不要な次元の削除
t = tf.zeros((1, 2, 1, 4, 1))
t_sqz = tf.squeeze(t, axis=(2, 4))
print(t.shape, ' --> ', t_sqz.shape)

# テンソルでの算術演算

tf.random.set_seed(1)

# 一様分布のテンソル
t1 = tf.random.uniform(shape=(5, 2), minval=-1.0, maxval=1.0)

# 標準正規分布のテンソル
t2 = tf.random.normal(shape=(5, 2), mean=0.0, stddev=1.0)

# 要素ごとの積を求める
t3 = tf.multiply(t1, t2).numpy()
print(t3)

# 平均を求める
t4 = tf.math.reduce_mean(t1, axis=0)
print(t4)

# 行列の積
t5 = tf.linalg.matmul(t1, t2, transpose_b=True)
print(t5) #(5, 5)

t6 = tf.linalg.matmul(t1, t2, transpose_a=True)
print(t6) # (2, 2)