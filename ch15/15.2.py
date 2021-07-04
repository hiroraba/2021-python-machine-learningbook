""" 15.2.2 - ドロップアウトによるニューラルネットワークの正則化 """
from math import log
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.ops.parallel_for.control_flow_ops import _loop_fn_has_config

# 畳み込みネットワークでl2正則化する方法
conv_layer = keras.layers.Conv2D(filters=16,
                                 kernel_size=(3, 3),
                                 kernel_regularizer=keras.regularizers.l2(0.001))

# 全結合ネットワークでl2正則化
fc_layer = keras.layers.Dense(units=16,
                              kernel_regularizer=keras.regularizers.l2(0.001))


# BinaryCrossEntropy(二値分類)
## ロジットを利用: from_logits=True
## 所属確率を利用: from_logits=False
bce_probas = keras.losses.BinaryCrossentropy(from_logits=False)
bce_logits = keras.losses.BinaryCrossentropy(from_logits=True)

logits = tf.constant([0.8])
probas = tf.keras.activations.sigmoid(logits)
tf.print('BCE (w Probas): {:.4f}'.format(bce_probas(y_true=[1], y_pred=probas)),
         '(w logits): {:.4f}'.format(bce_logits(y_true=[1], y_pred=logits)))

# CategoricalCrossentropy(多クラス分類)
cce_probas = keras.losses.CategoricalCrossentropy(from_logits=False)
cce_logits = keras.losses.CategoricalCrossentropy(from_logits=True)
logits = tf.constant([[1.5, 0.8, 2.1]])
probas = tf.keras.activations.softmax(logits)
tf.print('CCE (w Probas): {:.4f}'.format(cce_probas(y_true=[[0, 0, 1]], y_pred=probas)),
         '(w logits): {:.4f}'.format(cce_logits(y_true=[[0, 0, 1]], y_pred=logits)))

# SparseCategoricalCrossentropy(多クラス分類)
sp_cce_probas = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
sp_cce_logits = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
tf.print('Sparse CCE (w Probas): {:.4f}'.format(sp_cce_probas(y_true=[2], y_pred=probas)),
         '(w logits): {:.4f}'.format(sp_cce_logits(y_true=[2], y_pred=logits)))
