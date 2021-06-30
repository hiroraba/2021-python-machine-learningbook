""" 13.5 - 多層ニューラルネットワークでの活性化関数の選択 """
import numpy as np

X = np.array([1, 1.4, 2.5])
w = np.array([0.4, 0.3, 0.5])

def net_input(X, w):
    return np.dot(X, w)

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_activation(X, w):
    z = net_input(X, w)
    return logistic(z)

print('P(y=1|x) = %.3f' % logistic_activation(X, w)) # 0.88 = 88%でデータ点xが陽性クラスに所属する確率

# 多クラス分類
W = np.array([[1.1, 1.2, 0.8, 0.4],
              [0.2, 0.4, 1.0, 0.2],
              [0.6, 1.5, 1.2, 0.7]])
print(W[:, 0]) # バイアスユニット

# 最初の列は1でなければならない
A = np.array([[1, 0.1, 0.4, 0.6]])

Z = np.dot(W, A[0])
y_probas = logistic(Z)

print('Net Input: ', Z)
print('Output Units:', y_probas) # 足して1にならないので、所属確率として扱えない

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

y_probas = softmax(Z)
print('Probabilities: ', y_probas)

import tensorflow as tf
Z_tensor = tf.expand_dims(Z, axis=0)
print(tf.keras.activations.softmax(Z_tensor))


""" 13.5.3 - 双曲線正接関数を使って出力範囲を拡大する """

# ロジスティック関数が開区間(0, 1)の出力を返すのに対し、
# 開区間(-1, 1)の出力を返す、出力範囲が広いため、
# ロジスティック関数が0に近くなった場合に時間がかかるような現象を防ぐことができる

import matplotlib.pyplot as plt

def tanh(z):
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)


z = np.arange(-5, 5 , 0.005)
log_act = logistic(z)
tanh_act = tanh(z)

plt.ylim([-1.5, 1.5])
plt.xlabel('net input')
plt.ylabel('actiovation')
plt.axhline(1, color='black', linestyle=':')
plt.axhline(0.5, color='black', linestyle=':')
plt.axhline(0, color='black', linestyle=':')
plt.axhline(-0.5, color='black', linestyle=':')
plt.axhline(-1, color='black', linestyle=':')

plt.plot(z, tanh_act, linestyle='--', label='tanh')
plt.plot(z, log_act, linestyle='-.', label='logistic')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()