""" 15.1.2 - 離散畳込みを実行する """

# 一次元の畳込み
import numpy as np

def conv1d(x, w, p=0, s=1):
    # wを反転する
    w_rot = np.array(w[::-1])
    x_padded = np.array(x)
    if p > 0:
        zero_pad = np.zeros(shape=p)
        x_padded = np.concatenate([zero_pad, x_padded, zero_pad])
    res = []
    for i in range(0, int((len(x_padded) - len(w_rot)) / s) + 1, s):
        res.append(np.sum(x_padded[i:i+w_rot.shape[0]] * w_rot))
    return np.array(res)

x = [1, 3, 2, 4, 5, 6, 1, 3]
w = [1, 0, 3, 1, 2]

print('Conv1d Implementation:', conv1d(x,w,p=2, s=1))
print('Numpy Results:', np.convolve(x, w, mode='same'))

# 2次元の畳込み
import scipy.signal

X = [[1, 3, 2, 4],
     [5, 6, 1, 3],
     [1, 2, 0, 2],
     [3, 4, 3, 2]]
W = [[1, 0, 3],
     [1, 2, 1],
     [0, 1, 1]]

print('Scipy Results:\n', scipy.signal.convolve2d(X,W,mode='same'))