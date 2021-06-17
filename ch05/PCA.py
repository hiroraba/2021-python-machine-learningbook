from matplotlib.pylab import plt
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.pardir)
from common.load_wine import load_wine

# 主成分分析：高次元データにおいて分散が最大になる方向（ベクトル）を見つけ出し、
# 　　　　　　低い次元の新しい空間に射影（次元を削減する）
# 主成分分析のステップ
# 1 d次元のデータセットを標準化
# 2 標準化されたデータから共分散行列を作成
# 3 共分散行列を、固有ベクトルと固有値に変換
# 4 固有値を降順でソートして、対応する固有ベクトルを対応
# 5 大きい順からk個の固有ベクトルを選択する（kは削減後の次元数を表す）
# 6 手順5で得たk個の固有ベクトルから射影行列Wを作成する
# 7 Wを使ってd次元のデータセットXを変換し、新しいk次元の特徴量部分空間を取得する

## 1 d次元のデータセットを標準化
X_train_std, X_test_std, y_train, y_test = load_wine()

## 2 標準化されたデータから共分散行列を作成

### 共分散行列を作成
cov_mat = np.cov(X_train_std.T)
print('13次元の特徴量から得られる共分散行列', cov_mat.shape)

## 3 共分散行列を、固有ベクトルと固有値に変換
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('EigenValues\n %s' % eigen_vals)

## 4 固有値を降順でソートして、対応する固有ベクトルを対応
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

## 5 大きい順からk個の固有ベクトルを選択する（kは削減後の次元数を表す）
## 6 手順5で得たk個の固有ベクトルから射影行列Wを作成する
### 今回はk=2
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))

## 7 Wを使ってd次元のデータセットXを変換し、新しいk次元の特徴量部分空間を取得する
X_train_pca = X_train_std.dot(w)
print('特徴量部分空間のサイズ:', X_train_pca.shape)

# ラストにプロット
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0], X_train_pca[y_train==l, 1], c=c, marker=m, label=l)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
