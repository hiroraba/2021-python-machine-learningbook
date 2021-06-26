""" 5.2 - 線形判別分析による教師ありデータ圧縮 """

# 線形判別分析（LDA）: 特徴量抽出手法の一つ
#  - PCA: データセットに対して分散が最も大きい直交成分軸を見つける
#  - LDA: クラスの分離を最適化する特徴量部分空間を見つける

# LDAの実行手順
# 1 d次元のデータセットを標準化する
# 2 クラスごとにd次元の平均ベクトルを用意する
# 3 平均ベクトルを使って、クラス間変動行列SBと、クラス内変動行列SWを生成する
# 4 行列SWの逆行列とSBのかけた行列の固有ベクトルと固有値を計算する
# 5 固有値を降順でソートすることで、対応する固有ベクトルをランクつけする
# 6 d×k次元の変換行列Wを生成するために、最も大きいk個の固有値に対するk個の固有ベクトルを選択する
#   (固有ベクトルからWを生成)。固有ベクトルはこの行列の列である。
# 7 変換行列Wを使ってデータ点を新しい特徴量部分空間に射影する

from matplotlib.pylab import plt
from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.pardir)
from common.load_wine import load_wine

## 1 d次元のデータセットを標準化
X_train_std, X_test_std, y_train, y_test = load_wine()

## 2 クラスごとの平均ベクトルを計算する
mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))

## 3 平均ベクトルを使って、クラス間変動行列SBと、クラス内変動行列SWを生成する
d = 13 #特徴量の個数
S_W = np.zeros((d, d)) # クラス内変動行列SW

for label, mv in zip(range(1,4), mean_vecs):
    # クラスラベルが一様に分布していない場合は、変動行列を正規化する必要があるが、それは共分散行列のこと
    class_scatter = np.cov(X_train_std[y_train==label].T) 
    S_W += class_scatter

S_B = np.zeros((d, d)) # クラス間変動行列SB
mean_overall = np.mean(X_train_std, axis=0) # 全体平均
print(mean_overall)
for i, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train==i+1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1) # 列ベクトルを生成
    mean_overall = mean_overall.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

## 4 行列SWの逆行列とSBのかけた行列の固有ベクトルと固有値を計算する
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

## 5 固有値を降順でソートすることで、対応する固有ベクトルをランクつけする
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# 6 d×k次元の変換行列Wを生成するために、最も大きいk個の固有値に対するk個の固有ベクトルを選択する
#   (固有ベクトルからWを生成)。固有ベクトルはこの行列の列である。
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))

# 7 変換行列Wを使ってデータ点を新しい特徴量部分空間に射影する
X_train_lda = X_train_std.dot(w)

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train==l, 0], X_train_lda[y_train==l, 1] * (-1), c=c, label=l, marker=m)

plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()