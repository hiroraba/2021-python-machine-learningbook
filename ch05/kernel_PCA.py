""" 5.3 - カーネル主成分分析を使った非線形写像 """

# カーネル主成分分析
#  - 非線形問題に対応した主成分分析
#  - データセットを高次元の空間に写像して、その空間に対してPCAを適用する
#  - ※高次元空間への写像すると、非線形問題を線形分離可能にする
#  - 計算コストが非常に高いため、カーネルトリックを利用する

# RBFカーネル主成分分析の手順
# 1 カーネル（類似度）行列Kを計算する
# 2 カーネル行列を中心化する
#   - 明示的に特徴空間を計算しないので、新しい特徴空間でも中心が0であることが保証できないため
# 3 対応する固有値に基づき、中心化されたカーネル行列のk個の固有ベクトルを収集する。
#   - 固有ベクトルは主成分軸ではなく、すでに射影されているデータ

from os import fstatvfs
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
from matplotlib.pylab import plt

def rbf_kernal_pca(X, gamma, n_components):
    """
    パラメータ
    ----------
    X: {Numpy ndarray}, shape = [n_examples, n_features]
    gamma: float RBFカーネルのチューニングパラメータ
    n_components: int 返される主成分の個数

    戻り値
    ----------
    X_pc: {Numpy ndarray}, shape = [n_examples, k_features] 射影されたデータセット
    """

    # M×N次元のデータセットでペアごとにユークリッド距離の二乗を計算
    sq_dists = pdist(X, 'sqeuclidean')

    # ペア毎の距離を正方行列に変換
    mat_sq_dists = squareform(sq_dists)

    # 対称カーネル行列を計算
    K = exp(-gamma*mat_sq_dists)

    # カーネル行列を中心化
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # 中心化されたカーネル行列から固有対を取得: scipy.linalg.eightは昇順で返す
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    # 上位K個の固有ベクトルを収集
    X_pc = np.column_stack((eigvecs[:, i] for i in range(n_components)))

    # 対応する固有値を収集
    lambdas = [eigvals[i] for i in range(n_components)]
    return X_pc, lambdas

# 例1 半月形の分離
from sklearn.datasets import make_moons

## データセットの可視化: 線形分離できない
X, y = make_moons(n_samples=100, random_state=123)
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^')
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o')
plt.tight_layout()
plt.show()

## 標準のPCA: 線形分離できない
from sklearn.decomposition import PCA

scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
plt.scatter(X_spca[y==0, 0], X_spca[y==0, 1], color='red', marker='^')
plt.scatter(X_spca[y==1, 0], X_spca[y==1, 1], color='blue', marker='o')
plt.tight_layout()
plt.show()

## RBFカーネルPCA
X_kpca, _ = rbf_kernal_pca(X, gamma=15, n_components=2)
plt.scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='red', marker='^')
plt.scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='blue', marker='o')
plt.tight_layout()
plt.show()

# 例2:同心円の分離
from sklearn.datasets import make_circles

## データセットの可視化: 線形分離できない
X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^')
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o')
plt.tight_layout()
plt.show()

## 標準のPCA: 線形分離できない
X_spca = scikit_pca.fit_transform(X)
plt.scatter(X_spca[y==0, 0], X_spca[y==0, 1], color='red', marker='^')
plt.scatter(X_spca[y==1, 0], X_spca[y==1, 1], color='blue', marker='o')
plt.tight_layout()
plt.show()

## RBFカーネルPCA
X_kpca, _ = rbf_kernal_pca(X, gamma=15, n_components=2)
plt.scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], color='red', marker='^')
plt.scatter(X_kpca[y==1, 0], X_kpca[y==1, 1], color='blue', marker='o')
plt.tight_layout()
plt.show()