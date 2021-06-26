# 2次元のデータセットを準備する
from scipy.sparse.construct import random
from sklearn.datasets import make_blobs
import numpy as np

clusters = 7

X, y = make_blobs(n_samples=500, # データ点の数
                  n_features=2, # 特徴量
                  centers=clusters, # クラスタの数
                  cluster_std=40, # クラスタの標準偏差
                  shuffle=True,
                  random_state=0)

import matplotlib.pyplot as plt
# このデータをKmeansに適用させる
from sklearn.cluster import KMeans
def kmean(max_iter, init_param, n_clusters):
    km = KMeans(n_clusters=n_clusters, # クラスタの数
            init=init_param, # セントロイドの初期化方法
            n_init=10, # 異なるセントロイドの初期値を用いたアルゴリズムの試行回数
            max_iter=max_iter, # アルゴリズム内の最大試行回数
            tol=1e-04, # 収束と判定するための許容誤差
            random_state=0)
    y_km = km.fit_predict(X) # クラスタ中心の計算と各データ点のインデックス予測
    # クラスタリング結果を可視化
    clist = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i in range(n_clusters):
        plt.scatter(X[y_km==i,0],
                    X[y_km==i,1],
                    s=50,
                    c=clist[i],
                    edgecolor='black',
                    marker='s',
                    label='Cluster %d' % (i + 1))

    plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250,
            marker='*',
            c='red',
            edgecolor='black',
            label='Centroids')
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.tight_layout()
    plt.show()

kmean(max_iter=100, init_param='random', n_clusters=clusters)
kmean(max_iter=100, init_param='k-means++', n_clusters=clusters)