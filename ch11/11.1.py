""" 11.1 k-means法を使った類似度によるオブジェクトのグループ化 """

# 2次元のデータセットを準備する
from scipy.sparse.construct import random
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=150, # データ点の数
                  n_features=2, # 特徴量
                  centers=3, # クラスタの数
                  cluster_std=0.5, # クラスタの標準偏差
                  shuffle=True,
                  random_state=0)

import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c='white', marker='o', edgecolor='black', s=50)
plt.grid()
plt.tight_layout()
plt.show()

# このデータをKmeansに適用させる
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, # クラスタの数
            init='random', # セントロイドの初期化方法
            n_init=10, # 異なるセントロイドの初期値を用いたアルゴリズムの試行回数
            max_iter=300, # アルゴリズム内の最大試行回数
            tol=1e-04, # 収束と判定するための許容誤差
            random_state=0)
y_km = km.fit_predict(X) # クラスタ中心の計算と各データ点のインデックス予測

# クラスタリング結果を可視化
plt.scatter(X[y_km==0,0],
            X[y_km==0,1],
            s=50,
            c='limegreen',
            edgecolor='black',
            marker='s',
            label='Cluster 1')

plt.scatter(X[y_km==1,0],
            X[y_km==1,1],
            s=50,
            c='orange',
            edgecolor='black',
            marker='o',
            label='Cluster 2')

plt.scatter(X[y_km==2,0],
            X[y_km==2,1],
            s=50,
            c='lightblue',
            edgecolor='black',
            marker='v',
            label='Cluster 2')

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