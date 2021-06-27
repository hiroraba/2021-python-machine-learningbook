""" 11.3 - DBSCANを使って高密度の領域を特定する """

# 半月状のデータを作成
from numpy import random
from scipy.cluster.hierarchy import linkage
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200,
                  noise=0.05,
                  random_state=0)

import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1])
plt.tight_layout()
plt.show()

# K-meansと完全連結法でクラスタリングする
from sklearn.cluster import KMeans, AgglomerativeClustering
f, (ax1, ax2) = plt.subplots(1,2, figsize=(8, 3))
km = KMeans(n_clusters=2, random_state=0)
y_km = km.fit_predict(X)

ax1.scatter(X[y_km==0, 0],
            X[y_km==0, 1],
            c='lightblue',
            edgecolor='black',
            marker='o',
            s=40,
            label='cluster 1')
ax1.scatter(X[y_km==1, 0],
            X[y_km==1, 1],
            c='red',
            edgecolor='black',
            marker='s',
            s=40,
            label='cluster 2')
ax1.set_title('K-means Clustering')

ac = AgglomerativeClustering(n_clusters=2,
                             affinity='euclidean',
                             linkage='complete')

y_ac = ac.fit_predict(X)
ax2.scatter(X[y_ac==0, 0],
            X[y_ac==0, 1],
            c='lightblue',
            edgecolor='black',
            marker='o',
            s=40,
            label='cluster 1')
ax2.scatter(X[y_ac==1, 0],
            X[y_ac==1, 1],
            c='red',
            edgecolor='black',
            marker='s',
            s=40,
            label='cluster 2')
ax2.set_title('Agglomerative Clustering')
plt.legend()
plt.tight_layout()
plt.show()

# DBSCANアルゴリズムを適用
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.2, # 隣接点とみなす2点間の最大距離
            min_samples=5, # ボーダー点の最小個数
            metric='euclidean')

y_db = db.fit_predict(X)

plt.scatter(X[y_db==0, 0],
            X[y_db==0, 1],
            c='lightblue',
            edgecolor='black',
            marker='o',
            s=40,
            label='cluster 1')
plt.scatter(X[y_db==1, 0],
            X[y_db==1, 1],
            c='red',
            edgecolor='black',
            marker='s',
            s=40,
            label='cluster 2')
plt.legend()
plt.tight_layout()
plt.show()