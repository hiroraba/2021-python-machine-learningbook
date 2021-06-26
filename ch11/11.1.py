""" 11.1 k-means法を使った類似度によるオブジェクトのグループ化 """

# 2次元のデータセットを準備する
from scipy.sparse.construct import random
from sklearn import metrics
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

# クラスタ内誤差平方和（歪み）
print('Distortion: %.2f' % km.inertia_)

distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
    
plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()


""" 11.1.5 - シルエット図を使ってクラスタリングの性能を数値化する """

# シルエット係数の計算手順
# 1 クラスタの凝集度を計算する。凝集度は、クラスタのデータ点xと他の全データ点との平均距離として計算する
# 2 最も近いクラスタからの乖離度bを計算する。乖離度は、データ点xと最も近いクラスタ内の全データ点との平均距離として計算する
# 3 クラスタの凝集度と乖離度の差を大きい方の数値で割る
#  - 乖離度 >>> 凝集度の場合に1に近づく
#  - 乖離度により、他のクラスタとの非類似度が数値化され、凝集度によって同じクラスタ内の他のデータ点ての類似度が数値化されるため

# シルエット係数の可視化
km = KMeans(n_clusters=3,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]

# シルエット係数計算
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0,0
yticks = []

for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i)/n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper),
             c_silhouette_vals,
             height=1.0,
             edgecolor='none',
             color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.) # クラスタラベルの表示位置
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color='red', linestyle='--')
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.tight_layout()
plt.show()