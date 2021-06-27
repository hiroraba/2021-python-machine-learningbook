""" 11.2 - クラスタを階層木として構成する """

# 完全連結法に基づく凝集型階層的クラスタリングの手順
# 1 すべてのデータ点の距離行列を計算する
# 2 各データ点を単一のクラスタとみなして表現する
# 3 最も類似度の低いメンバーの距離に基づき、2つの最も近いクラスタをマージする
# 4 距離行列を更新する
# 5 クラスタが一つだけに残った状態になるまで、手順3-4を繰り返す

import matplotlib.pyplot as plt
from numpy.core.defchararray import index
import pandas as pd
import numpy as np
np.random.seed(123)

# ランダムなデータ点を作成
## 特徴量
variables = ['X', 'Y', 'Z']
## クラスラベル
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5,3])*10 # 5行3列のサンプルデータを作成
df = pd.DataFrame(X, columns=variables, index=labels)
print(df)

# 距離行列を計算する
from scipy.spatial.distance import pdist, squareform
row_dist = pd.DataFrame(squareform(pdist(X, metric='euclidean')), columns=labels, index=labels)

# 完全連結法に基づく凝集型階層的クラスタリング
from scipy.cluster.hierarchy import linkage
row_clusters = linkage(df.values, method='complete', metric='euclidean')
print(pd.DataFrame(row_clusters,
                   columns=['row label1',
                            'row label2',
                            'distance',
                            'no. of items in cluster.'],
                   index=['cluster %d' % (i + 1) for i in range(row_clusters.shape[0])]))

# 1列目、2列目は各クラスタにおける一番類似度の低いメンバーのインデックスを表す
# - メンバーをまとめて作られたクラスタは新しいインデックス5が振られる（元のデータが0~4のため）
# - クラスタ番号が増えるにつれ、各メンバーがクラスタにまとめられていく
# - 最終的にクラスタ内のメンバー数＝元のメンバー数（この場合5）になったら学習終了。

# 樹形図として表す
from scipy.cluster.hierarchy import dendrogram
row_dendr = dendrogram(row_clusters, labels=labels)
plt.ylabel('Euclidean distance')
plt.tight_layout()
plt.show()

""" 11.2.3 - 樹形図をヒートマップと組み合わせる """
fig = plt.figure(figsize=(8, 8), facecolor='white')
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation='left')

# クラスタリングのラベルにアクセスするにはleavesを使う
df_rowclust = df.iloc[row_dendr['leaves'][::-1]]

axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')

plt.show()

""" 11.2.4 - scikit-learnを使って凝集型階層的クラスタリングを適用する """
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=3,
                             affinity='euclidean',
                             linkage='complete') # 連結法（ここでは完全連結法）

labels = ac.fit_predict(X)
print('Cluster labels: %s' % labels)
# ID_0, 4が1つ目のクラスタ(ラベル1)に割当られている
# ID_1, 2が2つめのクラスタ(ラベル0)に割り当てられている
# ID_3は独自のクラスタに割当られている