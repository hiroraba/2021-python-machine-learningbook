""" 4.5.4 - 逐次特徴量選択アルゴリズム """

# 次元削減には「特徴量選択」と「特徴量抽出」がある
# 特徴量選択: 元の特徴量の一部を選択する
# 特徴量抽出: 特徴量の集合から情報を抽出する

# 逐次後退選択(SBS): 逐次特徴量選択アルゴリズム
# - 特徴量の次元を削減することで、分類器の性能の低下を最小限に抑えた上で計算効率を改善する
# - 特徴量を1つずつ削減しながら、一番性能の高い組み合わせを残していく

from numpy import testing
from scipy.sparse.construct import rand
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys, os
sys.path.append(os.pardir)
from common.load_wine import load_wine

class SBS():

    def __init__(self, estimator, k_features, scoring=accuracy_score, 
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimater = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        # 訓練データとテストデータに抽出
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # すべての特徴量の個数、列インデックス
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]

        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)

        self.scores_ = [score]

        # 特徴量が指定した個数になるまで繰り返す
        while dim > self.k_features:
            scores = []
            subsets = []

            # 特徴量の部分集合を表す列インデックスごとの処理を反復
            # combinations: データと、抽出する要素数を指定するとすべての組み合わせのタプルを返す
            for p in combinations(self.indices_, r=dim - 1):
                
                # 特徴量の部分集合を使ってスコアを算出
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)

                # 特徴量の部分集合を表すインデックスのリストを格納
                subsets.append(p)

            # 最良のスコアのインデックスを抽出
            best = np.argmax(scores)

            # 最良のスコアになる列インデックスを格納
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)

            # 特徴量を減らす
            dim -= 1
            # スコアを格納
            self.scores_.append(scores[best])

            
        # 最後に格納したスコア
        self.k_score_ = self.scores_[-1]
        return self
    
    
    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimater.fit(X_train[:, indices], y_train)
        y_pred = self.estimater.predict(X_test[:, indices])

        return self.scoring(y_test, y_pred)


from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

knn = KNeighborsClassifier(n_neighbors=5)

sbs=SBS(knn, k_features=1)

# データ読み込み
X_train_std, X_test_std, y_train, y_test = load_wine()

sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of Features')
plt.grid()
plt.tight_layout()
plt.show()

# 4次元が100%の正答率
k4 = list(sbs.subsets_[9])

knn.fit(X_train_std, y_train)
print('Training Accuracy:', knn.score(X_train_std, y_train))
print('Test Accuracy:', knn.score(X_test_std, y_test))

knn.fit(X_train_std[:, k4], y_train)
print('Training Accuracy:', knn.score(X_train_std[:, k4], y_train))
print('Test Accuracy:', knn.score(X_test_std[:, k4], y_test))
