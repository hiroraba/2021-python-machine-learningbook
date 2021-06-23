""" 7.5 - アダブーストによる弱学習器の活用 """

# アダブースト: 誤答を訂正していく弱学習器に基づくアルゴリズム
#  - 誤分類されたデータを重みを増やし、再学習させることで性能を向上させる

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

# 列名の設定
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

# クラス1の削除
df_wine = df_wine[df_wine['Class label'] != 1]
y = df_wine['Class label'].values

# 特徴量を2つに絞って取得する
X = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values

# クラスラベルを二値でエンコードして訓練データとテストデータに分割する
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# AdaboostClassifierをscikit-learnからインポートする
# ベース分類器には決定木を使用
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# max_depth=1にして決定株にする
tree = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=1)

## AdaBoostClassifier:
ada = AdaBoostClassifier(base_estimator=tree, 
                        n_estimators=500,
                        learning_rate=0.1,
                        random_state=1)

from sklearn.metrics import accuracy_score
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))

ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)

ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print('Adaboost train/test accuracies %.3f/%.3f' % (ada_train, ada_test))

# 決定領域を確認する
from itertools import product

# 決定領域を描画する最小値、最大値を生成
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1

# グリッドポイントを生成
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# 描画領域を2行2列に分割
f, axarr = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(8, 3))

# 決定領域のプロット、赤や青の散布図の作成などを実行
# 変数idxは各分類器を描画する行と列の位置を表すタプル

for idx, clf, tt in zip([0, 1], [tree, ada], ['Decision Tree', 'AdaBoost']):
    clf.fit(X_train, y_train)
    
    # 描画領域のすべてのデータ点についてクラスラベルを予測する
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    # クラスラベル間の決定領域を描画するために等高線を引く
    axarr[idx].contourf(xx, yy, z, alpha=0.3)

    # 訓練データをプロットする(クラス1)
    axarr[idx].scatter(X_train[y_train==0, 0],
                       X_train[y_train==0, 1],
                       c='blue',
                       marker='^',
                       s=50)

    # 訓練データをプロットする(クラス2)
    axarr[idx].scatter(X_train[y_train==1, 0],
                       X_train[y_train==1, 1],
                       c='green',
                       marker='o',
                       s=50)
    axarr[idx].set_title(tt)

axarr[0].set_ylabel('Alcohol', fontsize=12)
plt.tight_layout()
plt.text(0, -0.1, s='OD280/OD315 of diluted wines', ha='center', va='center', fontsize=12, transform=axarr[1].transAxes)
plt.show()