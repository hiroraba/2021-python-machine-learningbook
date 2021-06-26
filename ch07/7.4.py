""" 7.4 - バギング: ブートストラップ標本を使った分類器アンサンブルの構築 """

import pandas as pd

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

# BaggingClassifierをscikit-learnからインポートする
# ベース分類器には決定木を使用
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=1)

## BaggingClassifier:
## - n_estimaters: 分類器の数
## - max_samples: 各分類器に使う最大のインスタンス数(default:1.0)
## - max_features: 各分類器に使う最大の特徴量(default:1.0)
## - bootstrap: インスタンスの重複を許す
## - bootstrap_features: 特徴量の重複を許す
bag = BaggingClassifier(base_estimator=tree, 
                        n_estimators=500,
                        max_samples=1.0,
                        max_features=1.0,
                        bootstrap=True,
                        bootstrap_features=False,
                        n_jobs=1,
                        random_state=1)

from sklearn.metrics import accuracy_score
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))

bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)

bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print('Bagging train/test accuracies %.3f/%.3f' % (bag_train, bag_test))

# バギングアルゴリズムは、
# バリアンス（モデルの複雑さ）の抑制には効くが、バイアス（モデルと学習データのズレ）の抑制には効果がない。
# - 過学習の抑制を抑えることはできる