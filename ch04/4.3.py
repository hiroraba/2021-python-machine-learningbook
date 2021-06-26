""" 4.3 - データセットを訓練データセットとテストデータセットに分割する """
from numpy.core.fromnumeric import std
import pandas as pd
import numpy as np

# wineデータセットを読み込む
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

# 列名の設定
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

print(df_wine.head())

from sklearn.model_selection import train_test_split

# Xに特徴量、yにクラスラベル
# iloc:行番号または列番号を指定してデータを抽出
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:,0].values

# 訓練データとテストデータに分割(30%をテストデータにする)
# stratify=y 出力値（クラスラベル）が均等になるように
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

####################
## 4.4 特徴量の尺度を揃える
####################

# 特徴量の尺度を揃えないと、より大きな誤差がでる特徴量に従って最適化することになる
from sklearn.preprocessing import MinMaxScaler

# 正規化: 特徴量を0から1の間にスケーリングする
mms = MinMaxScaler()

X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.fit_transform(X_test)

# 標準化: 平均0, 標準偏差1になるようにスケーリングする
# 多くのアルゴリズムの場合は標準化のほうが効果的
#   - 多くの線形モデルが0,または0に近い乱数で初期化するので重みが学習しやすい
#   - 外れ値(例外的な値を持つモデル)に関する有益な情報が保持されるため、外れ値から受ける影響がすくなくなる
from sklearn.preprocessing import StandardScaler

# 標準化のインスタンス
stdsc = StandardScaler()

X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)

####################
## 4.5 有益な特徴量の選択
####################

# 訓練データセットが少ない場合は過学習を引き起こす可能性が高い
# 過学習: 訓練されたデータセットに対してモデルが複雑過ぎる
# 正則化を通じて、複雑なモデルに対してペナルティを課す

from sklearn.linear_model import LogisticRegression

# ロジスティック回帰のインスタンスを初期化
lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')
lr.fit(X_train_std, y_train)
print('訓練データに対する正答率:', lr.score(X_train_std, y_train))
print('テストデータに対する正答率:', lr.score(X_test_std, y_test))

# 正則化パラメータの効果を可視化
#  - 少ないほど、大きな重みにペナルティを与える

import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.subplot(111)

# 各係数の色をリスト化
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c, solver='liblinear', multi_class='ovr', random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[0])
    params.append(10**c)

weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    # カラム(列)毎の値をプロット
    plt.plot(params, weights[:, column], label=df_wine.columns[column+1], color=color)

plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.xscale('log')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.show()