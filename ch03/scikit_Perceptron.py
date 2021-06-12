#############################
# 1.データの読み込み
#############################

from sklearn import datasets
import numpy as np

#あやめデータを読み込み
iris = datasets.load_iris()

#3,4番目の特徴量
X = iris.data[:, [2, 3]].

#クラスラベル
Y = iris.target

print(X.shape)

#############################
# 2.データの分割
#############################

from sklearn.model_selection import train_test_split

# 訓練データとテストデータに分割する
# 全体の30%をテストデータにする
# random_stateに固定値を入れると再現可能な結果が得られる
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.3,random_state=1, stratify=Y)

# stratify=yにすると訓練サブセットとテストサブセットに含まれるクラスラベルの割合が入力データと同じになる
print('ラベル毎のデータ数(入力):', np.bincount(Y))
print('ラベル毎のデータ数(訓練データ):', np.bincount(Y_train))
print('ラベル毎のデータ数(訓練データ):', np.bincount(Y_test))

#############################
# 3.特徴量の標準化
#############################

# 特徴量のスケーリング
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# 訓練データの平均と標準偏差を計算
sc.fit(X_train)

# 平均と標準偏差から、データを標準化
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

print(X_train_std[0])

#############################
# 4.モデルの訓練と予測
#############################
from sklearn.linear_model import Perceptron

# 学習率0.1でパーセプトロンのインスタンスを作成
# random_stateを設定しているのはエポックごとに訓練データの並べ替えを再現できるように
ppn = Perceptron(eta0=0.1, random_state=1)

# 訓練データをモデルに適合
ppn.fit(X_train_std, Y_train)
# テストデータで予測を実施
y_pred = ppn.predict(X_test_std)
print('誤分類したデータの数: %d' % (Y_test != y_pred).sum())

#############################
# 5.正答率の算出
#############################
from sklearn.metrics import accuracy_score

print('正答率(accuracy_score): %.3f' % accuracy_score(Y_test, y_pred))
print('正答率(score): %.3f' % ppn.score(X_test_std, Y_test))