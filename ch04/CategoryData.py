""" 4.2 - カテゴリデータの処理 """
import pandas as pd

##################
## 4.2.2 pandasを使ったカテゴリデータのエンコーディング
##################

# サンプルデータの作成
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2'],
])

# 列名を設定
df.columns = ['color', 'size', 'price', 'classlabel']
print(df)

##################
## 4.2.3 順序特徴量のマッピング
##################

# Tシャツのサイズと整数を対応させる
# 大小関係を表現できるように数値を設定する
size_mapping = {'XL':3, 'L':2, 'M':1}
df['size'] = df['size'].map(size_mapping)
print(df)

##################
## 4.2.3 クラスラベルのエンコーディング
##################
import numpy as np

# クラスラベルと整数を対応させるディクショナリを生成
# 順序特徴量ではないので数値は何でもよい
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)

# 同じことをLineEncoderを使っても可能
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)

##################
## 4.2.4 名義特徴量でのone-hotエンコーディング
##################

# Tシャツの色、サイズ、価格を抽出
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()

# colorをエンコード
X[:, 0] = color_le.fit_transform(X[:,0])
print(X) # -> greenがblueより大きいと想定としてしまうので駄目

from sklearn.preprocessing import OneHotEncoder
X = df[['color', 'size', 'price']].values
# one-hotエンコーダ
color_ohe = OneHotEncoder()
print(color_ohe.fit_transform(X[:, 0].reshape(-1,1)).toarray())

# 列を選択的に変換したい場合
from sklearn.compose import ColumnTransformer

X = df[['color', 'size', 'price']].values
# (name, transformer, columns(適用したい列)) でタプルのリストを渡す
# 何も変換しない場合はtransformerにpassthrough
c_transf = ColumnTransformer([('onehot', OneHotEncoder(), [0]),('nothing','passthrough',[1,2])])
print(c_transf.fit_transform(X))

# pandasのget_dummiesでもone-hotエンコーディングができる
print(pd.get_dummies(df[['price','color','size']]))

# one-hotエンコーディングは計算量が多くなることがあるのでdrop_firstで最初の列(red)を削除する
# greenとblueが0であればredであることはわかるので、特徴量は失われていない
print(pd.get_dummies(df[['price','color','size']], drop_first=True))

# OneHotEncoderを使う場合はcategories=auto, drop=firstを指定
X = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer([('onehot', OneHotEncoder(categories='auto', drop='first'), [0]),('nothing','passthrough',[1,2])])
print(c_transf.fit_transform(X))
