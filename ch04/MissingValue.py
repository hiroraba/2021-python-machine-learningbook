""" 4.1 - 欠測データへの対処"""
import pandas as pd
from io import StringIO

#サンプルデータ作成
csv_data = '''A,B,C,D
              1.0,2.0,3.0,4.0
              5.0,6.0,,8.0
              10.0,11.0,12.0,'''

# サンプルデータ読み込み
df = pd.read_csv(StringIO(csv_data))

# 各特徴量の欠測値カウント
print(df.isnull().sum())

# 欠測値を含む行を削除
print(df.dropna())

# 欠測値を含む列を削除
print(df.dropna(axis=1))

# すべての列がNANである行だけを削除
print(df.dropna(how='all'))

# 非NaN値が4つ未満の行を削除
print(df.dropna(thresh=4))

########################
## 4.1.3 欠測値を補完する
########################

from sklearn.impute import SimpleImputer
import numpy as np

# 欠測値補完のインスタンス（平均値補完）
# strategy: 補完オプション(mean=平均値, median=中央値, most_frequent=最頻値)
imr = SimpleImputer(missing_values=np.nan, strategy='mean')

imr = imr.fit(df.values)
print(imr.transform(df.values))

