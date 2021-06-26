""" 10.1 - 線形回帰"""
 # 線形回帰の目標: 1つ以上の特徴量と連続地の目的変数の関係をモデルとして表現する
 #  - 同じ教師あり学習である分類との違いは、連続する尺度に基づいて出力を予測する

""" 10.2.1 Housingデータセットをデータフレームに読み込む """
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/hiroraba/python-machine-learning-book-3rd-edition/master/ch10/housing.data.txt', header=None, sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# 目的変数（予測したい値）をMEDV（住宅価格の中央値）とする

print(df.head())

""" 10.2.2 - データセットの重要な特性を可視化する """
# 探索的データ解析(EDA)
#  - 機械学習モデルを訓練する前の最初の重要なステップ

# 散布図行列
#  - データセットの特徴量のペアの相関関係を可視化できる
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix

cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

scatterplotmatrix(df[cols].values, figsize=(10, 8), names=cols, alpha=0.5)
plt.tight_layout()
plt.show()

""" 10.2.3 - 相関行列を使って関係を調べる """
from mlxtend.plotting import heatmap
import numpy as np

cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)
plt.show()

# 相関行列、散布図を見るとMEDVと一番相関が高いのはRM（一戸あたりの平均部屋数）に思える