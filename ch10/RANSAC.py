""" 10.4 - RANSACを使ったロバスト回帰モデルの学習 """

# 線形回帰モデルは外れ値の存在に大きく左右されることがある
#  - ここでは外れ値の除去については取り上げない代わりにRANSACを使ったロバスト回帰について調べる
# RANSAC
#  - 回帰モデルにデータのサブセット（正常値）を学習させる

# RANSACの手順
# 1 正常値としてランダムな数のデータ点を選択して、モデルを学習させる
# 2 学習済のモデルに対して、その他すべてのデータ点を評価し、ユーザ指定の許容範囲となる
#   データ点を正常値に追加する
# 3 すべての正常を使ってモデルを再学習する
# 4 正常値に対する学習済のモデルの誤差を推定する
# 5 モデルの性能がユーザ指定のしきい値の条件を満たしている場合、またはイテレーションが
#   既定の回数に達した場合は終了、そうでなければ1に戻る

# データの読み込み
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/hiroraba/python-machine-learning-book-3rd-edition/master/ch10/housing.data.txt', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

X = df[['RM']].values
y = df['MEDV'].values

from sklearn.linear_model import RANSACRegressor, LinearRegression
import numpy as np
import matplotlib.pyplot as plt
# RANSACRegressor
# max_trials: 最大イテレーション
# min_samples: ランダムに選択するデータ点の数
# loss: 誤差の計算方法(absolute_lossの場合は絶対値)
# residual_threshold: 正常値に含めるしきい値（5の場合は学習直線から5以内を正常値とする）
ransac = RANSACRegressor(LinearRegression(),
                         max_trials=100,
                         min_samples=50,
                         loss='absolute_loss',
                         residual_threshold=5.0,
                         random_state=0)

ransac.fit(X, y)

inliner_mask = ransac.inlier_mask_ # 正常値を表す真偽値
outliner_mask = np.logical_not(inliner_mask) # 外れ値を表す真偽値
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis]) # 予測値計算

# 正常値をプロット
plt.scatter(X[inliner_mask], y[inliner_mask], c='steelblue', edgecolor='white', marker='o', label='Inliners')
# 外れ値をプロット
plt.scatter(X[outliner_mask], y[outliner_mask], c='limegreen', edgecolor='white', marker='s', label='Outliners')
# 予測値をプロット
plt.plot(line_X, line_y_ransac, color='black', lw=2)
plt.legend(loc='upper left')
plt.show()