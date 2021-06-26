""" 10.7.2 - Housingデータセットで非線形関係をモデル化する """
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd

# データの読み込み

df = pd.read_csv('https://raw.githubusercontent.com/hiroraba/python-machine-learning-book-3rd-edition/master/ch10/housing.data.txt', header=None, sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

X = df[['LSTAT']].values
y = df['MEDV'].values

from sklearn.linear_model import LinearRegression
regr = LinearRegression()

# 2次と3次の特徴量を作成
from sklearn.preprocessing import PolynomialFeatures

quad = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quad.fit_transform(X)
X_cubic = cubic.fit_transform(X)

X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

# 特徴量の学習、予測、決定係数の計算
regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

# 2次の特徴量の学習、予測、決定係数の計算
regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quad.fit_transform(X_fit))
quad_r2 = r2_score(y, regr.predict(X_quad))

# 2次の特徴量の学習、予測、決定係数の計算
regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

# 各モデルの結果をプロット
import matplotlib.pyplot as plt
plt.scatter(X, y, label='Training Points', color='lightgray')
plt.plot(X_fit, y_lin_fit, label='Linear (d=1), $R^2=%.2f$' % linear_r2, color='blue', lw=2, linestyle=':')
plt.plot(X_fit, y_quad_fit, label='Quadratic (d=2), $R^2=%.2f$' % quad_r2, color='red', lw=2, linestyle='-')
plt.plot(X_fit, y_cubic_fit, label='Cubic (d=3), $R^2=%.2f$' % cubic_r2, color='green', lw=2, linestyle='--')
plt.xlabel('% lower status of population [LSTAT]')
plt.ylabel('Price in 1000$ [MDEV]')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()