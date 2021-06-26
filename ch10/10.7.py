""" 10.7 - 多項式回帰: 線形回帰モデルから曲線を見出す """
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


X = np.array([258.0, 270.0, 294.0, 320.0, 342.0, 368.0, 396.0, 446.0, 480.0, 586.0])[:, np.newaxis]
y = np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 360.8, 368.0, 391.2, 390.8])

# 1 多項式の2次の項を追加する
## 線形回帰（最小二乗法）モデルのクラスをインスタンス化
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
pr = LinearRegression()

## 2次の多項式特徴量クラスをインスタンス化
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)

# 2 比較を簡単にするために単回帰モデルを学習させる
lr.fit(X, y)
# newaxisで列ベクトルにする
X_fit = np.arange(250, 600, 10)[:, np.newaxis]
y_lin_fit = lr.predict(X_fit)

# 3 多項式回帰のために変換された重回帰モデルを学習させる
pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

# 4 結果をプロットする
import matplotlib.pyplot as plt
plt.scatter(X, y, label='Training Points')
plt.plot(X_fit, y_lin_fit, label='Linear fit', linestyle='--')
plt.plot(X_fit, y_quad_fit, label='Quadratic fit')
plt.xlabel('Explantory variable')
plt.ylabel('Predicted or known target values')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# 平均二乗誤差と決定係数を求める
from sklearn.metrics import mean_squared_error, r2_score
y_lin_pred = lr.predict(X)
y_quad_pred = pr.predict(X_quad)
print('Training MSE linear: %.3f, quadratic: %.3f' % (mean_squared_error(y, y_lin_pred),
                                                      mean_squared_error(y, y_quad_pred)))
    
print('Training R^2 linear: %.3f, quadratic: %.3f' % (r2_score(y, y_lin_pred),
                                                      r2_score(y, y_quad_pred)))
# 平均二乗誤差は多項式のほうが低い。決定係数は2次のモデルにより適合することを示している