""" 10.3 - 最小二乗線形回帰モデルの実装 """
import matplotlib.pyplot as plt
import numpy as np

# 基本的な線形回帰モデル
class LinearRegressionGD():

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors) # 重みの更新
            self.w_[0] += self.eta * errors.sum() # バイアス更新
            cost = (errors**2).sum() / 2.0 # コスト関数（誤差平方和）
            self.cost_.append(cost)
        return self 
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return self.net_input(X)

# データの読み込み
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/hiroraba/python-machine-learning-book-3rd-edition/master/ch10/housing.data.txt', header=None, sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# 訓練データ(RM)を読み込む
X = df[['RM']].values

# 目的変数はMEDV
y = df['MEDV'].values
print(y.shape)
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD()
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

# 回帰直線と訓練データをプロットする
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    return None

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms[RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')
plt.show()

# 部屋数5つの住宅価格を予測する
num_rooms_std = sc_x.transform(np.array([[5.0]]))
price_std = lr.predict(num_rooms_std)
print('Price in $1000s: %.3f' % sc_y.inverse_transform(price_std))

""" 10.3.2 - scikit-learnを使って回帰モデルの係数を推定する """
from sklearn.linear_model import LinearRegression

slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)

lin_regplot(X, y, slr)
plt.xlabel('Average number of rooms[RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.show()