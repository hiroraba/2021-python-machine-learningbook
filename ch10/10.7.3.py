""" 10.7.3 - ランダムフォレストを使って非線形関係に対処する """
import matplotlib.pyplot as plt

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    return None

# 決定木を使って、MEDV変数とLSTAT変数の非線形関係をモデル化してみる
## データの読み込み
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/hiroraba/python-machine-learning-book-3rd-edition/master/ch10/housing.data.txt', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

from sklearn.tree import DecisionTreeRegressor
X = df[['LSTAT']].values
y = df['MEDV'].values

# 決定木回帰モデルをインスタンス化
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)

# 一次元の配列に変換して、ソート後のインデックス配列を返す
sort_idx = X.flatten().argsort()

lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in 1000$ [MEDV]')
plt.show()

# ランダムフォレスト回帰
#  - 複数の決定木を組み合わせるため、決定木より汎化性能が高い
#  - データセットの外れ値に影響を受けず、チューニングも単純という利点がある
from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1].values # すべての特徴量を使う
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

from sklearn.ensemble import RandomForestRegressor
# ランダムフォレスト回帰のクラスをインスタンス化
# n_estimaters: 決定木の数
# criterion: 決定木の成長条件になる指標(mse=平均二乗誤差)
forest = RandomForestRegressor(n_estimators=1000,
                               criterion='mse',
                               random_state=1,
                               n_jobs=-1) 
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

# 平均二乗誤差と決定係数を出力
from sklearn.metrics import mean_squared_error, r2_score
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), 
                                       mean_squared_error(y_test, y_test_pred)))

print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), 
                                       r2_score(y_test, y_test_pred)))

# 予測値と残差をプロット
import matplotlib.pyplot as plt
plt.scatter(y_train_pred,
            y_train_pred - y_train,
            c='steelblue',
            edgecolor='white',
            marker='o',
            s=35,
            alpha=0.9,
            label='training data')
plt.scatter(y_test_pred,
            y_test_pred - y_test,
            c='limegreen',
            edgecolor='white',
            marker='s',
            s=35,
            alpha=0.9,
            label='Test data')
plt.xlabel('Predicted value')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
plt.xlim([-10, 50])
plt.tight_layout()
plt.show()