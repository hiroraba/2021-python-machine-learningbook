import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.plot_desicion_regions import plot_decision_regions
from common.load_iris import load_iris

class LogisticRegressionGD(object):

    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X ,y):
        rgen = np.random.RandomState(self.random_state)

        # 重みのサイズが入力に1足してるのは一番目の要素はバイアスに当たるため
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            self.cost_.append(cost)
 
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        # ロジスティックシグモイド関数を計算
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


X_train, X_test, Y_train, Y_test = load_iris()

# ロジスティック回帰がうまくいくのは2値分類タスクのみのためクラス0と1のみを考慮する
X_train_01_subset = X_train[(Y_train == 0) | (Y_train == 1)]
Y_train_01_subset = Y_train[(Y_train == 0) | (Y_train == 1)]

lrgd = LogisticRegressionGD()

# 訓練データをモデルに適合させる
lrgd.fit(X_train_01_subset, Y_train_01_subset)

plt = plot_decision_regions(X=X_train_01_subset,y=Y_train_01_subset,classifier=lrgd)
plt.show()