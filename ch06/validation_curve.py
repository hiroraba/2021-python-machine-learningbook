""" 6.3.2 - 検証曲線を使って過学習と学習不足を明らかにする """

import numpy as np
import sys, os
sys.path.append(os.pardir)
from common.load_cancer import load_cancer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve

# データの読み込み
X_train, X_test, y_train, y_test = load_cancer()

# パイプライン
# ロジスティック回帰のイテレーションが多いのは収束問題を回避するため
pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty='l2', 
                                           random_state=1, 
                                           solver='lbfgs', 
                                           max_iter=10000))

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
# 逆正則化パラメータを調整する
# 大きいと正則化の強さを下げる
train_scores, test_scores = validation_curve(estimator=pipe_lr,
                                             X=X_train, 
                                             y=y_train,
                                             param_name='logisticregression__C',
                                             param_range=param_range,
                                             cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation Accuracy')
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.xlabel('Parameter')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.ylim([0.8, 1.03])
plt.tight_layout()
plt.show()