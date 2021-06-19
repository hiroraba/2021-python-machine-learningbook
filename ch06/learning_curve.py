""" 6.3 - 学習曲線と検証曲線によるアルゴリズムの診断 """
""" 6.3.1 - 学習曲線を使ってバイアスとバリアンスの問題を診断する """

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

# バイアス: モデルと学習データのズレ
#  - 高いと学習不足に陥る
#  - モデルのパラメータを増やす、正則化の強さを下げるなどが対処法
# バリアンス: モデルの複雑度
#  - 高いと過学習に陥る
#  - 訓練データを増やす、特徴量選択や、特徴量抽出で次元削減するなどが対処法

# 学習曲線を使ってモデルを評価する
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# データの読み込み
X_train, X_test, y_train, y_test = load_cancer()

# パイプライン
# ロジスティック回帰のイテレーションが多いのは収束問題を回避するため
pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty='l2', 
                                           random_state=1, 
                                           solver='lbfgs', 
                                           max_iter=10000))

train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr,
                                                        X=X_train,
                                                        y=y_train,
                                                        train_sizes=np.linspace(0.1, 1.0, 10),
                                                        cv=10,
                                                        n_jobs=2)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training example')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.ylim([0.8, 1.03])
plt.tight_layout()
plt.show()