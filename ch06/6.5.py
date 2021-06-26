""" 6.5 - 様々な性能評価指標 """
""" 6.5.1 - 混同行列を解釈する """

import numpy as np
import sys, os
sys.path.append(os.pardir)

from common.load_cancer import load_cancer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC

# 混同行列: 学習アルゴリズムの性能を明らかにする行列
# - 真陽性(TP)、真隠性(TN)、偽陽性(FP)、偽陰性(FN)の4つの個数を表現する

# データの読み込み
X_train, X_test, y_train, y_test = load_cancer()

# 分類器の生成
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))

from sklearn.metrics import confusion_matrix
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)

# テストと予測のデータから混同行列を生成
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)
