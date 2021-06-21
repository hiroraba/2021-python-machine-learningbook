""" 6.5.2 - 分類モデルの適合率と再現率を最適化する """

import numpy as np
import sys, os
sys.path.append(os.pardir)

from common.load_cancer import load_cancer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC

# 適合率(PRE): TP/(TP+FP)
#  - 陽性クラスと判断したものの中から、真陽性だったものの割合
#  - 適合率に最適化すると、FNの値が大きくなる(健康患者を悪性判断しやすくなる)
# 再現率(REC): TP/(FN+TP)
#  - 実際に陽性だったものの中から、真陽性だったものの割合
#  - 再現率に最適化すると、FPの値が大きくなる(悪性腫瘍の不検出が起こりやすくなる)
#
# PREとRECの長所と短所のバランスを取るためにF1スコアがよく使われる
# F1 = 2 * ((PRE * REC)/ (PRE + REC))


# データの読み込み
X_train, X_test, y_train, y_test = load_cancer()

# 分類器の生成と、モデルの適合、予測を実施
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)

# 適合率、再現率、F1スコアを出力
from sklearn.metrics import precision_score, recall_score, f1_score

print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))