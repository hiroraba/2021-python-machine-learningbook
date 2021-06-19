""" 6.4 - グリッドサーチによる気合学習モデルのチューニング """
""" 6.4.1 - グリッドサーチを使ったハイパーパラメータのチューニング """

import numpy as np
import sys, os
sys.path.append(os.pardir)
from common.load_cancer import load_cancer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline, make_pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC

# データの読み込み
X_train, X_test, y_train, y_test = load_cancer()
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']},
              {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel':['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  refit=True,
                  n_jobs=1)

gs = gs.fit(X_train, y_train)
print('best score: %s, \nbest score params: %s' % (gs.best_score_, gs.best_params_))