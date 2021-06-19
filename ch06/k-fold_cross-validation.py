""" 6.2 - k分割交差検証を使ったモデルの性能の評価 """
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

# パイプライン
pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(random_state=1, solver='lbfgs'))

# データの読み込み
X_train, X_test, y_train, y_test = load_cancer()

# 層化k分割交差検証イテレータを表すStratifiedKFoldクラスのインスタンス化
kFold = StratifiedKFold(n_splits=10).split(X_train, y_train)
scores = []

# イテレータのインデックスと要素をループ処理: (上から順に)
#  データをモデルに適合
#  テストデータの正解率を算出
#  リストに正解率を追加
#  分割の番号、0以上の要素数、正解率を出力
for k, (train, test) in enumerate(kFold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, class dist: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))

# 正解率の平均と標準偏差を出力
print('\nCV Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# 交差検証のモデルの正解率を算出
from sklearn.model_selection import cross_val_score

# n_jobs: 並列ジョブ数
# cv: データセットの分割数
#     -1 で利用可能なすべてのCPUで交差検証を実行する
scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=2)

print('\nCV Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))