""" P76 3.5 - カーネルSVMを使った非線形問題の求解 """

#########################
## 1. 線形分離が不可能なデータの作成
#########################

import matplotlib.pyplot as plt
import numpy as np

# ランダムシードを指定
np.random.seed(1)

# 標準正規分布に従う乱数で200行2列の行列を生成
X_xor = np.random.randn(200, 2)

# 入力に対する排他的論理和を実行 y_xor = (200,)
# 排他的論理和=どっちかが1のときだけ出力が1
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)

# 真の場合は1,偽の場合は-1
y_xor = np.where(y_xor, 1, -1)

plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1], c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1], c='r', marker='s', label='-1')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.legend(loc='best')
plt.show()

#########################
## 2. RBFカーネルによる学習
#########################

import sys, os
from sklearn.svm import SVC
sys.path.append(os.pardir)
from common.plot_desicion_regions import plot_decision_regions

# gamma: 限界値パラメータ（大きくすると訓練データの影響が大きくなる）
# C:最適化パラメータ（小さいほど誤分類に対して寛大）
svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()