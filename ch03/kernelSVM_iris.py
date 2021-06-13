# RBFカーネルのirisデータの適用
from sklearn.svm import SVC
import sys, os
import numpy as np
sys.path.append(os.pardir)
from common.plot_desicion_regions import plot_decision_regions
from common.load_iris import load_iris
from sklearn.preprocessing import StandardScaler

# データ読み込み
X_train, X_test, y_train, y_test = load_iris()

# 訓練データの平均と標準偏差を計算
sc = StandardScaler()
sc.fit(X_train)

# 平均と標準偏差から、データを標準化
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 訓練データとテストデータを行方向に結合
X_combined_std = np.vstack((X_train_std, X_test_std))

# 訓練データとテストデータのクラスラベルを結合
y_combined = np.hstack((y_train, y_test))

# RBFカーネルを使ったSVM
# gamma:限界値パラメータ (大きいほど訓練データの影響が大きくなる)
gamma = 0.2
svm = SVC(kernel='rbf', C=10, random_state=1, gamma=gamma)
svm.fit(X_train_std, y_train)

plt = plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.title('svm with RBF - gamma:%.3f' % gamma)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()