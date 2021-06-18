""" 5.1.6 - 主成分分析 """
import sys, os
import numpy as np
sys.path.append(os.pardir)
from common.plot_desicion_regions import plot_decision_regions
from common.load_wine import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from matplotlib.pylab import plt

X_train_std, X_test_std, y_train, y_test = load_wine()

# 主成分数を指定して、PCAのインスタンスを生成
pca = PCA(n_components=2)

# ロジスティック回帰のインスタンス作成
lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')

# PCAによる次元削減
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.fit_transform(X_test_std)

lr.fit(X_train_pca, y_train)

plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()