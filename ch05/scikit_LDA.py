""" 5.2.6 - scikit-learnによる線形判別分析 """
import sys, os
import numpy as np
sys.path.append(os.pardir)
from common.plot_desicion_regions import plot_decision_regions
from common.load_wine import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from matplotlib.pylab import plt

X_train_std, X_test_std, y_train, y_test = load_wine()

# 主成分数を指定して、PCAのインスタンスを生成
lda = LDA(n_components=2)

# ロジスティック回帰のインスタンス作成
lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')

# LDAによる次元削減
X_train_lda = lda.fit_transform(X_train_std, y_train)

lr = lr.fit(X_train_lda, y_train)

plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.show()