""" 5.3.4 - scikit-learnのカーネル主成分分析 """
from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA
from matplotlib.pylab import plt
X, y = make_moons(n_samples=100, random_state=123)

scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)

plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], color='red', marker='^')
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1], color='blue', marker='o')
plt.tight_layout()
plt.show()