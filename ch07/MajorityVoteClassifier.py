""" 7.2 - 多数決による分類器の結合 """
""" 7.2.1 - 単純な多数決分類器を実装する """

from scipy.sparse.construct import random
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """ 多数決アンサンブル分類器 
    
    パラメータ
    ------------
    classifiers : array-like, shape = [n_classifiers]
       アンサンブルの様々な分類器
    
    vote : str, {'classlabel', 'probability'} ('default': 'classlabel')
    　　'classlabel'の場合、クラスラベルの予測はクラスラベルのargmaxに基づく
    　　'probability'の場合、クラスラベルの予測はクラスの所属確率のargmaxに基づく

    weights : array-like, shape = [n_classifiers] (optional, default=None)
        分類器は重要度で重み付けされる
    """

    def __init__(self, classifiers, vote='classlabel', weights=None):
        
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key,
                                  value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    
    def fit(self, X, y):
        """ 分類器を学習させる

        パラメータ
        ----------
        X: {array-like, sparse matrix}, shape : [n_exemples, n_features]
        　訓練データからなる行列
        y: {array-like}, shape : [n_examples]
        　クラスラベルからなる行列
        
        戻り値
        -----------
        self : object
        
        """

        # パラメータチェック
        if self.vote not in {'probability', 'classlabel'}:
            raise ValueError("vote must be probability or classlabel")
        
        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError('number of weights and classifier must be equal')

        # LabelEncoderを使ってクラスラベルが0から始まるようにエンコード
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []

        # 訓練データを分類器に一つずつ適合させる
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        """ クラスラベルを予測する

        パラメータ
        ----------
        X : {array-like, sparse matrix}, shape= [n_examples, n_features]
　　　　　　　訓練データからなる行列
        戻り値
        ----------
        maj_vote : array-like, shape: [n_examples]
        　　各分類期から多数決で予測されたクラスラベル

        """

        if self.vote == 'probability':
            # axis=1:各データ点で一番平均値が大きい方の大きいクラスラベルをとる
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            
            # 各データ点のクラス確率に重みをかけて足し合わせた値が値が最大となる
            # 列番号を返す 
            maj_vote = np.apply_along_axis(
                lambda x:
                np.argmax(np.bincount(x, weights=self.weights)),
                axis=1,
                arr=predictions
            )
        # 各データ点に最大値を与えるクラスラベルを抽出
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote


    def predict_proba(self, X):
        """ Xのクラス確率を予測する
        
        パラメータ
        ----------
        X : {array-like, sparse matrix} shape = [n_examples, n_features]
        　訓練ベクトル: n_examplesはデータ点の個数、n_featuresは特徴量の個数

        戻り値:
        ----------
        avg_proba: array-like shape = [n_examples, n_classes]
        　各データに対する各クラスで重みを付けた平均確率

        """

        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        # axis=0にすることにより、各分類器が出した平均を取る
        avg_probas = np.average(probas, axis=0, weights=self.weights)
        return avg_probas

    def get_params(self, deep=True):
        """ GridSearchの実行時に分類器のパラメータ名を取得 """
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items:
                for key, value in step.get_params(deep=True).items():
                    out['%s__%s' % (name, key)] = value
            return out

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# VersicolorクラスとVersinicaクラスのデータ点のみを分類する
iris = datasets.load_iris()

## 50個目から150個目までがVersicolorクラスとVersinicaクラス
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)

# ３つの分類器を訓練する
#  - ロジスティック回帰分類器
#  - 決定木分類器
#  - k最近傍法分類器

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

clf1 = LogisticRegression(penalty='l2', C=0.001, solver='lbfgs', random_state=1)
clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
pipe1 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf1]])

pipe3 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf3]])

mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])

clf_labels = ['Logstic Regression', 'Desicion Tree', 'KNN', 'Majority Voting']
print('10-fold cross validation')

for clf, label in zip([pipe1, clf2, pipe3, mv_clf], clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))