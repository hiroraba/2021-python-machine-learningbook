""" 6.1 - パイプラインによるワークフローの効率化 """

""" 6.1.1 - Breast Cancer Wisconcin データセットを読み込む """

# 1 データセットを読み込む
import pandas as pd
from scipy.sparse.construct import rand

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                 'breast-cancer-wisconsin/wdbc.data', header=None)

# 2 30個の特徴量をNumpy配列Xに割り当てる。LabelEncoderを使って、元のクラスラベルの文字列表現を整数に割り当てる
from sklearn.preprocessing import LabelEncoder

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

# 3 データセットを訓練データとテストデータに分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

""" 6.1.2 パイプラインで変換器と推定器を結合する """

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

# 4 パイプラインの作成
## スケーリング、主成分分析、ロジスティック回帰を指定
pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(random_state=1, solver='lbfgs'))

pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))