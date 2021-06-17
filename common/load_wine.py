import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def load_wine():
    # wineデータセットを読み込む
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

    # 列名の設定
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']
    # Xに特徴量、yにクラスラベル
    # iloc:行番号または列番号を指定してデータを抽出
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:,0].values

    # 訓練データとテストデータに分割(30%をテストデータにする)   
    # stratify=y 出力値（クラスラベル）が均等になるように
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


    # 標準化: 平均0, 標準偏差1になるようにスケーリングする
    # 多くのアルゴリズムの場合は標準化のほうが効果的
    #   - 多くの線形モデルが0,または0に近い乱数で初期化するので重みが学習しやすい
    #   - 外れ値(例外的な値を持つモデル)に関する有益な情報が保持されるため、外れ値から受ける影響がすくなくなる

    # 標準化のインスタンス
    stdsc = StandardScaler()

    return stdsc.fit_transform(X_train), stdsc.fit_transform(X_test), y_train, y_test