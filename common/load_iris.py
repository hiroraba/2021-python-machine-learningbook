from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

def load_iris():
    iris = datasets.load_iris()
    #3,4番目の特徴量
    X = iris.data[:, [2, 3]]
    #クラスラベル
    Y = iris.target
    
    return train_test_split(X, Y, test_size=0.3,random_state=1, stratify=Y)