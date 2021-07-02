""" 14.6 - TensorFlow Estimator """

# データの読み込み
import tensorflow as tf
import pandas as pd
from tensorflow._api.v2 import feature_column
from tensorflow.python.data.ops.readers import _DEFAULT_READER_BUFFER_SIZE_BYTES
from tensorflow.python.feature_column.feature_column_v2 import categorical_column_with_vocabulary_file

dataset_path = tf.keras.utils.get_file("auto-mpg.data",
                                       "http://archive.ics.uci.edu/ml/machine-learning"
                                       "-databases/auto-mpg/auto-mpg.data")
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'ModelYear', 'Origin']
df = pd.read_csv(dataset_path, 
                 names=column_names, 
                 na_values='?', 
                 comment='\t', 
                 sep=' ', 
                 skipinitialspace=True)

## NA行を削除
df = df.dropna()
df = df.reset_index(drop=True)

## 訓練データセットとテストデータセットに分割
import sklearn
import sklearn.model_selection

df_train, df_test = sklearn.model_selection.train_test_split(df, train_size=0.8)
train_stats = df_train.describe().transpose()
numeric_column_names = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration']

# 数字の特徴量は標準化する
df_train_norm, df_test_norm = df_train.copy(), df_test.copy()
for col_name in numeric_column_names:
    mean = train_stats.loc[col_name, 'mean']
    std = train_stats.loc[col_name, 'std']
    df_train_norm.loc[:, col_name] = (df_train_norm.loc[:, col_name] - mean)/std
    df_test_norm.loc[:, col_name] = (df_test_norm.loc[:, col_name] - mean)/std
    
print(df_train_norm.tail())

numeric_features = []
for col_name in numeric_column_names:
    numeric_features.append(tf.feature_column.numeric_column(key=col_name))    

# ModelYearを区間に分ける
feature_year = tf.feature_column.numeric_column(key='ModelYear')
bucketized_features = []
bucketized_features.append(tf.feature_column.bucketized_column(
    source_column=feature_year, 
    boundaries=[73, 76, 79]))

# Originをクラスラベルに変換する
feature_origin = tf.feature_column.categorical_column_with_vocabulary_list(
    key='Origin',
    vocabulary_list=[1, 2, 3]
)

# クラスラベルを全結合列に変換する ex. 1 => [1, 0, 0], 2 => [0, 1, 0]
categorical_indicator_features = []
categorical_indicator_features.append(
    tf.feature_column.indicator_column(feature_origin)
)

""" 14.6.2 - 既存のEstimatorを使った機械学習 """

# 既存のEstimatorの4つの手順
# 1 データを読み込むための入力関数を定義する
# 2 データセットを特徴量列に変換する
# 3 Estimatorをインスタンス化する
# 4 Estimatorのtrain, evaluate, predictメソッドを使う

# 1 データを読み込むための入力関数を定義する
# 入力特徴量とクラスラベルのタプルを返す
def train_input_fn(df_train, batch_size=8):
    df = df_train.copy()
    train_x, train_y = df, df.pop('MPG')
    dataset = tf.data.Dataset.from_tensor_slices((dict(train_x), train_y))
    
    return dataset.shuffle(1000).repeat().batch(batch_size)

ds = train_input_fn(df_train_norm)
batch = next(iter(ds))
print('Keys:', batch[0].keys())

# テストデータセット用の入力関数
def eval_input_fn(df_test, batch_size=8):
    df = df_test.copy()
    test_x, test_y = df, df.pop('MPG')
    dataset = tf.data.Dataset.from_tensor_slices((dict(test_x), test_y))
    
    return dataset.batch(batch_size)

# 2 データセットを特徴量列に変換する
all_feature_columns = (numeric_features + 
                       bucketized_features +
                       categorical_indicator_features)

# 3 Estimatorをインスタンス化する
# MPG(燃費)予測は回帰予測のためDNNRegressorを使う

regressor = tf.estimator.DNNRegressor(feature_columns=all_feature_columns,
                                      hidden_units=[32, 10],
                                      model_dir='models/autompg-dnnregressor/')

# 4 Estimatorのtrain, evaluate, predictメソッドを使う
import numpy as np
EPOCHS = 1000
BATCH_SIZE = 8
total_steps = EPOCHS * int(np.ceil(len(df_train) / BATCH_SIZE))
print('Training Steps:', total_steps)
regressor.train(input_fn=lambda:train_input_fn(df_train_norm, batch_size=BATCH_SIZE),steps=total_steps)

# trainを呼び出すと自動的に保存されるので、再度呼び出してみる
reloaded_regressor = tf.estimator.DNNRegressor(feature_columns=all_feature_columns,
                                      hidden_units=[32, 10],
                                      warm_start_from='models/autompg-dnnregressor/',
                                      model_dir='models/autompg-dnnregressor/')

# 予測性能を評価する
eval_results = reloaded_regressor.evaluate(
    input_fn=lambda:eval_input_fn(df_test_norm, batch_size=8)
)
print('Average-Loss: {:.4f}'.format(eval_results['average_loss']))

# 目的変数の値を予測する
pred_res = regressor.predict(input_fn=lambda: eval_input_fn(df_test_norm, batch_size=8))
print(next(iter(pred_res))) 

# ブースト決定木アルゴリズム
boosted_tree = tf.estimator.BoostedTreesRegressor(
    feature_columns=all_feature_columns,
    n_batches_per_layer=20,
    n_trees=200
)
boosted_tree.train(input_fn=lambda:train_input_fn(df_train_norm, batch_size=BATCH_SIZE))
eval_results = boosted_tree.evaluate(
    input_fn=lambda:eval_input_fn(df_test_norm, batch_size=8)
)
print('Average-Loss: {:.4f}'.format(eval_results['average_loss']))
