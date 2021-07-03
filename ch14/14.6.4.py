""" 14.6.4 - 既存のKerasモデルからEstimatorを作成する """
from numpy.core.fromnumeric import reshape
import tensorflow as tf
import numpy as np

## データを作成
tf.random.set_seed(1)
np.random.seed(1)
x = np.random.uniform(low=-1, high=1, size=(200, 2))
y = np.ones(len(x))
y[x[:, 0] * x[:, 1] < 0] = 0
x_train = x[:100,:]
y_train = y[:100]
x_valid = x[100:, :]
y_valid = y[100:]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,), name='input-features'),
    tf.keras.layers.Dense(units=4, activation='relu'),
    tf.keras.layers.Dense(units=4, activation='relu'),
    tf.keras.layers.Dense(units=4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

## 1 入力関数を定義
def train_input_fn(x_train, y_train, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices(({'input-features': x_train}, y_train.reshape(-1,1)))
    return dataset.shuffle(100).repeat().batch(batch_size)

def eval_input_fn(x_test, y_test, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices(({'input-features': x_test}, y_test.reshape(-1,1)))
    return dataset.batch(batch_size)

## 2 特徴量列を定義
features = [tf.feature_column.numeric_column(key='input-features', shape=(2,))]

## 3 kerasのモデルからestimatorを生成
model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])
my_estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir='models/estimator-for-XOR')

## 4 Estimatorを使う
num_epochs = 200
batch_size = 2
steps_per_epoch = np.ceil(len(x_train)/batch_size)

my_estimator.train(input_fn=lambda: train_input_fn(x_train, y_train, batch_size), 
                   steps=num_epochs * steps_per_epoch)
results = my_estimator.evaluate(input_fn=lambda: eval_input_fn(x_valid, y_valid, batch_size))
print(results)