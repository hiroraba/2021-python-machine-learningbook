""" 14.6.3 - Estimatorを使ってMNISTの手書き文字を分類する """
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

BUFFER_SIZE = 10000
BATCH_SIZE = 64
NUM_EPOCHS = 20

# 訓練データは60000個ある
steps_per_epoch = np.ceil(60000 / BATCH_SIZE)

# 入力画像とラベルの前処理
def preprocess(item):
    image = item['image']
    label = item['label']
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.reshape(image, (-1, ))
    return {'image-pixels': image}, label[..., tf.newaxis]

## 1 入力関数の定義
def train_input_fn():
    datasets = tfds.load(name='mnist')
    mnist_train = datasets['train']
    dataset = mnist_train.map(preprocess)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return dataset.repeat()

def eval_input_fn():
    datasets = tfds.load(name='mnist')
    mnist_train = datasets['test']
    dataset = mnist_train.map(preprocess).batch(BATCH_SIZE)
    return dataset

## 2 特徴量列を定義する
image_feature_column = tf.feature_column.numeric_column(key='image-pixels', shape=(28*28))

## 3 Estimatorを作成する
dnn_classifier = tf.estimator.DNNClassifier(feature_columns=[image_feature_column],
                                            hidden_units=[32, 16],
                                            n_classes=10,
                                            model_dir='models/minist-dnn/')

## 4 訓練と評価
dnn_classifier.train(input_fn=train_input_fn, steps=NUM_EPOCHS * steps_per_epoch)
eval_result = dnn_classifier.evaluate(input_fn=eval_input_fn)
print(eval_result) 
    