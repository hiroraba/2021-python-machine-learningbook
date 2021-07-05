""" 16.3 - リカレントニューラルネットワークの実装: tensorFlowでのシーケンスモデルの構築 """

## 映画レビューデータの準備
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd

df = pd.read_csv('movie_data.csv', encoding='utf-8')

## 手順1: datasetの作成
target = df.pop('sentiment')
ds_raw = tf.data.Dataset.from_tensor_slices((df.values, target.values))

### 調査
for ex in ds_raw.take(3):
    print(ex[0].numpy()[0][:50], ex[1])
    
tf.random.set_seed(1)
ds_raw = ds_raw.shuffle(50000, reshuffle_each_iteration=False)
ds_raw_test = ds_raw.take(25000)
ds_raw_train_valid = ds_raw.skip(25000) # 訓練、検証データセット
ds_raw_train = ds_raw_train_valid.take(20000) # 訓練データセット
ds_raw_valid = ds_raw_train_valid.skip(20000) # 検証データセット

## 手順2: 一意なトークンを特定
from collections import Counter
tokenizer = tfds.deprecated.text.Tokenizer()
token_counts = Counter()

for example in ds_raw_train:
    tokens = tokenizer.tokenize(example[0].numpy()[0])
    token_counts.update(tokens)
    
print('Vocab-size:', len(token_counts))