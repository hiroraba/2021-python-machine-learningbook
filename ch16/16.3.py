""" 16.3 - リカレントニューラルネットワークの実装: tensorFlowでのシーケンスモデルの構築 """

## 映画レビューデータの準備
from numpy.lib.npyio import recfromcsv
import tensorflow as tf
from tensorflow.python.keras.backend import rnn
from tensorflow.python.ops.gen_math_ops import mod
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
    
## 手順3: トークンを整数にエンコード
encoder = tfds.deprecated.text.TokenTextEncoder(token_counts)
example_str = 'This is an example'
print(encoder.encode(example_str))

## 手順3-A: 変換用の関数を定義
def encode(text_tensor, label):
    text = text_tensor.numpy()[0]
    encoded_text = encoder.encode(text)
    return encoded_text, label

## 手順3-B: encode関数をラッピングしてTensorFlow演算子に変換
def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))

ds_train = ds_raw_train.map(encode_map_fn)
ds_valid = ds_raw_valid.map(encode_map_fn)
ds_test = ds_raw_test.map(encode_map_fn)

# 訓練データの形状をチェック
tf.random.set_seed(1)
for example in ds_train.shuffle(1000).take(5):
    print('Sequence length:', example[0].shape)
    
## データごとにサイズがまちまちなため、揃える
ds_subset = ds_train.take(8)
for example in ds_subset:
    print('Individual size:', example[0].shape)
    
## サブセットをバッチに分割
ds_batched = ds_subset.padded_batch(4, padded_shapes=([-1], []))
for batch in ds_batched:
    print('Batch dimension:', batch[0].shape)
    
train_data = ds_train.padded_batch(32, padded_shapes=([-1], []))
valid_data = ds_valid.padded_batch(32, padded_shapes=([-1], []))
test_data = ds_test.padded_batch(32, padded_shapes=([-1], []))

from tensorflow.keras.layers import Embedding
model = tf.keras.Sequential()
# 埋め込み層
# input_dim: 一意な単語の数
# output_dim: 出力する埋め込み特徴量の数
# input_length: シーケンスの長さ
model.add(Embedding(input_dim=100, output_dim=6, input_length=20, name='embed-layer'))
model.summary()

#_________________________________________________________________
# Layer (type)                 Output Shape              Param #   
#=================================================================
# embed-layer (Embedding)      (None, 20, 6)             600       
#=================================================================
# Total params: 600
# Trainable params: 600
# Non-trainable params: 0
#_________________________________________________________________

## RNNモデルを構築する
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense
model = Sequential()
# 埋め込み層
model.add(Embedding(input_dim=1000, output_dim=32))
# 全結合リカレント層
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))
# 非リカレントの全結合層
model.add(Dense(1))

# 感情分析のためのRNNモデルを構築する
embedding_dim = 20

# 一意なトークンの数 + パディングプレースホルダ + トークンの集合に出現しない単語の特徴量
vocab_size = len(token_counts) + 2
tf.random.set_seed(1)

## モデルを構築
bi_lstm_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size,
                              output_dim=embedding_dim,
                              name='embed-layer'),
    # Bidirectionalにすると双方向になる（入力シーケンスが順方向と逆方向が2つになる）
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, 
                                                       name='lstm-layer'),
                                  name='bidir-lstm'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
bi_lstm_model.summary()

## コンパイルと訓練
bi_lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=['accuracy'])
#history = bi_lstm_model.fit(train_data, 
#                           validation_data=valid_data, 
#                           epochs=10)

## テストデータで評価
#test_results = bi_lstm_model.evaluate(test_data)
#print('Test Acc.: {:.2f}%'.format(test_results[1]*100))

from collections import Counter

# 感情分析に重要な情報が含まれているのは最後の段落という仮設をたて、最後の100個のトークンのみを学習データとする
def preprocess_datasets(ds_raw_train, ds_raw_valid, ds_raw_test, max_seq_length=None, batch_size=32):
    ## 手順2: 一意なトークンを特定
    tokenizer = tfds.deprecated.text.Tokenizer()
    token_counts = Counter()
    
    for example in ds_raw_train:
        tokens = tokenizer.tokenize(example[0].numpy()[0])
        if max_seq_length is not None:
            tokens = tokens[-max_seq_length:]
        token_counts.update(tokens)
        
    print('Vocab-size:', len(token_counts))
    
    ## 手順3: テキストをエンコード
    encoder = tfds.deprecated.text.TokenTextEncoder(token_counts)
    
    ## 手順3-A: 変換用の関数を定義
    def encode(text_tensor, label):
        text = text_tensor.numpy()[0]
        encoded_text = encoder.encode(text)
        if max_seq_length is not None:
            encoded_text = encoded_text[-max_seq_length:]
        return encoded_text, label

    ## 手順3-B: encode関数をラッピングしてTensorFlow演算子に変換
    def encode_map_fn(text, label):
        return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))
    
    ds_train = ds_raw_train.map(encode_map_fn)
    ds_valid = ds_raw_valid.map(encode_map_fn)
    ds_test = ds_raw_test.map(encode_map_fn)
    
    ## データセットをバッチ分割
    train_data = ds_train.padded_batch(batch_size, padded_shapes=([-1], []))
    valid_data = ds_valid.padded_batch(batch_size, padded_shapes=([-1], []))
    test_data = ds_test.padded_batch(batch_size, padded_shapes=([-1], []))
    
    return (train_data, valid_data, test_data, len(token_counts))

from tensorflow.keras.layers import Embedding, Bidirectional, SimpleRNN, LSTM, GRU

def build_rnn_model(embedding_dim, 
                    vocab_size, 
                    recurrent_type='SimpleRNN',
                    n_recurrent_units=64,
                    n_recurrent_layers=1,
                    bidirectional=True):
    
    tf.random.set_seed(1)
    
    ## モデルを構築
    model = tf.keras.Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='embed-layer'))
    
    for i in range(n_recurrent_layers):
        return_sequences = (i < n_recurrent_layers-1)
        
        if recurrent_type == 'SimpleRNN':
            recurrent_layer = SimpleRNN(units=n_recurrent_units,
                                        return_sequences=return_sequences,
                                        name='simprnn-layer-{}'.format(i))
        elif recurrent_type == 'LSTM':
            recurrent_layer = LSTM(units=n_recurrent_units,
                                        return_sequences=return_sequences,
                                        name='lstm-layer-{}'.format(i))
        elif recurrent_type == 'GRU':
            recurrent_layer = GRU(units=n_recurrent_units,
                                        return_sequences=return_sequences,
                                        name='gru-layer-{}'.format(i))
        
        if bidirectional:
            recurrent_layer = Bidirectional(recurrent_layer,name='bidir-' + recurrent_layer.name)
        
        model.add(recurrent_layer)
        
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    return model

batch_size = 32
embedding_dim = 20
max_seq_length = None
recurrent_type = "SimpleRNN"
bidirectional = False
train_data, valid_data, test_data, n = preprocess_datasets(ds_raw_train, 
                                                           ds_raw_valid, 
                                                           ds_raw_test,
                                                           max_seq_length=max_seq_length,
                                                           batch_size=batch_size)

vocab_size = n + 2

rnn_model = build_rnn_model(embedding_dim, 
                            vocab_size, 
                            recurrent_type=recurrent_type,
                            bidirectional=bidirectional)
rnn_model.summary()

rnn_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])
history = rnn_model.fit(train_data, validation_data=valid_data, epochs=10)
hist = history.history
results = rnn_model.evaluate(test_data)
print_results = 'Test Acc: {:.2f}%'.format(results[1]*100)
import matplotlib.pyplot as plt

x_arr = np.arange(len(hist['loss'])) + 1
fig = plt.figure(figsize=(12, 4))

ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist['loss'], '-o', label='Train Loss')
ax.plot(x_arr, hist['val_loss'], '--<', label='Validation Loss')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')

ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist['accuracy'], '-o', label='Train acc.')
ax.plot(x_arr, hist['val_accuracy'], '--<', label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
plt.title(print_results + ' ' +  recurrent_type + ' seq_length:' + str(max_seq_length) + ' bidirectional:' + str(bidirectional))
plt.show()