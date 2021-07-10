""" 16.3.2 - 文字レベルの言語モデルをTensorFlowで実装する """
import numpy as np
from tensorflow.python.ops.gen_math_ops import mod

with open('1268-0.txt', 'r') as fp:
    text = fp.read()
    
start_index = text.find('THE MYSTERIOUS ISLAND')
end_index = text.find('End of the Project Gutenberg')

# 全体のテキストから法定表示のものを除く
text = text[start_index:end_index]
char_set = set(text)

print('Total Length:', len(text))
print('Unique Charactors', len(char_set))

chars_sorted = sorted(char_set)
# 文字列辞書を整数へマッピング
char2int = {ch:i for i, ch in enumerate(chars_sorted)}
char_array = np.array(chars_sorted)

# テキストをエンコード
text_encoded = np.array([char2int[ch] for ch in text], dtype=np.int32)

print(text[:15], '== Encoding ==>', text_encoded[:15])
print(text_encoded[15:21], '== Reverse ==>', ''.join(char_array[text_encoded[15:21]]))

import tensorflow as tf
ds_text_encoded = tf.data.Dataset.from_tensor_slices(text_encoded)

# 次の単語を80個の候補から選ぶということは多クラス分類に相当するため、
# 入力変数と目的変数を準備する。

# 41単語をバッチとして分割する
seq_length = 40
chunk_size = seq_length + 1
ds_chunks = ds_text_encoded.batch(chunk_size, drop_remainder=True)
 
 # X（入力変数）とY(目的変数)を分離する
 # X:1文字目から40文字目 Y:2文字目から41文字目とする
def split_input_target(chunk):
     input_seq = chunk[:-1]
     target_seq = chunk[1:]
     return input_seq, target_seq
 
ds_sequences = ds_chunks.map(split_input_target)

# データセットをバッチに分割する
# 訓練データ1つに入力変数と目的変数が含まれる
BATCH_SIZE = 64
BUFFER_SIZE = 10000
ds = ds_sequences.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

 
def build_model(vocab_size, embedding_dim, rnn_units):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True),
        tf.keras.layers.Dense(vocab_size)
        ]
    )
    return model

## 訓練パラメータ
charset_size = len(char_array)
embedding_dim = 256
rnn_units = 512

tf.random.set_seed(1)
model = build_model(vocab_size=charset_size,
                    embedding_dim=embedding_dim,
                    rnn_units=rnn_units)

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
#model.fit(ds, epochs=20)
#model.save('1268-0')

model = tf.keras.models.load_model("1268-0")

# RNNモデルから文章を生成する
def sample(model, starting_str, len_generated_text=500, max_input_length=40, scale_factor=1.0):
    encoded_input = [char2int[s] for s in starting_str]
    encoded_input = tf.reshape(encoded_input, (1, -1))
    
    generated_str = starting_str
    
    model.reset_states()
    for i in range(len_generated_text):

        # 多クラス分類なので、ソフトマックスの出力結果をlogitsで受け取る
        logits = model(encoded_input)
        logits = tf.squeeze(logits, 0)
        
        scaled_logits = logits * scale_factor
        
        # 次の一文字を取得する
        new_char_indx = tf.random.categorical(scaled_logits, num_samples=1)
        new_char_indx = tf.squeeze(new_char_indx)[-1].numpy()
        
        #一文字を足す
        generated_str += str(char_array[new_char_indx])
        
        new_char_indx = tf.expand_dims([new_char_indx], 0)
        encoded_input = tf.concat([encoded_input, new_char_indx], axis=1)
        encoded_input = encoded_input[:, -max_input_length:]
        
    return generated_str

tf.random.set_seed(1)
print(sample(model, starting_str='The island'))

# scale_factorを小さくするとソフトマックスの出力確率が均一に近づくため、ランダム性が高くなる
print(sample(model, starting_str='The island', scale_factor=0.5))
        
    