""" 16.2 - リカーレンスネットワーク: シーケンスモデルの構築 """
import tensorflow as tf

tf.random.set_seed(1)
rnn_layer = tf.keras.layers.SimpleRNN(units=2, use_bias=True, return_sequences=True)
# 1次元目はバッチ数、2次元目はシーケンス数、3次元目は特徴量
rnn_layer.build(input_shape=(None, None, 5))
w_xh, w_oo, b_h = rnn_layer.weights
print('W_xh shape:', w_xh.shape)
print('W_oo shape:', w_oo.shape)
print('b_h shape:', b_h.shape)

x_seq = [[1.0]*5, [2.0]*5, [3.0]*5]
x_seq = tf.convert_to_tensor(x_seq, dtype=tf.float32)
print(x_seq)

output = rnn_layer(tf.reshape(x_seq, shape=(1, 3, 5)))
out_man = []
print(len(x_seq))
for t in range(len(x_seq)):
    xt = tf.reshape(x_seq[t], (1, 5))
    print('Time Step {} =>'.format(t))
    print(' Input   :', xt.numpy())
    
    ht = tf.matmul(xt, w_xh) + b_h
    print(' Hidden  :', ht.numpy())
    
    if t > 0:
        prev_o = out_man[t-1]
    else:
        prev_o = tf.zeros(shape=(ht.shape))
    ot = ht + tf.matmul(prev_o, w_oo)
    ot = tf.math.tanh(ot)
    out_man.append(ot)
    
    print(' output(Manual) :', ot.numpy())
    print(' Simple RNN output'.format(t), output[0][t].numpy())