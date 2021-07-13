""" 17.3.3 - 生成器ネットワークと識別器ネットワークを実装する """
import tensorflow as tf
import numpy as np
from tensorflow._api.v2 import image
from tensorflow.keras import layers
from tensorflow.python import training
from tensorflow.python.ops.gen_array_ops import pad
from tensorflow.python.ops.gen_math_ops import mod
from tensorflow.python.ops.variable_scope import default_variable_creator

if tf.config.list_physical_devices('GPU'):
    device_name = tf.test.gpu_device_name()
else:
    device_name = '/CPU:0'

# 生成器ネットワーク
# 1 入力として、サイズ20のベクトルzを受け取る
# 2 全結合層にzを渡してサイズの6272に拡大する
# 3 転置畳み込みで 7 * 7 * 128(空間次元7*7でチャネルが128)のテンソルに変換する
# 4 転置畳み込みで28*28になるまで特徴量をアップサンプリングする
# 5 最後に活性化関数を使う

def make_dcgan_generator(z_size=20, output_size=(28, 28, 1), n_filters=128, n_blocks=2):
    size_factor = 2**n_blocks
    hidden_size = (output_size[0]//size_factor, output_size[1]//size_factor)
    
    model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(z_size,)), # 入力層
         tf.keras.layers.Dense(units=n_filters*np.prod(hidden_size), use_bias=False), # 全結合層
         tf.keras.layers.BatchNormalization(), # バッチ正規化
         tf.keras.layers.LeakyReLU(), # Leaky ReLU
         tf.keras.layers.Reshape((hidden_size[0], hidden_size[1], n_filters)), # 行列を28*28に変換
         tf.keras.layers.Conv2DTranspose(filters=n_filters, 
                                         kernel_size=(5, 5), 
                                         strides=(1, 1), 
                                         padding='same', 
                                         use_bias=False), # 畳み込み(空間次元は不変)
         tf.keras.layers.BatchNormalization(),
         tf.keras.layers.LeakyReLU()
         ])
    nf = n_filters
    
    for i in range(n_blocks):
        nf = nf//2
        model.add(tf.keras.layers.Conv2DTranspose(filters=nf, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())
        
    model.add(tf.keras.layers.Conv2DTranspose(filters=output_size[2], kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    
    return model

# 識別器ネットワーク
# 1 28*28*1の画像を受け取る
# 2 3つの畳み込み層を使って空間次元を半分にしてチャネルを倍にする
# 3 バッチ正規化、ReLU、ドロップアウト層を通す
# 4 最後に7*7のカーネルと単一のフィルタを使って1*1*1に削減する

def make_dcgan_discriminator(input_size=(28,28,1), n_filters=64, n_blocks=2):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_size),
        tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(5, 5), strides=(1, 1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3)
    ])
    
    nf = n_filters
    for i in range(n_blocks):
        nf = nf*2
        model.add(tf.keras.layers.Conv2D(filters=nf, kernel_size=(5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=(7, 7), padding='valid'))
    model.add(tf.keras.layers.Reshape((1,)))
    
    return model

import tensorflow_datasets as tfds

mnist_bldr = tfds.builder('mnist')
mnist_bldr.download_and_prepare()
mnist = mnist_bldr.as_dataset(shuffle_files=False)

def preprocess(ex, mode='uniform', z_size=20):
    image = ex['image']
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image*2 - 1.0
    if mode == 'uniform':
        input_z = tf.random.uniform(shape=(z_size,), minval=-1.0, maxval=1.0)
    elif mode == 'normal':
        input_z = tf.random.uniform(shape=(z_size))
        
    return input_z, image

gen_model = make_dcgan_generator()
gen_model.summary()

disc_model = make_dcgan_discriminator()
disc_model.summary()

num_epochs = 100
batch_size = 128
image_size = (28, 28)
z_size = 20
mode_z = 'uniform'
lambda_gp = 10.0

tf.random.set_seed(1)
np.random.seed(1)

## データセットの準備
mnist_trainset = mnist['train']
mnist_trainset = mnist_trainset.map(preprocess)
mnist_trainset = mnist_trainset.shuffle(10000)
mnist_trainset = mnist_trainset.batch(batch_size, drop_remainder=True)

## モデルの準備
with tf.device(device_name):
    gen_model = make_dcgan_generator()
    gen_model.build(input_shape=(None, z_size))
    
    disc_model = make_dcgan_discriminator()
    disc_model.build(input_shape=(None, np.prod(image_size)))
    
import time

## オプティマイザー
g_optimizer = tf.keras.optimizers.Adam(0.0002)
d_optimizer = tf.keras.optimizers.Adam(0.0002)

if mode_z == 'uniform':
    fixed_z = tf.random.uniform(shape=(batch_size, z_size), minval=-1.0, maxval=1.0)
elif mode_z == 'normal':
    fixed_z = tf.random.uniform(shape=(batch_size, z_size))
    
def create_samples(g_model, input_z):
    g_output = g_model(input_z, training=False)
    images = tf.reshape(g_output, (batch_size, *image_size))
    return (images+1)/2.0

all_losses = []
epoch_samples = []
start_time = time.time()

for epoch in range(1, num_epochs+1):
    epoch_losses = []
    
    for i,(input_z, input_real) in enumerate(mnist_trainset):
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            g_output = gen_model(input_z, training=True)
            d_critics_real = disc_model(input_real, training=True)
            d_critics_fake = disc_model(g_output, training=True)
            
            ## 生成器ネットワークの損失関数の計算
            g_loss = -tf.math.reduce_mean(d_critics_fake)
            
            ## 識別器ネットワークの損失関数の計算
            # 本物、偽物の出力の距離を損失とする（EM距離）
            # お互いの出力値が近づく＝本物と偽物の差分が減る と損失が少なくなる
            d_loss_real = -tf.math.reduce_mean(d_critics_real)
            d_loss_fake = tf.math.reduce_mean(d_critics_fake)
            d_loss = d_loss_real + d_loss_fake
            
            ## 勾配ペナルティ
            with tf.GradientTape() as gp_tape:
                alpha = tf.random.uniform(shape=[d_critics_real.shape[0], 1, 1, 1], minval=0.0, maxval=1.0)
                interpolated = (alpha*input_real + (1-alpha)*g_output) # 偽物と本物で補完された画像からなるバッチができる
                gp_tape.watch(interpolated)
                # 補完データの損失関数の計算
                d_critics_intp = disc_model(interpolated)
            
            # 補完データの勾配
            grads_intp = gp_tape.gradient(d_critics_intp, [interpolated, ])[0]
            grads_intp_l2 = tf.sqrt(tf.reduce_sum(tf.square(grads_intp), axis=[1, 2, 3]))
            
            grads_penalty = tf.reduce_mean(tf.square(grads_intp_l2 - 1.0))
            d_loss = d_loss + grads_penalty
            
        ## 最適化：勾配を計算して、適用
        d_grads = d_tape.gradient(d_loss, disc_model.trainable_variables)
        d_optimizer.apply_gradients(grads_and_vars=zip(d_grads, disc_model.trainable_variables))
        
        g_grads = g_tape.gradient(g_loss, gen_model.trainable_variables)
        g_optimizer.apply_gradients(grads_and_vars=zip(g_grads, gen_model.trainable_variables))
        epoch_losses.append((g_loss.numpy(), d_loss.numpy(), d_loss_real.numpy(), d_loss_fake.numpy()))
        
    all_losses.append(epoch_losses)
    print('Epoch {:03d} | ET {:.2f} | Avg Losses >> G/D {:.4f}/{:.4f} [D-Real: {:.4f} D-Fake: {:.4f}]'.format(epoch, (time.time() - start_time)/60, *list(np.mean(all_losses[-1], axis=0))))
    epoch_samples.append(create_samples(gen_model, fixed_z).numpy())