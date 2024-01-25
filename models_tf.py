#!/usr/bin/python3

import numpy as np
import tensorflow as tf
import keras_cv as KCV

def Attention(**kwargs):
    # args
    channel = kwargs.get('channel', 768)
    num_heads = kwargs.get('num_heads', 8)
    qkv_bias = kwargs.get('qkv_bias', False)
    drop_rate = kwargs.get('drop_rate', 0.1)
    assert channel % num_heads == 0
    # network
    inputs = tf.keras.Input((None, channel)) # inputs.shape = (b, s, c)
    results = tf.keras.layers.Dense(channel * 3, use_bias = qkv_bias)(inputs) # results.shape = (b, s, 3c)
    results = tf.keras.layers.Reshape((-1, 3, num_heads, channel // num_heads))(results) # results.shape = (b, s, 3, h, c // h)
    results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,3,1,4)))(results) # results.shape = (b, 3, h, s, c // h)
    q, k, v = tf.keras.layers.Lambda(lambda x: (x[:,0,...], x[:,1,...], x[:,2,...]))(results) # q.shape = (b, h, s, c // h)
    qk = tf.keras.layers.Lambda(lambda x, s: tf.matmul(x[0], tf.transpose(x[1], (0,1,3,2))) * s, arguments = {'s': (channel // num_heads) ** -0.5})([q, k]) # qk.shape = (b, h, s, s)
    attn = tf.keras.layers.Softmax(axis = -1)(qk)
    attn = tf.keras.layers.Dropout(rate = drop_rate)(attn)
    qkv = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.matmul(x[0], x[1]), (0, 2, 1, 3)))([attn, v]) # qkv.shape = (b, s, h, c // h)
    qkv = tf.keras.layers.Reshape((-1, channel))(qkv)
    results = tf.keras.layers.Dense(channel, use_bias = qkv_bias)(qkv)
    results = tf.keras.layers.Dropout(rate = drop_rate)(results)
    return tf.keras.Model(inputs = inputs, outputs = results, name = kwargs.get('name', None))

def ABlock(**kwargs):
    # args
    channel = kwargs.get('channel', 768)
    mlp_ratio = kwargs.get('mlp_ratio', 4)
    drop_rate = kwargs.get('drop_rate', 0.1)
    num_heads = kwargs.get('num_heads', 8)
    qkv_bias = kwargs.get('qkv_bias', False)
    groups = kwargs.get('groups', 1)
    # network
    inputs = tf.keras.Input((None, None, None, channel))
    # attention
    skip = inputs
    results = tf.keras.layers.LayerNormalization()(inputs) # results.shape = (batch, T, H, W, channel)
    shape = tf.keras.layers.Lambda(lambda x: tf.shape(x))(results)
    results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0],
                                                              tf.shape(x)[1] * tf.shape(x)[2] * tf.shape(x)[3],
                                                              tf.shape(x)[4])))(results) # results.shape = (batch, T * H * W, channel)
    results = Attention(**kwargs)(results)
    results = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], x[1]), output_shape = (None, None, None, channel))([results, shape]) # results.shape = (batch, T, H, W, channel)
    results = tf.keras.layers.Dropout(drop_rate)(results)
    results = tf.keras.layers.Add()([skip, results])
    # mlp
    skip = results
    results = tf.keras.layers.LayerNormalization()(results)
    results = tf.keras.layers.Conv3D(channel * mlp_ratio, kernel_size = (1,1,1), padding = 'same', activation = tf.keras.activations.gelu, groups = groups)(results)
    results = tf.keras.layers.Dropout(rate = drop_rate)(results)
    results = tf.keras.layers.Conv3D(channel, kernel_size = (1,1,1), padding = 'same', groups = groups)(results)
    results = tf.keras.layers.Dropout(rate = drop_rate)(results)
    results = tf.keras.layers.Add()([skip, results])
    return tf.keras.Model(inputs = inputs, outputs = results, name = kwargs.get('name', None))

def Extractor(**kwargs):
    # args
    in_channel = kwargs.get('in_channel', 3)
    hidden_channels = kwargs.get('hidden_channels', 512)
    depth = kwargs.get('depth', 12)
    mlp_ratio = kwargs.get('mlp_ratio', 4.)
    drop_rate = kwargs.get('drop_rate', 0.1)
    qkv_bias = kwargs.get('qkv_bias', False)
    num_heads = kwargs.get('num_heads', 8)
    groups = kwargs.get('groups', 1)
    # network
    inputs = tf.keras.Input((None, None, None, in_channel)) # inputs.shape = (batch, t, h, w, in_channel)
    results = tf.keras.layers.LayerNormalization()(inputs)
    results = tf.keras.layers.Conv3D(hidden_channels, kernel_size = (1, 1, 1), padding = 'same', activation = tf.keras.activations.gelu)(inputs) # results.shape = (batch, t, h, w, hidden_channels[0])
    results = tf.keras.layers.Dropout(rate = drop_rate)(results)
    # blocks
    for i in range(depth):
        results = ABlock(channel = hidden_channels, qkv_bias = qkv_bias, num_heads = num_heads, **kwargs)(results) # results.shape = (batch, 9, 9, 9, hidden_channels)
    results = tf.keras.layers.LayerNormalization()(results)
    results = tf.keras.layers.Conv3D(hidden_channels, kernel_size = (1, 1, 1), padding = 'same', activation = tf.keras.activations.tanh, groups = groups)(results) # results.shape = (batch, 1, 1, 1, hidden_channels[1])
    return tf.keras.Model(inputs = inputs, outputs = results, name = kwargs.get('name', None))

def Predictor(**kwargs):
    in_channel = kwargs.get('in_channel', 3)

    inputs = tf.keras.Input((None,None,None,in_channel))
    results = Extractor(**kwargs)(inputs)
    results = tf.keras.layers.Dense(1)(results)
    return tf.keras.Model(inputs = inputs, outputs = results)

def PredictorSmall(**kwargs):
    # args
    hidden_channels = kwargs.get('hidden_channels', 512)
    depth = kwargs.get('depth', 12)
    # network
    return Predictor(hidden_channels = hidden_channels, depth = depth, **kwargs)

def PredictorBase(**kwargs):
    # args
    hidden_channels = kwargs.get('hidden_channels', 512)
    depth = kwargs.get('depth', 24)
    # network
    return Predictor(hidden_channels = hidden_channels, depth = depth, **kwargs)

if __name__ == "__main__":
    inputs = np.random.normal(size = (1,9,9,9,4))
    uniformer = PredictorSmall(in_channel = 4, groups = 1)
    trainer.save('predictor.keras')
    outputs = trainer(inputs)
    print(outputs.shape)
