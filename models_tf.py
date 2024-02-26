#!/usr/bin/python3

import numpy as np
import tensorflow as tf

def MLPMixer(**kwargs):
  hidden_dim = kwargs.get('hidden_dim', 768)
  num_blocks = kwargs.get('num_blocks', 12)
  tokens_mlp_dim = kwargs.get('tokens_mlp_dim', 384)
  channels_mlp_dim = kwargs.get('channels_mlp_dim', 3072)

  inputs = tf.keras.Input((9,9,9,4)) # inputs.shape = (batch, 9, 9, 9, 4)
  results = tf.keras.layers.LayerNormalization()(inputs)
  results = tf.keras.layers.Reshape((9**3,4))(results) # results.shape = (batch, 9**3, 4)
  results = tf.keras.layers.Dense(hidden_dim, activation = tf.keras.activations.gelu)(results)
  results = tf.keras.layers.Dropout(rate = drop_rate)(results)
  for i in range(num_blocks):
    # 1) spatial mixing
    skip = results
    results = tf.keras.layers.LayerNormalization()(results)
    results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,1)))(results)
    results = tf.keras.layers.Dense(tokens_mlp_dim, activation = tf.keras.activations.gelu)(results)
    results = tf.keras.layers.Dense(9**3)(results)
    results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,1)))(results)
    results = tf.keras.layers.Add()([results, skip])
    # 2) channel mixing
    skip = results
    results = tf.keras.layers.LayerNormalization()(results)
    results = tf.keras.layers.Dense(channels_mlp_dim, activation = tf.keras.activations.gelu)(results)
    results = tf.keras.layers.Dense(hidden_dim)(results)
    results = tf.keras.layers.Add()([results, skip])
  results = tf.keras.layers.LayerNormalization()(results)
  results = tf.keras.lyaers.Lambda(lambda x: tf.math.reduce_mean(x, axis = -1))(results)
  return tf.keras.Model(inputs = inputs, outputs = results)

def Predictor(model_type = 'b16'):
  configs = {
    'b16': {'patch_size': 16, 'hidden_dim': 768, 'num_blocks': 12, 'tokens_mlp_dim': 384, 'channels_mlp_dim': 3072},
    'b32': {'patch_size': 32, 'hidden_dim': 768, 'num_blocks': 12, 'tokens_mlp_dim': 384, 'channels_mlp_dim': 3072},
    'l16': {'patch_size': 16, 'hidden_dim': 1024, 'num_blocks': 24, 'tokens_mlp_dim': 512, 'channels_mlp_dim': 4096},
  }
  # network
  return MLPMixer(**configs[model_type])

if __name__ == "__main__":
    inputs = np.random.normal(size = (1,9,9,9,4))
    predictor = Predictor()
    predictor.save('predictor.keras')
    outputs = predictor(inputs)
    print(outputs.shape)

