# -*- coding: utf-8 -*-
from RNN import RNN
from public_functions import padding

__author__ = 'Bingning Wang'
__mail__ = 'research@bingning.wang'
import numpy as np
import tensorflow as tf

hidden_size = 280
embedding_size = 300
vocab_size = 1000
text = [np.random.randint(0, 1000, size=np.random.randint(5, 100)) for i in range(20)]
text = padding(text)

rnn_back = RNN(hidden_size=hidden_size, back_wards=True, input_size=embedding_size)
raw_inputs = tf.placeholder(tf.int32)

inputs_value = tf.split()

W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), trainable=True,
                name="embedding")
_inputs = tf.nn.embedding_lookup(W, raw_inputs)
