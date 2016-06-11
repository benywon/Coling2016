# -*- coding: utf-8 -*-
from public_functions import padding

__author__ = 'Bingning Wang'
__mail__ = 'research@bingning.wang'
import numpy as np
import tensorflow as tf

word_list = []
vocab_size = 1000
embedding_size = 50
for i in range(100):
    length = np.random.randint(20, 40)
    word_seq = np.random.randint(0, vocab_size, size=length)
    word_list.append(word_seq)
word_list, _ = padding(word_list)

raw_inputs = tf.placeholder(dtype=tf.int32)

W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), trainable=False,
                name="embedding")
_inputs = tf.nn.embedding_lookup(W, raw_inputs)
tf.nn.rnn_cell.BasicLSTMCell
sess = tf.Session()
sess.run(tf.initialize_all_variables())

cm = sess.run(_inputs, feed_dict={raw_inputs: word_list})
