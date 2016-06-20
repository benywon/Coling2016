# -*- coding: utf-8 -*-
from public_functions import padding

__author__ = 'Bingning Wang'
__mail__ = 'research@bingning.wang'
import numpy as np
import tensorflow as tf

hidden_size = 280
max_length = 200
max_sentence_length = 50
embedding_size = 300
vocab_size = 1000
text = [np.random.randint(0, 1000, size=np.random.randint(5, 100)) for i in range(max_sentence_length)]
text = padding(text, return_matrix_for_size=True, max_len=max_length)

raw_inputs = tf.placeholder(tf.int32, shape=[max_sentence_length, max_length], name='sda')
number_inputs = tf.placeholder(tf.bool, shape=[max_length], name='siho')
inputs_value_original = tf.split(0, max_sentence_length, value=raw_inputs)
mm = tf.boolean_mask(ll, number_inputs)
ll = tf.gather(inputs_value[0], 0)
mm = tf.boolean_mask(ll, number_inputs)
sess = tf.Session()
ss = tf.trainable_variables()
for sss in ss:
    print sss.named
sess.run(tf.initialize_all_variables())

cc = sess.run([ll, mm], feed_dict={raw_inputs: text[0], number_inputs: np.asarray(text[1][0], dtype=np.bool)[0]})

print cc
