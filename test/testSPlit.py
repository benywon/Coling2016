# -*- coding: utf-8 -*-
from RNN import RNN
from public_functions import *

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
pad_result = padding(text, return_matrix_for_size=True, max_len=max_length)
text = pad_result[0]
text_index = pad_result[1]

raw_inputs = tf.placeholder(tf.int32, shape=[max_sentence_length, max_length], name='sda')
seq_number_inputs = tf.placeholder(tf.bool, shape=[max_sentence_length, max_length], name='siho')
number_inputs = tf.placeholder(tf.int32, name='temp')
inputs_value_original = tf.split(0, max_sentence_length, value=raw_inputs)
inputs_value_original_index = tf.split(0, max_sentence_length, value=seq_number_inputs)
output = []

for i in range(14):
    ll = tf.gather(inputs_value_original, i)
    pp = tf.gather(inputs_value_original_index, i)
    jj = tf.boolean_mask(ll, pp)
    output.append(jj)

rnn = RNN(hidden_size=hidden_size, back_wards=True, return_all_hidden_states=True, input_size=embedding_size)


# ttt = []
#
#
def modles(pre, i):
    ll = tf.gather(inputs_value_original, i)
    pp = tf.gather(inputs_value_original_index, i)
    jj = tf.boolean_mask(ll, pp)
    rnn(inputs=_inputs)
    states = rnn.states
    return states


ccc = tf.scan(modles, tf.range(number_inputs), initializer=tf.zeros([hidden_size]))

W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), trainable=True,
                name="embedding")
init_output = []
for raw_input in output:
    _inputs = tf.nn.embedding_lookup(W, raw_input)
    rnn(inputs=_inputs)
    states = rnn.states
    init_output.append(states)

# mm = tf.boolean_mask(ll, seq_number_inputs)
sess = tf.Session()
ss = tf.trainable_variables()
for sss in ss:
    print sss.name
sess.run(tf.initialize_all_variables())

tes = gen_yes_no_array(14, max_sentence_length)
cc = sess.run(init_output,
              feed_dict={raw_inputs: text, number_inputs: 14, seq_number_inputs: text_index})

print cc
