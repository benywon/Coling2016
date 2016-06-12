# -*- coding: utf-8 -*-
__author__ = 'Bingning Wang'
__mail__ = 'research@bingning.wang'

import numpy as np

from IAGRU import *


class RNN:
    def __init__(self, hidden_size, input_size, init_scale=0.5, only_return_last_state=False, back_wards=False,
                 activate_function=tf.tanh):
        self.only_return_last_state = only_return_last_state
        self.back_wards = back_wards
        self.activate_function = activate_function
        self.init_scale = init_scale
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.initializer = tf.truncated_normal_initializer(stddev=init_scale)
        with tf.variable_scope('RNN_%d' % back_wards):
            self.W_hh = tf.get_variable('W_hh', shape=[self.hidden_size, self.hidden_size])
            self.W_ih = tf.get_variable('W_ih', shape=[self.input_size, self.hidden_size])
            self.b_h = tf.get_variable('b_h', shape=[self.hidden_size], initializer=tf.constant_initializer(0.0))

    def __call__(self, inputs):
        if self.back_wards:
            self._inputs = tf.reverse(inputs, dims=[True, False])
        else:
            self._inputs = inputs
        with tf.variable_scope('RNN', initializer=self.initializer):
            self._states = self._compute_hidden()

    def rnn_step(self, pre_state, x):
        pre_state = tf.reshape(pre_state, [1, self.hidden_size])
        x = tf.reshape(x, [1, self.input_size])
        hidden_state = self._inner_rnn_step(pre_state, x)
        h = tf.reshape(hidden_state, [self.hidden_size])
        return h

    def _inner_rnn_step(self, pre_state, x):
        hidden_temp = tf.matmul(pre_state, self.W_hh) + tf.matmul(x, self.W_ih) + self.b_h
        hidden_state = tf.tanh(hidden_temp)
        return hidden_state

    def _compute_hidden(self):
        """ Compute vanilla-RNN states and predictions. """

        with tf.variable_scope('states'):
            states = tf.scan(self.rnn_step, self.inputs,
                             initializer=self.initial_state, name='states')

        return states

    @property
    def initial_state(self):
        return tf.zeros([self.hidden_size],
                        name='initial_state')

    @property
    def inputs(self):
        """ A 2-D float32 placeholder with shape `[dynamic_duration, input_size]`. """
        return self._inputs

    @property
    def states(self):
        """ A 2-D float32 Tensor with shape `[dynamic_duration, hidden_layer_size]`. """
        if self.back_wards:
            states = tf.reverse(self._states, dims=[True, False])
        else:
            states = self._states

        return states


class GRU(RNN):
    def __init__(self, inner_activation=tf.tanh, **kwargs):
        # init parent attributes
        RNN.__init__(self, **kwargs)
        self.inner_activation = inner_activation
        with tf.variable_scope('GRU'):
            self.W_hz = tf.get_variable('W_hz', shape=[self.hidden_size, self.hidden_size])
            self.W_iz = tf.get_variable('W_iz', shape=[self.input_size, self.hidden_size])
            self.W_ir = tf.get_variable('W_ir', shape=[self.input_size, self.hidden_size])
            self.W_hr = tf.get_variable('W_hr', shape=[self.hidden_size, self.hidden_size])
            self.b_r = tf.get_variable('b_r', shape=[self.hidden_size], initializer=tf.constant_initializer(0.0))
            self.b_z = tf.get_variable('b_z', shape=[self.hidden_size], initializer=tf.constant_initializer(0.0))

    def _inner_rnn_step(self, pre_state, x):
        zt, rt = self.get_zt_rt(pre_state, x)
        rt_th1 = rt * pre_state
        hidden_temp = tf.matmul(x, self.W_ih) + self.b_h
        hidden_hat = self.inner_activation(hidden_temp + tf.matmul(rt_th1, self.W_hh))
        hidden_state = (1 - zt) * pre_state + zt * hidden_hat
        return hidden_state

    def get_zt_rt(self, hidden, x):
        zt_temp = tf.matmul(hidden, self.W_hz) + tf.matmul(x, self.W_iz) + self.b_z
        rt_temp = tf.matmul(hidden, self.W_hr) + tf.matmul(x, self.W_ir) + self.b_r
        zt = self.activate_function(zt_temp)
        rt = self.activate_function(rt_temp)
        return zt, rt


if __name__ == '__main__':

    hidden_size = 280
    embedding_size = 300
    vocab_size = 1000

    rnn = InnerAttentionGRU(hidden_size=hidden_size, input_size=embedding_size)
    rnn_back = RNN(hidden_size=hidden_size, back_wards=True, input_size=embedding_size)
    raw_inputs = tf.placeholder(tf.int32)
    W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), trainable=True,
                    name="embedding")
    _inputs = tf.nn.embedding_lookup(W, raw_inputs)
    rnn_back(inputs=_inputs)
    states = rnn_back.states
    previous = tf.gather(states, 0)
    rnn.set_attention(previous)
    rnn(inputs=_inputs)
    sess = tf.Session()
    ss = tf.trainable_variables()
    for sss in ss:
        print sss.name
    sess.run(tf.initialize_all_variables())
    print 'start done'
    for i in range(500):
        text = np.random.randint(0, 1000, size=150)
        cc = sess.run([states, rnn.states, previous], feed_dict={raw_inputs: text})
        print cc

    print type(cc)
