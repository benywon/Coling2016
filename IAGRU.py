# -*- coding: utf-8 -*-
import tensorflow as tf

from RNN import GRU

__author__ = 'Bingning Wang'
__mail__ = 'research@bingning.wang'


class InnerAttentionGRU(GRU):
    def __init__(self, **kwargs):
        # init parent attributes
        GRU.__init__(self, **kwargs)
        with tf.variable_scope('IAGRU'):
            self.M_qz = tf.get_variable('M_qz', shape=[self.hidden_size, self.hidden_size])
            self.M_qr = tf.get_variable('M_qr', shape=[self.hidden_size, self.hidden_size])
        self.attention = tf.Variable(tf.zeros(self.hidden_size))

    def set_attention(self, attention):
        self.attention = tf.reshape(attention, [1, self.hidden_size])

    def get_zt_rt(self, hidden, x):
        attention_z = tf.matmul(self.attention, self.M_qz)
        attention_r = tf.matmul(self.attention, self.M_qr)
        zt_temp = tf.matmul(hidden, self.W_hz) + tf.matmul(x, self.W_iz) + self.b_z + attention_z
        rt_temp = tf.matmul(hidden, self.W_hr) + tf.matmul(x, self.W_ir) + self.b_r + attention_r
        zt = self.activate_function(zt_temp)
        rt = self.activate_function(rt_temp)
        return zt, rt