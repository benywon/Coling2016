# -*- coding: utf-8 -*-
from IAGRU import *
from dataPocessor import dataprocess

__author__ = 'Bingning Wang'
__mail__ = 'research@bingning.wang'


class SemanticConsistencey(dataprocess):
    def __init__(self, **kwargs):
        # init parent attributes
        dataprocess.__init__(self, **kwargs)


class model:
    def __init__(self, embedding_matrix, hidden_size=150, learning_rate=0.01, train_embedding=False):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.sentence = [tf.placeholder(dtype=tf.int32, name='sentence1')] * 5
        self.embedding_matrix = tf.Variable(embedding_matrix, name='Embedding', trainable=train_embedding)
        doc_sentence_embedding = [tf.nn.embedding_lookup(self.embedding_matrix, raw_inputs) for raw_inputs in
                                  self.sentence]
