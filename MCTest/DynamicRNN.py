# -*- coding: utf-8 -*-
from RNN import RNN

__author__ = 'Bingning Wang'
__mail__ = 'research@bingning.wang'
import tensorflow as tf


class model:
    def __init__(self, embedding_matrix, margin=0.2, hidden_size=150, embedding_size=100, learning_rate=0.01,
                 max_sentence_length=1000,
                 max_sentence_number=100,
                 train_embedding=False):
        self.embedding_size = embedding_size
        self.margin = margin
        self.hidden_size = hidden_size
        self.max_sentence_number = max_sentence_length,
        self.learning_rate = learning_rate
        self.number_sentences_of_doc = 20
        with tf.variable_scope("embedding"):
            self.embedding_matrix = tf.Variable(embedding_matrix, name='EmbeddingMatrix', trainable=train_embedding)
        self._forward_rnn = RNN(hidden_size=self.hidden_size, input_size=self.embedding_size)
        self.input_document = tf.placeholder(tf.int32, shape=[max_sentence_number, max_sentence_length],
                                             name='input_document')
        self.input_document_pad_index = tf.placeholder(tf.int32, shape=[max_sentence_number, max_sentence_length],
                                                       name='input_document_pad_index')
        self.input_question = tf.placeholder(tf.int32, shape=[max_sentence_length], name='input_question')
        self.input_question_pad_index = tf.placeholder(tf.int32, shape=[max_sentence_length],
                                                       name='input_question_pad_index')
        self.candidate_answer = tf.placeholder(tf.int32, shape=[4, max_sentence_length], name='candidate_answer')
        self.candidate_answer_pad_index = tf.placeholder(tf.int32, shape=[4, max_sentence_length],
                                                         name='candidate_answer_pad_index')

    def set_sentence_number(self, number):
        """
        this is the real number of the sentence in a document
        :param number: number of sentences
        :return: none
        """
        self.number_sentences_of_doc = number

    def _train_function(self):
        output = []
        inputs_value_original = tf.split(0, self.max_sentence_number, value=self.input_document)
        inputs_value_original_index = tf.split(0, self.max_sentence_number, value=self.input_document_pad_index)
        for i in range(self.number_sentences_of_doc):
            ll = tf.gather(inputs_value_original, i)
            pp = tf.gather(inputs_value_original_index, i)
            jj = tf.boolean_mask(ll, pp)
            output.append(jj)
        document_embedding = [tf.nn.embedding_lookup(self.embedding_matrix, one_sentence) for one_sentence in output]
        for one_sent_embedding in document_embedding:
            one_sent_representation = self._forward_rnn(one_sent_embedding)
