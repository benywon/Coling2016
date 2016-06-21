# -*- coding: utf-8 -*-
from RNN import RNN
from public_functions import dot_vectors, cosine_tf

__author__ = 'Bingning Wang'
__mail__ = 'research@bingning.wang'
import tensorflow as tf


class DiscreteRNN:
    pass


class model:
    def __init__(self, embedding_matrix, margin=0.2, hidden_size=150, embedding_size=100, learning_rate=0.01,
                 paragraph_hidden_size=200,
                 max_sentence_length=1000,
                 max_sentence_number=100,
                 paragraph_activation_function=tf.sigmoid,
                 train_embedding=False):
        self.paragraph_activation_function = paragraph_activation_function
        self.paragraph_hidden_size = paragraph_hidden_size
        self.embedding_size = embedding_size
        self.margin = margin
        self.hidden_size = hidden_size
        self.max_sentence_number = max_sentence_length,
        self.learning_rate = learning_rate
        self.number_sentences_of_doc = 20
        with tf.variable_scope("embedding"):
            self.embedding_matrix = tf.Variable(embedding_matrix, name='EmbeddingMatrix', trainable=train_embedding)
        self._forward_rnn = RNN(hidden_size=self.hidden_size, input_size=self.embedding_size,
                                return_all_hidden_states=False, return_method='max')
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
        self.right_answer = tf.placeholder(tf.int32, shape=[1], name='right_answer')

        with tf.name_scope("paragraph_RNN"):
            self.W_hp = tf.Variable(
                tf.truncated_normal(shape=[self.hidden_size, self.paragraph_hidden_size], stddev=0.5),
                name='W_hp')
            self.W_pp = tf.Variable(
                tf.truncated_normal(shape=[self.paragraph_hidden_size, self.paragraph_hidden_size], stddev=0.5),
                name='W_pp')
            self.b_p = tf.Variable(tf.constant(0.0, shape=[self.paragraph_hidden_size]), name='b_h')
        self.paragraph_score_projection = tf.Variable(
            tf.truncated_normal(shape=[self.paragraph_hidden_size], mean=0.5, stddev=0.5),
            name='paragraph_score_projection')

        self._train_function()

    def set_sentence_number(self, number):
        """
        this is the real number of the sentence in a document
        :param number: number of sentences
        :return: none
        """
        self.number_sentences_of_doc = number

    def _train_function(self):
        # first we need to represent question
        question_input_idx = tf.boolean_mask(self.input_question, self.input_document_pad_index)
        question_embedding = tf.nn.embedding_lookup(self.embedding_matrix, question_input_idx)
        question_representation = self._forward_rnn(question_embedding)

        # then we represent the document

        def get_representation(split_number, value, value_pad_index):
            output = []
            inputs_value_original = tf.split(0, split_number, value=value)
            inputs_value_original_index = tf.split(0, split_number, value=value_pad_index)
            for i in range(self.number_sentences_of_doc):
                ll = tf.gather(inputs_value_original, i)
                pp = tf.gather(inputs_value_original_index, i)
                jj = tf.boolean_mask(ll, pp)
                output.append(jj)
            doc_embedding = [tf.nn.embedding_lookup(self.embedding_matrix, one_sentence) for one_sentence in output]
            doc_representation = []
            for one_sent_embedding in doc_embedding:
                one_sent_representation = self._forward_rnn(one_sent_embedding)
                doc_representation.append(one_sent_representation)
            return doc_representation

        document_representation = get_representation(self.max_sentence_number, self.input_document,
                                                     self.input_document_pad_index)
        answer_representation = get_representation(4, self.candidate_answer, self.candidate_answer_pad_index)

        paragraph_state = question_representation

        for representation in document_representation:
            # pre_state = tf.reshape(pre_state, [1, self.hidden_size])
            # x = tf.reshape(x, [1, self.input_size])
            hidden_temp = tf.matmul(paragraph_state, self.W_pp) + tf.matmul(representation, self.W_hp) + self.b_p
            paragraph_state_temp = self.paragraph_activation_function(hidden_temp)
            score = dot_vectors(paragraph_state_temp, self.paragraph_score_projection)
            paragraph_state = tf.cond(tf.less(score, 0.5), lambda: paragraph_state, lambda: paragraph_state_temp)
        score = []
        for candidate_answers in answer_representation:
            score.append(cosine_tf(candidate_answers, paragraph_state))
        scores = tf.pack(score)
        scores_total = tf.reduce_sum(scores)
        score_right = tf.gather(scores, self.right_answer)
        margin = score_right * 3 - scores_total
        self.loss = tf.maximum(0.0, self.margin - margin)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        self.train = optimizer.minimize(self.loss)
