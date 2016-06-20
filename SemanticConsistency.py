# -*- coding: utf-8 -*-
import sys
import time

from IAGRU import *
from dataPocessor import dataprocess
from public_functions import *

__author__ = 'Bingning Wang'
__mail__ = 'research@bingning.wang'


class SemanticConsistencey(dataprocess):
    def __init__(self, **kwargs):
        # init parent attributes
        dataprocess.__init__(self, **kwargs)
        print 'data loaded'

    def Train(self):
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        with tf.Graph().as_default():
            sess = tf.Session()
            model = Model(embedding_matrix=self.wordEmbedding, train_embedding=False)
            # self.restore_model()
            saver = tf.train.Saver(tf.all_variables())
            sess.run(tf.initialize_all_variables())
            one_patch = self.train_data[129]
            print 'start train'
            for i in range(30):
                print 'round\t%d' % i
                loss_list = []
                right = 0.
                for one_data in self.test_data:
                    sample = one_data[1]
                    feed_dict = {
                        model.sentence0: sample[0],
                        model.sentence1: sample[1],
                        model.sentence2: sample[2],
                        model.sentence3: sample[3],
                        model.sentence4: sample[3],
                        model.test_sentence0: sample[0],
                        model.test_sentence1: sample[1],
                        model.test_sentence2: sample[2],
                        model.test_sentence3: sample[3],
                        model.test_sentence4: one_data[2],
                        model.test_sentence5: one_data[3],
                    }
                    _prediction = sess.run([model.value1, model.value2], feed_dict=feed_dict)
                    if (_prediction[0] > _prediction[1]):
                        prediction = 1
                    else:
                        prediction = 2
                    true_label = one_data[4]
                    if prediction == true_label:
                        right += 1
                acc = right / len(self.test_data)
                print 'accuracy%f' % acc
                for j, patch in enumerate(self.train_data):
                    sample = one_patch[2][0:5]
                    feed_dict = {
                        model.sentence0: sample[0],
                        model.sentence1: sample[1],
                        model.sentence2: sample[2],
                        model.sentence3: sample[3],
                        model.sentence4: sample[4],
                    }
                    _, loss = sess.run([model.train, model.loss], feed_dict=feed_dict)
                    loss_list.append(loss)
                    b = (
                        "Process\t" + str(j) + " in total:" + str(
                            len(self.train_data)) + ' loss: ' + str(
                            loss))
                    sys.stdout.write('\r' + b)
                loss_mean = np.mean(loss_list)
                print 'this round average loss=%f' % loss_mean
                path = saver.save(sess, checkpoint_prefix)
                print("Saved model checkpoint to {}\n".format(path))

    def restore_model(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(
                "/home/bingning/PycharmProjects/Coling2016/runs/1465721180/checkpoints/")
            saver.restore(sess, '/home/bingning/PycharmProjects/Coling2016/runs/1465721180/checkpoints/model')
            print 'done'


class Model:
    def __init__(self, embedding_matrix, margin=0.2, hidden_size=150, embedding_size=100, learning_rate=0.01,
                 train_embedding=False):
        self.embedding_size = embedding_size
        self.margin = margin
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.sentence0 = tf.placeholder(dtype=tf.int32, name='sentence0')
        self.sentence1 = tf.placeholder(dtype=tf.int32, name='sentence1')
        self.sentence2 = tf.placeholder(dtype=tf.int32, name='sentence2')
        self.sentence3 = tf.placeholder(dtype=tf.int32, name='sentence3')
        self.sentence4 = tf.placeholder(dtype=tf.int32, name='sentence4')
        self.sentence5 = tf.placeholder(dtype=tf.int32, name='sentence5')
        with tf.variable_scope("embedding"):
            self.embedding_matrix = tf.Variable(embedding_matrix, name='EmbeddingMatrix', trainable=train_embedding)
        self._forward_iagru = InnerAttentionGRU(hidden_size=self.hidden_size, input_size=self.embedding_size)
        self._train_function()
        self._test_function()

    def _train_function(self):
        doc_sentence_embedding = [tf.nn.embedding_lookup(self.embedding_matrix, raw_inputs) for raw_inputs in
                                  [self.sentence0, self.sentence1, self.sentence2, self.sentence3, self.sentence4]]

        doc_sentence_representation = []
        for i, sentence_embedding in enumerate(doc_sentence_embedding):
            self._forward_iagru(sentence_embedding)
            representation_whole = self._forward_iagru.states
            representation = self.get_RNN_representation(representation_whole)
            doc_sentence_representation.append(representation)
            self._forward_iagru.set_attention(representation)  # next step attention

        sentence_norm = [tensorflow_l1_norm(x) for x in doc_sentence_representation]

        def calc_los_pair(in_sentence1_norm, in_sentence2_norm):
            pair_margin = in_sentence2_norm - in_sentence1_norm
            return tf.maximum(0.0, self.margin - pair_margin)

        self.loss = 0
        for j in range(1, 5):
            for i in range(j):
                self.loss += calc_los_pair(sentence_norm[i], sentence_norm[j])
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00005)
        self.train = optimizer.minimize(self.loss)

    def _test_function(self):
        doc_sentence_embedding = [tf.nn.embedding_lookup(self.embedding_matrix, raw_inputs) for raw_inputs in
                                  [self.test_sentence0, self.test_sentence1, self.test_sentence2,
                                   self.test_sentence3, self.test_sentence4, self.test_sentence5]]
        for i, sentence_embedding in enumerate(doc_sentence_embedding[0:4]):
            self._forward_iagru(sentence_embedding)
            representation_whole = self._forward_iagru.states
            representation = self.get_RNN_representation(representation_whole)
            self._forward_iagru.set_attention(representation)
        self._forward_iagru(doc_sentence_embedding[4])
        representation_whole = self._forward_iagru.states
        sentence1_representation = self.get_RNN_representation(representation_whole)
        self._forward_iagru(doc_sentence_embedding[5])
        representation_whole2 = self._forward_iagru.states
        sentence2_representation = self.get_RNN_representation(representation_whole2)
        self.value1 = tensorflow_l1_norm(sentence1_representation)
        self.value2 = tensorflow_l1_norm(sentence2_representation)
        self.test = tf.less(self.value1, self.value2)

    @staticmethod
    def get_RNN_representation(input_sentence_representation):
        return tf.reduce_mean(input_sentence_representation, reduction_indices=0)


if __name__ == '__main__':
    sem = SemanticConsistencey(Reload=False)
    sem.Train()
