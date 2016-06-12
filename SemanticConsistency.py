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
            optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
            train_op = optimizer.minimize(model.loss)
            saver = tf.train.Saver(tf.all_variables())
            sess.run(tf.initialize_all_variables())
            print 'start train'
            for i in range(30):
                print 'round\t%d' % i
                loss_list = []
                for j, patch in enumerate(self.train_data):
                    sample = patch[2][0:5]
                    feed_dict = {
                        model.sentence[0]: sample[0],
                        model.sentence[1]: sample[1],
                        model.sentence[2]: sample[2],
                        model.sentence[3]: sample[3],
                        model.sentence[4]: sample[4],
                    }
                    _, loss = sess.run([train_op, model.loss], feed_dict=feed_dict)
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
        self.margin = margin
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.sentence = [tf.placeholder(dtype=tf.int32, name='sentence1')] * 5
        with tf.variable_scope("embedding"):
            self.embedding_matrix = tf.Variable(embedding_matrix, name='EmbeddingMatrix', trainable=train_embedding)
        doc_sentence_embedding = [tf.nn.embedding_lookup(self.embedding_matrix, raw_inputs) for raw_inputs in
                                  self.sentence]
        forward_iagru = InnerAttentionGRU(hidden_size=hidden_size, input_size=embedding_size)

        doc_sentence_representation = []
        for i, sentence_embedding in enumerate(doc_sentence_embedding):
            forward_iagru(sentence_embedding)
            representation_whole = forward_iagru.states
            representation = self.get_RNN_representation(representation_whole)
            doc_sentence_representation.append(representation)
            forward_iagru.set_attention(representation)  # next step attention

        sentence_norm = [tensorflow_l1_norm(x) for x in doc_sentence_representation]
        self.sentence_norm = sentence_norm

        def calc_los_pair(in_sentence1_norm, in_sentence2_norm):
            pair_margin = in_sentence2_norm - in_sentence1_norm
            return tf.maximum(0.0, self.margin - pair_margin)

        self.loss = 0
        for j in range(1, 5):
            for i in range(j):
                self.loss += calc_los_pair(sentence_norm[i], sentence_norm[j])
        optimizer = tf.train.AdadeltaOptimizer()
        global_step = tf.Variable(0, name="global_step", trainable=False)
        self.train = optimizer.minimize(self.loss, global_step=global_step)

    def get_RNN_representation(self, input_sentence_representation):
        return tf.reduce_mean(input_sentence_representation, reduction_indices=0)


if __name__ == '__main__':
    sem = SemanticConsistencey(Reload=False)
    sem.Train()
