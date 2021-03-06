# -*- coding: utf-8 -*-
import cPickle
import os
import random
import re
import threading

import numpy as np
import tensorflow as tf
import theano
import theano.tensor as T

dtype = theano.config.floatX
rng = np.random.RandomState(2016)
__author__ = 'benywon'


def find_number(string):
    pattern = re.compile(r'.*?(\d+).*')
    res = re.findall(pattern, string)
    return int(res[0])


def clean_string(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower()


def clean_word(word):
    """
    clean a word
    :param word:  like alss's
    :return: word
    """
    word = word.replace(r'(', '')
    word = word.replace(r')', '')
    word = word.replace(r'{', '')

    word = word.replace(r'}', '')
    if len(re.findall(r'(\d+)', word)) > 0:
        return "NUMBER"
    else:
        word = word.replace(r"'s", "")
        word = word.replace(r"?", "")
        return word


def clean_str_remove(string):
    """
    Tokenization/string cleaning
    :param string: in string
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", "", string)
    string = re.sub(r"\s{2,}", "", string)
    return string.lower()


def clean_str(string):
    """
    Tokenization/string cleaning
    :param string: in string
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def dump_file(obj, filepath):
    print 'dump file:' + filepath
    with open(filepath, 'wb') as f:
        cPickle.dump(obj, f, cPickle.HIGHEST_PROTOCOL)
    print 'done!'


def load_file(filepath):
    print 'load file:' + filepath
    with open(filepath, 'rb') as f:
        obj = cPickle.load(f)
    return obj


def sample_weights(sizeX, sizeY, low=-1., high=1., method='svd'):
    """
        it has been proved that the max singular value of a matirx can not
        exceed 1 for the non exploding RNN issues
        :param high: high bound
        :param low: low bound
        :param sizeY: the initiation matrix size y
        :param sizeX:the initiation matrix size x
        :return: the svd matrix remove max value
        """
    if method == 'random':
        return rng.normal(size=(sizeX, sizeY), loc=0.5, scale=0.3)
    else:
        values = np.ndarray([sizeX, sizeY], dtype=dtype)
        for dx in xrange(sizeX):
            vals = np.random.uniform(low=low, high=high, size=(sizeY,))
            # vals_norm = np.sqrt((vals**2).sum())
            # vals = vals / vals_norm
            values[dx, :] = vals
        _, svs, _ = np.linalg.svd(values)
        # svs[0] is the largest singular value
        values = values / svs[0]
        return values


def cosine(x, y):
    return T.dot(x, y) / (T.sqrt(T.sum(x ** 2)) * T.sqrt(T.sum(y ** 2)))


def cosine_tf(x, y):
    return dot_vectors(x, y) / (tensorflow_l2_norm(x) * tensorflow_l2_norm(y))


def batch_cosine(x, y):
    return T.batched_dot(x, y) / (T.sqrt(T.sum(x ** 2)) * T.sqrt(T.sum(y ** 2)))


def padding(sequence, pads=0, max_len=None, dtype='int32', return_matrix_for_size=False):
    v_length = [len(x) for x in sequence]  # every sentence length
    max_len = max(v_length) if max_len is None else max_len
    v_length = map(lambda z: z if z <= max_len else max_len, v_length)
    x = (np.ones((len(sequence), max_len)) * pads).astype(dtype)
    for idx, s in enumerate(sequence):
        trunc = s[:max_len]
        x[idx, :len(trunc)] = trunc
    if return_matrix_for_size:
        v_matrix = np.asanyarray([map(lambda item: 1 if item < line else 0, range(max_len)) for line in v_length],
                                 dtype=dtype)
        return x, v_matrix
    return x, np.asarray(v_length, dtype='int32')


def pad_index2distribution(index, classes):
    li = [0] * classes
    li[index] = 1
    return li


def gen_yes_no_array(num_real, max_length):
    keep = [1] * num_real + [0] * (max_length - num_real)
    return np.asarray(keep, dtype=np.bool)


def get_dir_files(director):
    file_list = []
    for parent, dirnames, filenames in os.walk(director):
        for filename in filenames:
            file_list.append(os.path.join(parent, filename))
    return file_list


class sort_utils(threading.Thread):
    """
    sort_map_before should be {id,list()}
    the list() is value list ,and it should be match if we sort the value
    """

    def __init__(self, sort_list_before, threads_number):
        threading.Thread.__init__(self)
        self.sort_list_before = sort_list_before
        self.length = len(sort_list_before)
        self.threads_number = threads_number
        self.sorted_list_tuple = []

    def sort_list(self):
        sorted_list_index = sorted(xrange(len(self.sort_list_before)), key=lambda k: self.sort_list_before[k])
        sorted_list = [self.sort_list_before[x] for x in sorted_list_index]
        sorted_list_tuple = []
        i = 1
        one_patch = [sorted_list_index[0]]
        while i < self.length:
            if sorted_list[i] == sorted_list[i - 1]:
                one_patch.append(sorted_list_index[i])
            else:
                sorted_list_tuple.append(one_patch)
                one_patch = [sorted_list_index[i]]
            i += 1
        sorted_list_tuple.append(one_patch)
        self.sorted_list_tuple = sorted_list_tuple

    @staticmethod
    def sort_map(sort_map_before):
        threads_number = len(sort_map_before[0])
        threads = []
        for i in xrange(threads_number):
            unsort_list = [x[i] for x in sort_map_before]
            sorter = sort_utils(unsort_list, i)
            threads.append(sorter)
            sorter.start()
        # check whether all threads done
        for thread in threads:
            while thread.is_alive():
                pass
        sorted_tuple_list = [x.sorted_list_tuple for x in threads]
        # then it is the tuple we calculate
        pass  # TODO

    def run(self):  # Overwrite run() method, put what you want the thread do here
        self.sort_list()


def random_sample(in_list):
    length = len(in_list)
    return in_list[random.randint(0, length - 1)]


def dot_vectors(vec1, vec2):
    vec1 = tf.transpose(tf.reshape(vec1, [1, tf.shape(vec1)[0]]))
    vec2 = tf.reshape(vec2, [1, tf.shape(vec2)[0]])
    res = tf.matmul(vec2, vec1)
    return tf.gather(tf.gather(res, 0), 0)


def tensorflow_l1_norm(input_tensor):
    abs_value = tf.abs(input_tensor)
    return tf.reduce_sum(abs_value)


def tensorflow_l2_norm(input_tensor):
    square_value = tf.square(input_tensor)
    return tf.sqrt(tf.reduce_sum(square_value))


if __name__ == '__main__':
    print gen_yes_no_array(19, 40)
