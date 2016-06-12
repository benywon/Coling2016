# -*- coding: utf-8 -*-
import cPickle

import numpy as np

from public_functions import clean_word, clean_str, clean_str_remove, random_sample
from word2vec import word2vector, load_word2vec300withoutLOOP

__author__ = 'benywon'

datadir = 'data'


class dataprocess:
    def __init__(self, Reload=False,
                 padding=False,
                 max_length=50,
                 embedding_size=100):
        self.padding = padding
        self.Max_length = max_length
        self.EmbeddingSize = embedding_size
        self.wordEmbedding = []
        self.reload = Reload
        self.dataDir = ''
        self.train_data = []
        self.test_data = []
        self.dev_data = []
        self.word2id = {'_NULL_': 0}
        self.vocabularySize = 0
        self.train_size = 0
        self.test_size = 0
        self.dev_size = 0
        self.dump_filepath = datadir + '/roc_story.pickle'
        if Reload:
            self.build_data()
        else:
            self.load_data_from_file()

    def build_data(self):
        print 'start reload data'
        train_path = datadir + '/train.tsv'
        test_path = datadir + '/test.tsv'
        dev_path = datadir + '/dev.tsv'

        def deal_one_line(line, istrain):
            tab_splits = line.split('\t')
            doc_id = tab_splits[0]
            if istrain:
                title = self.get_sentence_id_list(sentence=tab_splits[1])
                doc = [self.get_sentence_id_list(sentence=x) for x in tab_splits[2:7]]
                return [doc_id, title, doc]
            else:
                context = [self.get_sentence_id_list(sentence=x) for x in tab_splits[1:5]]
                candidate1 = self.get_sentence_id_list(sentence=tab_splits[5])
                candidate2 = self.get_sentence_id_list(sentence=tab_splits[6])
                true_answer = int(tab_splits[7])
                return [doc_id, context, candidate1, candidate2, true_answer]

        def deal_one_data(filepath, istrain=True):
            print 'load ' + filepath + '.....'
            with open(filepath, 'rb') as f:
                lines = f.readlines()
            data = [deal_one_line(x, istrain) for x in lines[1:]]
            return data

        self.train_data = deal_one_data(train_path)
        noisy_sentence_pool = [x[2][4] for x in self.train_data]
        for one_sample in self.train_data:
            one_sample[2].append(random_sample(noisy_sentence_pool))
        self.test_data = deal_one_data(test_path, istrain=False)
        self.dev_data = deal_one_data(dev_path, istrain=False)
        self.calc_data_stat()
        self.build_word2vec()
        self.transform_to_numpy_format()
        self.dump_data_to_file()
        print 'data process done!!'

    def dump_data_to_file(self):
        print 'start saving data to..' + self.dump_filepath
        obj = {'train': self.train_data, 'test': self.test_data, 'dev': self.dev_data, 'word2id': self.word2id,
               'word2vec': self.wordEmbedding}

        with open(self.dump_filepath, 'wb') as f:
            cPickle.dump(obj, f, cPickle.HIGHEST_PROTOCOL)

    def load_data_from_file(self):
        print 'start load data from ' + self.dump_filepath
        with open(self.dump_filepath, 'rb') as f:
            obj = cPickle.load(f)
        self.train_data = obj['train']
        self.test_data = obj['test']
        self.dev_data = obj['dev']
        self.word2id = obj['word2id']
        self.wordEmbedding = obj['word2vec']
        print 'data loaded'

    def transform_to_numpy_format(self):
        for i in xrange(self.train_size):
            for j in range(6):
                self.train_data[i][2][j] = np.asarray(self.train_data[i][2][j], dtype='int32')
        for i in xrange(self.test_size):
            for j in range(4):
                self.test_data[i][1][j] = np.asarray(self.test_data[i][1][j], dtype='int32')
            self.test_data[i][2] = np.asarray(self.test_data[i][2], dtype='int32')
            self.test_data[i][3] = np.asarray(self.test_data[i][3], dtype='int32')
        for i in xrange(self.dev_size):
            for j in range(4):
                self.dev_data[i][1][j] = np.asarray(self.dev_data[i][1][j], dtype='int32')
            self.dev_data[i][2] = np.asarray(self.dev_data[i][2], dtype='int32')
            self.dev_data[i][3] = np.asarray(self.dev_data[i][3], dtype='int32')

    def calc_data_stat(self):
        self.train_size = len(self.train_data)
        self.test_size = len(self.test_data)
        self.dev_size = len(self.dev_data)

    def get_sentence_id_list(self, sentence, add_vacabulary=True, divider=' ', max_length=None):
        sentence = clean_str(sentence)
        return [self.get_word_id(word, add_vacabulary) for word in sentence.split(divider)][0:max_length]

    def get_word_id(self, word, add_vacabulary):
        if word.endswith(','):
            word = word.replace(',', '')
        word = str(word)
        if word.endswith('?'):
            word = word.replace('?', '')
        if word.endswith('!'):
            word = word.replace('!', '')
        word = clean_word(word)
        if word in self.word2id:
            return self.word2id[word]
        else:
            if add_vacabulary:
                self.word2id[word] = len(self.word2id)
                return self.word2id[word]
            else:
                return 0

    def build_word2vec(self):
        print 'embedding length:' + str(self.EmbeddingSize)
        self.vocabularySize = len(self.word2id)
        assert len(self.word2id) > 0, 'you have not load word2id!!'
        self.wordEmbedding = np.zeros(shape=(self.vocabularySize, self.EmbeddingSize), dtype='float32')
        if not (self.EmbeddingSize == 300):
            word2vec = word2vector(self.EmbeddingSize)
            for (word, word_id) in self.word2id.items():
                vec = word2vec.returnWordVec(clean_str_remove(word))
                self.wordEmbedding[word_id] = vec
        else:
            self.wordEmbedding = load_word2vec300withoutLOOP(self.word2id)


if __name__ == '__main__':
    data = dataprocess(Reload=False)
    print data.dataDir
