# -*- coding: utf-8 -*-
import cPickle
import re

import numpy as np

from ModelBase import base_Location
from public_functions import padding

__author__ = 'Bingning Wang'
__mail__ = 'research@bingning.wang'


class mctestData():
    def __init__(self, RELOAD=False):
        self.word_embedding = None
        self.embedding_size = None
        self.vocabulary_size = None
        self.data_train = None
        self.data_test = None
        self.data_dev = None
        self.max_sentence_length = 0
        self.max_sentence_number = 0
        self.word2id = {}
        if RELOAD:
            self.reload_file()
        else:
            self.load_data()

    def load_data(self):
        with open(base_Location + 'data/MCTest/mc_data.pickle') as f:
            data = cPickle.load(f)

        def load_stats(data):
            sentence_numbers = [len(x[0]) for x in data[0]]
            sentence_numbers += [len(x[0]) for x in data[1]]
            sentence_numbers += [len(x[0]) for x in data[2]]
            self.max_sentence_number = max(sentence_numbers)
            sentence_lengths = [max([len(x) for x in y[0]]) for y in data[0]]
            sentence_lengths += [max([len(x) for x in y[0]]) for y in data[1]]
            sentence_lengths += [max([len(x) for x in y[0]]) for y in data[2]]
            self.max_sentence_length = max(sentence_lengths)

        null_vector = [0] * self.max_sentence_length

        def pad_data(document):
            document = list(document)
            sentence_number = len(document)
            document += [null_vector] * (self.max_sentence_number - sentence_number)
            padded_data, padded_index = padding(document, return_matrix_for_size=True)
            padded_index = np.asanyarray(padded_index, dtype=np.bool)

        load_stats(data)
        self.word_embedding = data[4]
        self.word2id = data[3]

    def reload_file(self, in_type='train', word2id=None, add_word=False):
        if word2id is None:
            word2id = dict()
            word2id['NULL'] = 0

        def transferWord(word_in):
            # chars = str(lmz.stem(word_in))
            # if chars.endswith('i'):
            #     chars = word_in
            return str(word_in).lower()

        def get_sentences(line):
            ps = re.sub(r'[^a-zA-Z0-9\.\,\?\!\']', ' ', line)  # Split on punctuations and hex characters
            s = re.sub(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', '\t', ps)  # Split on sentences
            ws = re.sub(r'(\W)', r' \1 ', s)  # Put spaces around punctuations
            ws = re.sub(r" ' ", r"'", ws)  # Remove spaces around '
            # ns = re.sub(r'(\d+)', r' <number> ', ws) # Put spaces around numbers
            hs = re.sub(r'-', r' ', ws)  # Replace hyphens with space
            rs = re.sub(r' +', r' ', hs)  # Reduce multiple spaces into 1
            rs = rs.lower().strip()
            return rs.split('\t')

        def only_words(line):
            ps = re.sub(r'[^a-zA-Z0-9\']', r' ', line)
            ws = re.sub(r'(\W)', r' \1 ', ps)  # Put spaces around punctuations
            ws = re.sub(r" ' ", r"'", ws)  # Remove spaces around '
            # ns = re.sub(r'(\d+)', r' <number> ', ws) # Put spaces around numbers
            hs = re.sub(r'-', r' ', ws)  # Replace hyphens with space
            rs = re.sub(r' +', r' ', hs)  # Reduce multiple spaces into 1
            rs = rs.lower().strip().split(' ')
            return rs

        def clean_sentence(line):
            ps = re.sub(r'[^a-zA-Z0-9\.\,\?\!\']', ' ', line)  # Split on punctuations and hex characters
            ws = re.sub(r'(\W)', r' \1 ', ps)  # Put spaces around punctuations
            ws = re.sub(r" ' ", r"'", ws)  # Remove spaces around '
            # ns = re.sub(r'(\d+)', r' <number> ', ws) # Put spaces around numbers
            hs = re.sub(r'-', r' ', ws)  # Replace hyphens with space
            rs = re.sub(r' +', r' ', hs)  # Reduce multiple spaces into 1
            rs = rs.lower().strip()
            return rs

        def get_answer_index(a):
            answer_to_index = {
                'A': 0,
                'B': 1,
                'C': 2,
                'D': 3,
            }
            return answer_to_index[a]

        train_file = './data/MCTest/1CR_mc500.train.tsv'
        valid_file = './data/MCTest/1CR_mc500.dev.tsv'
        testfile = './data/MCTest/1CR_mc500.test.tsv'
        answers_file = './data/MCTest/mc500.train.ans'
        valid_answers_file = './data/MCTest/mc500.dev.ans'
        test_answers_file = './data/MCTest/mc500.test.ans'
        questions = []

        if in_type == 'train':
            q_file = open(train_file, 'r')
            a_file = open(answers_file, 'r')
        elif in_type == 'test':
            q_file = open(testfile, 'r')
            a_file = open(test_answers_file, 'r')
        elif in_type == 'dev':
            q_file = open(valid_file, 'r')
            a_file = open(valid_answers_file, 'r')
        else:
            raise Exception('invalid input type!!!')
        print('start parse ' + in_type + ' file')
        questions_data = q_file.readlines()
        answers_data = a_file.readlines()
        assert (len(questions_data) == len(answers_data))
        for i in xrange(len(questions_data)):
            question_line = questions_data[i]
            answer_line = answers_data[i]

            question_pieces = question_line.strip().split('\t')
            assert (len(question_pieces) == 23)  # the default format can be divided by tag

            answer_pieces = answer_line.strip().split('\t')
            assert (len(answer_pieces) == 4)

            text = question_pieces[2]  # the document
            text = text.replace('\\newline', ' ')
            sentences = get_sentences(text)

            statements = list(list(list()))
            for s in sentences:
                tokens = s.strip().split()
                idVec = list()
                for token in tokens:
                    chars = token
                    if chars == '.':
                        continue
                    if chars not in word2id:
                        if add_word:
                            word2id[chars] = len(word2id)
                        else:
                            word2id[chars] = 0
                    idVec.append(word2id[chars])
                statements.append(idVec)
            for j in range(4):  # there are four questions one article
                q_index = (j * 5) + 3
                q_words = question_pieces[q_index]
                q_words = clean_sentence(q_words).split()
                q_words_vec = []
                for word in q_words[1:]:
                    chars = transferWord(word)
                    if chars == '.':
                        continue
                    if chars not in word2id:
                        if add_word:
                            word2id[chars] = len(word2id)
                        else:
                            word2id[chars] = 0

                    q_words_vec.append(word2id[chars])
                options = [
                    only_words(question_pieces[q_index + 1]),
                    only_words(question_pieces[q_index + 2]),
                    only_words(question_pieces[q_index + 3]),
                    only_words(question_pieces[q_index + 4]),
                ]
                correct = get_answer_index(answer_pieces[j])

                answerVec = list(list())
                for words in options:
                    wordv = []
                    for word in words:
                        chars = transferWord(word)
                        if chars == '.':
                            continue
                        if chars not in word2id:
                            if add_word:
                                word2id[chars] = len(word2id)
                            else:
                                word2id[chars] = 0
                        wordv.append(word2id[chars])
                    answerVec.append(wordv)
                # answer = options[correct]
                # answerVec = returnVec(answer)
                article_no = len(questions)
                questions.append([statements, q_words_vec, correct, answerVec, sentences, q_words])
        print('parse ' + in_type + ' done')
        return questions, word2id


if __name__ == '__main__':
    c = mctestData()
