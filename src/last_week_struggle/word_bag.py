
# coding: utf-8

import numpy as np
from itertools import groupby
import multiprocessing as mp
import pickle

def job_tokens_word_bag(doc_list):
    pid = doc_list[0]
    tokens = doc_list[1:]
    tokens = [token for token in tokens if token in words_set]
    indexs = [words.index(token) for token in tokens]
    freq_dict = dict()
    for key, grouped in groupby(indexs):
        freq_dict[key] = len(list(grouped))
    return pid, freq_dict


if __name__ == '__main__':

    pool = mp.Pool(processes=30)

    with open('proper_encoded.txt', 'r') as f:
        raw_data = f.readlines()
    
    raw_data = list(map(lambda i: i.strip().split(','), raw_data))

    with open('words.txt', 'r') as f:
        words = f.readlines()

    words = list(map(lambda i: i.strip(), words))

    words_set = set(words)


    id_doc = list(map(job_tokens_word_bag, raw_data))

    id_doc = pool.map(job_tokens_word_bag, raw_data)

    pickle.dump(id_doc, open('word_bags.p', 'wb'))

