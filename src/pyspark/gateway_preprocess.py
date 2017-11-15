import re
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from itertools import groupby
import numpy as np
import multiprocessing as mp
import glob
from functools import reduce
import datetime

id_pattern = re.compile('<row Id=\"([\d]*)\"')
content_pattern = re.compile('Text=\"([\W\w]*)\"')
noise_pattern = re.compile('&[#]*[\w]*;')

tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def job_filter(input_str):
    return id_pattern.search(input_str) and content_pattern.search(input_str)


def job_extract(input_str):
    postid = id_pattern.search(input_str).group(1)
    content = content_pattern.search(input_str).group(1)
    content = noise_pattern.sub('', content)
    return postid, content


def job_cleanup_format(input_tuple):
    postid, content = input_tuple
    return int(postid), content.strip()


def job_split_content(input_tuple):
    postid, input_str = input_tuple
    input_str = input_str.lower()
    raw_tokens = tokenizer.tokenize(input_str)
    stemmed_tokens = [stemmer.stem(token) for token in raw_tokens]
    stemmed_tokens = map(stemmer.stem, raw_tokens)
    stemmed_tokens_without_stopword = filter(lambda i: i not in stop_words, stemmed_tokens)
    return postid, list(stemmed_tokens_without_stopword)


def process_file(file_name):
    processed_file = list()
    corpus_words = set()
    with open(file_name, 'r') as f:
        for line in f:
            if job_filter(line):
                tmp = job_extract(line)
                tmp = job_cleanup_format(tmp)
                tmp = job_split_content(tmp)
                processed_file.append(tmp)
                corpus_words.add(tmp[1])
    return processed_file, corpus_words


def job_word_to_index(input_tuple):
    postid, tokens = input_tuple
    content_indexed = [total_corpus_words.index(token) for token in tokens]
    content_freq = dict()
    for key, grouped in groupby(content_indexed):
        content_freq[key] = len(list(grouped))
    indexs = set(content_freq.keys())
    result_list = list()
    for i in range(WORD_COUNT):
        if i in indexs:
            result_list.append(content_freq.get(i))
        else:
            result_list.append(0)

    return postid, np.array(result_list)


file_lists = glob.glob('/public/ruochwang2/split_file/*')

FILE_NUM = 100

pool = mp.Pool(processes=90)

results = pool.map(process_file, file_lists[:FILE_NUM])

print('Finished processing files, start to count words...')

total_corpus_words = reduce(lambda a, b: a | b, map(lambda i: i[1], results))

print('Finished counting words, start to count WORD_COUNT and DOCUMENT_COUNT...')

WORD_COUNT = len(total_corpus_words)
total_corpus_words = list(total_corpus_words)
DOCUMENT_COUNT = reduce(lambda i, j: len(i[1]) + len(j[1]), results)

print('WORD_COUNT = {word}, DOCUMENT_COUNT = {document}'.format(word=WORD_COUNT, document=DOCUMENT_COUNT))

print('Building up tf matrix...')

tf_matrix = pool.map(job_word_to_index, results)

del results
print('TF Matrix complete...')

print('Calculating document frequency...')
document_freq = reduce(lambda i, j: i * 1 + j, pool.map(lambda i: i != 0, tf_matrix))

print('Calculating IDF')
idf_array = np.log(DOCUMENT_COUNT / document_freq)

del document_freq

print('Calculating TFIDF...')
tfidf = pool.map(lambda i: (i[0], i[1] * idf_array), tf_matrix)
del tf_matrix

words_dump = '\n'.join(total_corpus_words)
words_file_path = '~/words_dump.txt'

print('dump words to {}'.format(words_file_path))
with open(words_file_path, 'w') as f:
    f.write(words_dump)

tfidf_dump = pool.map(lambda i: str(i[0]) + ',' + ','.join(i[1]), tfidf)

del tfidf

tfidf_file_path = '~/tfidf.csv'

print('dump tfidf to {}'.format(tfidf_file_path))
with open(tfidf_file_path, 'w') as f:
    f.writelines(tfidf_dump)

print('Done! TIME: {}'.format(datetime.datetime.now()))



