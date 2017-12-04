import re
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
# from itertools import groupby
# import numpy as np
import multiprocessing as mp
import glob

# from functools import reduce
# import datetime

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
    # corpus_words = set()
    with open(file_name, 'r') as f:
        for line in f:
            if job_filter(line):
                tmp = job_extract(line)
                tmp = job_cleanup_format(tmp)
                tmp = job_split_content(tmp)
                processed_file.append(tmp)
                # corpus_words.add(tmp[1])
    return processed_file  # , corpus_words


def dump_processed_file(input_list):
    for document in input_list:
        id, tokens = document
        result_str = str(id) + ',' + ','.join(tokens) + '\n'
        yield result_str


if __name__ == '__main__':
    file_lists = glob.glob('/public/ruochwang2/split_files/*')

    pool = mp.Pool(processes=60)

    results = pool.map(process_file, file_lists)

    print('Finished processing files, dumping...')

    with open('huangxiaofengsb.txt', 'w') as f:
        for result in results:
            file_iter = dump_processed_file(result)
            for line in file_iter:
                f.write(line)
