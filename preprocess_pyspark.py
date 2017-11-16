
# coding: utf-8

# In[1]:


import findspark
findspark.init()
import pyspark


# In[2]:


import re
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from itertools import groupby
import numpy as np


# In[3]:


sc = pyspark.SparkContext(appName="test")


# In[4]:


sc


# In[5]:


filePath = 'hdfs://0.0.0.0:9000/user/bitnami/group_project_data/data_simple.xml'

raw_data = sc.textFile(filePath)


# In[6]:


id_pattern = re.compile('<row Id=\"([\d]*)\"')
content_pattern = re.compile('Text=\"([\W\w]*)\"')
noise_pattern = re.compile('&[#]*[\w]*;')


def job_filter(input_str) :
    return id_pattern.search(input_str) and content_pattern.search(input_str)

def job_extract(input_str):
    postid = id_pattern.search(input_str).group(1)
    content = content_pattern.search(input_str).group(1)
    content = noise_pattern.sub('', content)
    return postid, content

def job_cleanup_format(input_tuple):
    postid, content = input_tuple
    return int(postid), content.strip()

records_with_content = raw_data.filter(job_filter)
raw_id_content = records_with_content.map(job_extract)
cleaned_id_content = raw_id_content.map(job_cleanup_format)


# In[7]:


# see what happened now
cleaned_id_content.take(4)


# In[8]:


tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# to lower case, no punctuation, stemmed, filter stop words
# this python version is highly depend on nltk, may consider sth else later
def job_split_content(input_tuple):
    postid, input_str = input_tuple
    input_str = input_str.lower()
    raw_tokens = tokenizer.tokenize(input_str)
    stemmed_tokens = [stemmer.stem(token) for token in raw_tokens]
    stemmed_tokens = map(stemmer.stem, raw_tokens)
    stemmed_tokens_without_stopword = filter(lambda i: i not in stop_words, stemmed_tokens)
    return postid, list(stemmed_tokens_without_stopword)

# this step seems to use up a lot of time!
id_tokens =  cleaned_id_content.map(job_split_content)

# check first two records
# id_tokens.take(2)


# In[9]:


class CorpusWordsSet(pyspark.AccumulatorParam):
    def zero(self, value=set()):
        return set()
    
    def addInPlace(self, acc1, acc2):
        return acc1 | acc2;


# In[10]:


corpus_words = sc.accumulator(set(), CorpusWordsSet())

def job_add_tokens_to_dict(records):
    _, tokens = records
    corpus_words.add( set(tokens))
    
document_count_rdd = id_tokens.map(job_add_tokens_to_dict)
DOCUMENT_COUNT = document_count_rdd.count() # force accumulator to run
    
corpus_words = list(corpus_words.value)
WORD_COUNT = len(corpus_words)


# In[11]:


WORD_COUNT_broadcasted = sc.broadcast(WORD_COUNT)
corpus_words_broadcasted = sc.broadcast(corpus_words)

def job_word_to_index(input_tuple):
    postid, tokens = input_tuple
    content_indexed = [corpus_words_broadcasted.value.index(token) for token in tokens] # [index1, index2, index1, index1, index3], for example
    content_freq = dict()
    for key, grouped in groupby(content_indexed):
        content_freq[key] = len(list(grouped))
    indexs = set(content_freq.keys())
    result_list = list()
    for i in range(WORD_COUNT_broadcasted.value):
        if i in indexs:
            result_list.append(content_freq.get(i))
        else:
            result_list.append(0)
        
    return postid, np.array(result_list)

id_freq_dicts = id_tokens.map(job_word_to_index)
id_freq_dicts.take(2)


# In[12]:


#def test_job_word_to_index(input_tuple):
#    postid, tokens = input_tuple
#    content_indexed = [corpus_words_broadcasted.value.index(token) for token in tokens] # [index1, index2, index1, index1, index3], for example
#    content_freq = dict()
#    for key, grouped in groupby(content_indexed):
#        content_freq[key] = len(list(grouped))
#    indexs = set(content_freq.keys())
#    result_list = list()
#    for i in range(WORD_COUNT_broadcasted.value):
#        if i in indexs:
#            result_list.append(content_freq.get(i))
#        else:
#            result_list.append(0)
#        
#    return postid, np.array(result_list)[:20]
#
#test_id_freq_dicts = id_tokens.map(test_job_word_to_index)
#test_id_freq_dicts.take(2)


# 
# Now **id_freq_dicts** is a term frequency (TF) matrix. Next we will try to convert it into a TF-IDF matrix    
# I would say IDF is the most time consuming part for now. Because it required aggregation among all records.(which is distributed among clusters)

# In[13]:


id_freq_dicts.cache()


# In[14]:


#test_id_freq_dicts.cache()


# Never mind, this damn part takes more than 20 mins and doesn't seems to stop anyway...    
# We HAVE TO change a way of aggregate...(
# 
# PS: I looked into log file afterwards. Each job takes like 0.3 second to finish. Then this 5k words would take 25 mins to finish. T T

# In[15]:


def job_cal_document_freq(word_freq_array1, word_freq_array2):
    document_freq1 = word_freq_array1!=0
    document_freq2 = word_freq_array2!=0
    return document_freq1*1 + document_freq2

get_ipython().magic(u'time document_freq = id_freq_dicts.map(lambda i:i[1]).reduce(job_cal_document_freq)')


# 0.342 second seems accpetable...

# In[16]:


idf_array = np.log(DOCUMENT_COUNT/document_freq)


# In[17]:


idf_array


# In[18]:


#test_idf_array = idf_array[:20]


# In[19]:


#test_idf_array


# In[20]:


id_tfidf = id_freq_dicts.map(lambda i: (i[0], i[1] * idf_array))


# In[21]:


id_tfidf.take(2)


# In[22]:


#test_id_tfidf = test_id_freq_dicts.map(lambda i: (i[0], i[1] * test_idf_array))


# In[23]:


#test_id_tfidf.take(2)


# In[24]:


#usefortest_r_whole = np.linalg.qr(test_id_tfidf.map(lambda x:x[1]).collect())[1]


# In[25]:


#usefortest_r_whole.shape


# In[26]:


##test_r_whole


# In[27]:


def get_divide_count(row_num, col_num):
    dividedcount = int(row_num / (col_num * 1.5))
    if dividedcount < 1:
        dividedcount = 1;
    return dividedcount;


# In[28]:


DividedCount = get_divide_count(DOCUMENT_COUNT, 20);


# In[29]:


def get_split_array(dividedcount):
    return np.linspace(1,1,dividedcount)


# In[30]:


weight_array = get_split_array(DividedCount)


# In[31]:


split_id_tfidf = id_tfidf.randomSplit(weight_array, 0)


# In[32]:


len(split_id_tfidf)


# In[33]:


##test_split_id_tfidf = test_id_tfidf.randomSplit(weight_array, 0)


# In[34]:


##len(test_split_id_tfidf)


# In[35]:


##test_split_id_tfidf[0].count()


# In[36]:


def cal_r_martix(m1, m2):
    return np.linalg.qr(np.concatenate((m1,  m2)))[1]


# In[37]:


def cal_split_r_martix(m1):
    return np.linalg.qr(m1.map(lambda x:x[1]).collect())[1]


# In[38]:


distributed_r = reduce(cal_r_martix, map(cal_split_r_martix, split_id_tfidf))


# In[39]:


distributed_r


# In[40]:


##test_distributed_r = reduce(cal_r_martix, map(cal_split_r_martix, test_split_id_tfidf))


# In[41]:


##test_distributed_r

