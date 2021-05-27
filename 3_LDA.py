import os
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import re
from torchtext import data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import random
import tarfile
from tfidf_function import tfidf_process
from sys import version_info
if version_info.major == 2:
    import urllib as urldownload
else:
    import urllib.request as urldownload
from sklearn import metrics
import scipy.io as sio
def get_file_name(file_dir, name="dirs"):
    res_dirs = list()
    res_files = list()
    for root, dirs, files in os.walk(file_dir):
        res_dirs += dirs
        res_files += files

    if name == "dirs":
        return res_dirs
    else:
        return res_files

def replace(matched):
    return " " + matched.group("m") + " "

def tokenize_line_en(line):
   line = re.sub(r"\t", "", line)
   line = re.sub(r"^\s+", "", line)
   line = re.sub(r"\s+$", "", line)
   line = re.sub(r"<br />", "", line)
   line = re.sub(r"(?P<m>\W)", replace, line)
   line = re.sub(r"\s+", " ", line)
   return line.split()


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


class BasicDataset(data.Dataset):

    @classmethod
    def download_or_unzip(cls, root):
        if not os.path.exists(root):
            os.mkdir(root)
        path = os.path.join(root, cls.dirname)
        if not os.path.isdir(path):
            tpath = os.path.join(root, cls.filename)
            if not os.path.isfile(tpath):
                print('downloading')
                urldownload.urlretrieve(cls.url, tpath)
            with tarfile.open(tpath, 'r') as tfile:
                print('extracting')
                tfile.extractall(root)
        return os.path.join(path, '')

    @classmethod
    def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True, root='./data', **kwargs):
        path = cls.download_or_unzip(root)
        examples = cls(text_field, label_field, path=path, **kwargs).examples
        if shuffle: random.shuffle(examples)
        dev_index = -1 * int(dev_ratio*len(examples))

        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))


class NewsGroup(BasicDataset):

    '''newsgroups'''
#    url ='http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz'
#    filename = '20news-18828.tar.gz'
#    dirname = '20news-18828'
    '''mini_newsgroups'''
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/mini_newsgroups.tar.gz"
    filename = 'mini_newsgroups.tar.gz'
    dirname = 'mini_newsgroups'
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]
        path = self.dirname if path is None else path

        if examples is None:
            examples = []
            class_dirs = get_file_name(path)
            for class_dir_name in class_dirs:
                class_dir_path = os.path.join(path, class_dir_name)
                file_names = get_file_name(class_dir_path, "files")
                for file in file_names:
                    file_path = os.path.join(class_dir_path, file)
                    try:
                        with open(file_path) as f:
                            raw_data = f.read() 
                                                                               
                            if len(raw_data.split(' ')) > 400:
                                raw_data = ' '.join(raw_data.split(' ')[0:400])
                            
                            examples += [data.Example.fromlist([raw_data, class_dir_name], fields)]
                    except:
                        continue
        super(NewsGroup, self).__init__(examples, fields, **kwargs) 
print("Loading data...")
TEXT = data.Field(lower=True, tokenize=tokenize_line_en, include_lengths=True, batch_first=False, sequential=True,fix_length=1000)
LABEL = data.Field(sequential=False)
train, test = NewsGroup.splits(TEXT, LABEL)
print("Building vocabulary...")
TEXT.build_vocab(train)
LABEL.build_vocab(train)

train_iter, test_iter = data.BucketIterator.splits((train, test), sort_key = lambda x:len(x.text),
                                                   sort_within_batch=True,
                                                   batch_size=20, device=-1,
                                                   repeat = False)
x_transformed,total,Data_label = tfidf_process(train_iter=train_iter,val_iter=[],test_iter=test_iter)




'''Trln'''
load_fn_20news_x = 'LDA_data/20newsdata.mat'
load_fn_20news_y = 'LDA_data/20newslabel.mat'

load_data_20news_x = sio.loadmat(load_fn_20news_x)['news']
load_data_20news_y = sio.loadmat(load_fn_20news_y)['label'].reshape(-1,)

'''auto'''

load_fn_20news_x = 'LDA_data/20newsdata_auto.mat'
load_fn_20news_y = 'LDA_data/20newslabel_auto.mat'

load_data_20news_x_auto = sio.loadmat(load_fn_20news_x)['news']
load_data_20news_y_auto = sio.loadmat(load_fn_20news_y)['label'].reshape(-1,)





'''Trln'''
lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
y_pred = lda.fit(load_data_20news_x, load_data_20news_y).predict(load_data_20news_x)
accuracy = metrics.accuracy_score(load_data_20news_y, y_pred)
print("Trln",accuracy) 


'''auto'''
lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
y_pred_auto = lda.fit(load_data_20news_x_auto, load_data_20news_y_auto).predict(load_data_20news_x_auto)
accuracy = metrics.accuracy_score(load_data_20news_y_auto, y_pred_auto)
print("auto",accuracy) 

'''tfidf'''
lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
y_pred = lda.fit(x_transformed.toarray(), Data_label).predict(x_transformed.toarray())
accuracy = metrics.accuracy_score(y_pred, Data_label)
print("tfidf",accuracy) 

