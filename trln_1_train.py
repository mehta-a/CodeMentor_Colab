from __future__ import print_function, division
import time
import torchtext
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torchtext.vocab import Vectors
from tqdm import tqdm
import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import LSTM, LSTM, Bidirectional
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import keras.backend as K
import csv
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler,scale
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sys import version_info
if version_info.major == 2:
    import urllib as urldownload
else:
    import urllib.request as urldownload

import tarfile
import re
import random
from torchtext import data
keras.backend.clear_session()
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
    url ='http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz'
    filename = '20news-18828.tar.gz'
    dirname = '20news-18828'
    '''mini_newsgroups'''
#    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/mini_newsgroups.tar.gz"
#    filename = 'mini_newsgroups.tar.gz'
#    dirname = 'mini_newsgroups'
    
    
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
class TRLN():
    def __init__(self):
        self.img_rows = 1000
        self.img_cols = 300 
        self.num_classes = 500
        self.img_shape = (self.img_rows, self.img_cols)


        optimizer = Adam(0.0002, 0.5)
        optimizer_2 = Adam(0.0001, 0.5)
       

        self.recognition= self.build_recognition()      
        self.embedding= self.build_embedding()
        self.embedding.compile(loss=[self.mutual_info_loss_n],
            optimizer=optimizer_2)


        rec_input = Input(shape=(self.num_classes,))
        rec_out = self.recognition(rec_input)
        emb_out = self.embedding(rec_out)

        self.combined = Model(rec_input,emb_out)
        self.combined.compile(loss=[self.mutual_info_loss],
            optimizer=optimizer)
        
        emb_input = Input(shape=self.img_shape)
        emb_out2 = self.embedding(emb_input)
        rec_out2 = self.recognition(emb_out2)

        self.combined2 = Model(emb_input,rec_out2)
        self.combined2.compile(loss=[self.mutual_info_loss],optimizer=optimizer)

    def build_embedding(self):        
        emb_input = Input(shape=self.img_shape)
        model = Sequential()
        model.add(LSTM(256,input_shape=self.img_shape,return_sequences=True))
        model.add(Bidirectional(LSTM(256)))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.num_classes, activation='softmax'))
        C = model(emb_input) 
        model.summary()

        return Model(emb_input, C)

10
0    def build_recognition(self):
    
        C = Input(shape=(self.num_classes,))
        
        model = Sequential()
        model.add(Dense(np.prod((self.num_classes,1)), input_dim=self.num_classes))
        model.add(Reshape((self.num_classes,1)))
        model.add(LSTM(256,return_sequences=True))
        model.add(Bidirectional(LSTM(256)))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        
        
        rec_out = model(C)
        
        model.summary()

        return Model(C, rec_out)


    def mutual_info_loss(self, c, c_given_x):
        eps = 1e-8
        conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
        entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

        return conditional_entropy + entropy
       
    def mutual_info_loss_n(self, c, c_given_x):
        eps = 1e-8
        conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
        entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

        return -1*(conditional_entropy + entropy)

    def save_model(self):
        self.embedding.save('embedding.h5')
        self.recognition.save('recognition.h5')
        

    def train(self, epochs, batch_size=20, sample_interval=50):
        loss2 = 0
        count = 0
        Data_label = [0]
        text_num=1
        print("Loading data...")
        TEXT = data.Field(lower=True, tokenize=tokenize_line_en, include_lengths=True, batch_first=False, sequential=True,fix_length=1000)
        LABEL = data.Field(sequential=False)
        train, test = NewsGroup.splits(TEXT, LABEL)

        print("Building vocabulary...")
        TEXT.build_vocab(train)
        LABEL.build_vocab(train)
        with open("txt/label.txt", 'a') as f:
            print(LABEL.vocab.itos,file=f)


        train_iter, test_iter = data.BucketIterator.splits((train, test), sort_key = lambda x:len(x.text),
                                                           sort_within_batch=True,
                                                           batch_size=batch_size, device=-1,
                                                           repeat = False)
        url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.vec'           
        TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))
        
        ems =TEXT.vocab.itos
        



        with open('loss.csv', 'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['loss1','losss2','epoc'])
        
        for epoch in range(epochs):            
            for batch in train_iter:
                x = batch.text
                x, lengths = x
                
                if len(x[1])<20:
                    continue
                embeddings = nn.Embedding(TEXT.vocab.vectors.size()[0], TEXT.vocab.vectors.size()[1])
                embeddings.weight.data.copy_(TEXT.vocab.vectors)
                embeddings =embeddings(x)
                embedded = embeddings.transpose(0, 1)
                embeddings1=embedded[0:int(len(embedded)/2),:,:]
                embeddings2=embedded[int(len(embedded)/2):20,:,:]
                #loss1               
                emb_out = self.embedding.predict(embeddings1.detach().numpy())   
                sampled = np.argmax(emb_out, axis=1).reshape(-1, 1)
                sampled = to_categorical(sampled, num_classes=self.num_classes)
                
                sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)                
                sampled_labels = to_categorical(sampled_labels, num_classes=self.num_classes)
                             
                loss1_ = self.combined.train_on_batch(sampled_labels, sampled_labels)
                loss1 = self.combined.train_on_batch(emb_out, emb_out)
                


                #loss2
                if epoch >= 4:
                    emb_input=embeddings1.detach().numpy()        
                    emb_out2 = self.embedding.predict(embeddings2.detach().numpy())                    

                    loss2 = self.embedding.train_on_batch(emb_input, emb_out2)
                        
    
                with open('loss.csv', 'a', newline='') as csvFile:
                    writer = csv.writer(csvFile)                
                    writer.writerow([loss1,loss2, count])
                print ("loss1",loss1,"loss2:",loss2,"epoc:",count)
                count+=1
            for batch in test_iter:
                x = batch.text
                x, lengths = x
                
                if len(x[1])<20:
                    continue
                embeddings = nn.Embedding(TEXT.vocab.vectors.size()[0], TEXT.vocab.vectors.size()[1])
                embeddings.weight.data.copy_(TEXT.vocab.vectors)
                embeddings =embeddings(x)
                embedded = embeddings.transpose(0, 1)
                embeddings1=embedded[0:int(len(embedded)/2),:,:]
                embeddings2=embedded[int(len(embedded)/2):20,:,:]
                #loss1               
    
                sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
                sampled_labels = to_categorical(sampled_labels, num_classes=self.num_classes)             
                emb_out = embeddings1
                loss1 = self.combined2.train_on_batch(embeddings1.detach().numpy(), embeddings1.detach().numpy())
    
                #loss2
                if epoch >= 4:
                    emb_input=embeddings1.detach().numpy()        
                    emb_out2 = self.embedding.predict(embeddings2.detach().numpy())                    

                    loss2 = self.embedding.train_on_batch(emb_input, emb_out2)
                        
    
                with open('loss.csv', 'a', newline='') as csvFile:
                    writer = csv.writer(csvFile)                
                    writer.writerow([loss1,loss2, count])
                print ("loss1",loss1,"loss2:",loss2,"epoc:",count)
                count+=1
        Data_c = [0]
        Data_c_img = [0]
        ori=[0]
        for batch in train_iter:

                x = batch.text
                x, lengths = x
                y = batch.label
                
                if len(Data_label)==1:                    
                    Data_label=y
                else:    
                    Data_label=np.append(Data_label,y,axis=0)
                embeddings = nn.Embedding(TEXT.vocab.vectors.size()[0], TEXT.vocab.vectors.size()[1])
                embeddings.weight.data.copy_(TEXT.vocab.vectors)
                embeddings =embeddings(x)
                embedded = embeddings.transpose(0, 1)
                embedded = embedded.detach().numpy()          
                emb_out = self.embedding.predict(embedded)
                rec_out = self.recognition.predict(emb_out)

                if len(Data_c)==1:                    
                    Data_c=emb_out
                    Data_c_img=rec_out
                    ori=embedded

                else:    
                    Data_c=np.append(Data_c,emb_out,axis=0)
                    Data_c_img=np.append(Data_c_img,rec_out,axis=0)
                    ori=np.append(ori,embedded,axis=0)
                x=x.T.numpy()
                y=y.numpy()            
                for go in range(len(x)):
                    textname="txt/"+str(text_num)+".txt"
                    
                    with open(textname, 'a') as f:   
                        e=x[go]
                        print("label",file=f)
                        print(y[go],file=f)
                        print("txt",file=f)
                        for g in e:
                            print(ems[g],end='',file=f)
                            print(" ",end='',file=f)
                        print("",file=f)
                        print("txt_num",file=f)   
                        for g in e:
                            print(g,file=f)
                    text_num+=1
                    
                    
                    
        for batch in test_iter:

                x = batch.text
                x, lengths = x
                y = batch.label
                Data_label=np.append(Data_label,y,axis=0)
                embeddings = nn.Embedding(TEXT.vocab.vectors.size()[0], TEXT.vocab.vectors.size()[1])
                embeddings.weight.data.copy_(TEXT.vocab.vectors)
                embeddings =embeddings(x)
                embedded = embeddings.transpose(0, 1)
                embedded = embedded.detach().numpy()
                emb_out = self.embedding.predict(embedded)
                rec_out = self.recognition.predict(emb_out)
                
                if len(Data_c)==1:                    
                    Data_c=emb_out
                    Data_c_img=rec_out
                    ori=embedded
                else:    
                    Data_c=np.append(Data_c,emb_out,axis=0)
                    Data_c_img=np.append(Data_c_img,rec_out,axis=0)
                    ori=np.append(ori,embedded,axis=0)
                    
                x=x.T.numpy()
                y=y.numpy()
                
                for go in range(len(x)):
                    textname="txt/"+str(text_num)+".txt"
                    
                    with open(textname, 'a') as f:   
                        e=x[go]
                        print("label",file=f)
                        print(y[go],file=f)
                        print("txt",file=f)
                        for g in e:
                            print(ems[g],end='',file=f)
                            print(" ",end='',file=f)
                        print("",file=f)
                        print("txt_num",file=f)   
                        for g in e:
                            print(g,file=f)
                    text_num+=1
                    
                    
                    
        return Data_c,Data_label



if __name__ == '__main__':
    tStart = time.time()
    trln = TRLN()
    c,label=trln.train(epochs=15, batch_size=20, sample_interval=50)
    
    #save data
    c2=scale(c)
    trln.save_model()    
    c_scale = open('20news_embedding.pkl', "wb")	
    pickle.dump(c2, c_scale)
    c_scale.close()    
    label_file = open('20news_label.pkl', "wb")	
    pickle.dump(label, label_file)
    label_file.close()    
    import scipy.io as sio 
    sio.savemat('20newsdata.mat', {'news': c2}) 
    sio.savemat('20newslabel.mat', {'label': label}) 
    tEnd = time.time()
    print (tEnd - tStart)
