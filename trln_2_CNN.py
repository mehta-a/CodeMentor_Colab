import torchtext
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torchtext.vocab import Vectors
from tqdm import tqdm
import os
from sys import version_info
if version_info.major == 2:
    import urllib as urldownload
else:
    import urllib.request as urldownload

import tarfile
import re
import random
from torchtext import data
import keras
import keras.backend as K
class CNN_TRLN(nn.Module):
    def __init__(self, in_channels, out_channels, batch_size,trln):
        super(CNN_TRLN, self).__init__()
        self.nonstatic_embeddings = nn.Embedding(TEXT.vocab.vectors.size()[0], TEXT.vocab.vectors.size()[1])        
        self.nonstatic_embeddings.weight.data.copy_(TEXT.vocab.vectors)
        self.embeddings = self.nonstatic_embeddings 
        self.trln = trln
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = in_channels, out_channels = in_channels, kernel_size = n) for n in (1,2,3,4)])
        self.dropout_train, self.dropout_test = nn.Dropout(p = 0.5), nn.Dropout(p = 0)
        self.linear = nn.Linear(in_features=801, out_features=out_channels, bias = True)
    
    def forward(self, x, train = True):
        x, lengths = x
        
        lengths = Variable(lengths.view(-1, 1).float())
        embedded = self.embeddings(x)        
        trln_embed =self.trln.predict(embedded.transpose(0, 1).detach().numpy())     
        
        trln_embed =torch.FloatTensor(trln_embed)
        trln_embed = torch.cat([trln_embed, lengths], dim = 1)
        
        embedded = embedded.transpose(1, 2)
        embedded = embedded.transpose(0, 2)
        
        concatted_features = torch.cat([conv(embedded) for conv in self.convs if embedded.size(2) >= conv.kernel_size[0]], dim = 2)
        activated_features = torch.nn.functional.relu(concatted_features)
        pooled = torch.nn.functional.max_pool1d(activated_features, activated_features.size(2)).squeeze(2)

        ensemble = torch.cat([pooled, trln_embed], dim = 1)

        dropped = self.dropout_train(ensemble) if train else self.dropout_test(ensemble)
        output = self.linear(ensemble)
 
        logits = torch.nn.functional.log_softmax(output, dim = 1)
        return logits

    def predict(self, x):
        logits = self.forward(x, train = False)
        return logits.max(1)[1] + 1
    
    def train(self, train_iter, val_iter, num_epochs, learning_rate = 1e-3, plot = False):
        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        loss_vec = []
        
        for epoch in tqdm(range(1, num_epochs+1)):
            epoch_loss = 0
            for batch in train_iter:
                x = batch.text
                y = batch.label
                optimizer.zero_grad()
                y_p = self.forward(x)                
                loss = criterion(y_p, y-1)
                loss.backward()                
                optimizer.step()
                epoch_loss += loss.item()
                
            self.model = model
            
            loss_vec.append(epoch_loss / len(train_iter))
            if epoch % 1 == 0:
                acc = self.validate(val_iter)
                print('Epoch {} loss: {} | acc: {}'.format(epoch, loss_vec[epoch-1], acc))
                self.model = model
        if plot:
            plt.plot(range(len(loss_vec)), loss_vec)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()
        print('\nModel trained.\n')
        self.loss_vec = loss_vec
        self.model = model

    def test(self, test_iter):
        upload, trues = [], []
        for batch in test_iter:
            x, y = batch.text, batch.label
            probs = self.predict(x)
            upload += list(probs.numpy())
            trues += list(y.numpy())
            
        correct = sum([1 if i == j else 0 for i, j in zip(upload, trues)])
        accuracy = correct / len(trues)
        print('Testset Accuracy:', accuracy)
        #save result
        np.save('pred_trln.npy',upload) 
        np.save('y_trln.npy',trues) 
        with open("predictions.txt", "w") as f:
            for u in upload:
                f.write(str(u) + "\n")    
    def validate(self, val_iter):
        y_p, y_t, correct = [], [], 0
        for batch in val_iter:
            x, y = batch.text, batch.label
            probs = self.model.predict(x)[:len(y)]
            y_p += list(probs.data)
            y_t += list(y.data)
        
        correct = sum([1 if i == j else 0 for i, j in zip(y_p, y_t)])
        accuracy = correct / len(y_p)
        return accuracy
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


def mutual_info_loss(c, c_given_x):
        eps = 1e-8
        conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
        entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))
        return conditional_entropy + entropy
def mutual_info_loss_n(c, c_given_x):
        eps = 1e-8
        conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
        entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))
        return -1*(conditional_entropy + entropy)

if __name__ == '__main__':
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
    url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{}.vec'    
    TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

    
    trln = keras.models.load_model('embedding.h5',custom_objects={'mutual_info_loss': mutual_info_loss,'mutual_info_loss_n': mutual_info_loss_n})
    model = CNN_TRLN(in_channels = 300, out_channels = 20, batch_size = 20, trln=trln)
    model.train(train_iter = train_iter, val_iter = test_iter, num_epochs = 46, learning_rate = 1e-4, plot = False)
    model.test(test_iter)


