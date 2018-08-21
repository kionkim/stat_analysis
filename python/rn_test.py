# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import nltk
import collections
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, auc

#os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'

word_freq = collections.Counter()
max_len = 0
num_rec = 0

with open('../data/umich-sentiment-train.txt', 'rb') as f:
    for line in f:
        label, sentence = line.decode('utf8').strip().split('\t')
        words = nltk.word_tokenize(sentence.lower())
        if len(words) > max_len:
            max_len = len(words)
        for word in words:
            word_freq[word] += 1
        num_rec += 1
		
MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40
# most_common output -> list
word2idx = {x[0]: i+2 for i, x in enumerate(word_freq.most_common(MAX_FEATURES - 2))}
word2idx ['PAD'] = 0
word2idx['UNK'] = 1


idx2word= {i:v for v, i in word2idx.items()}
vocab_size = len(word2idx)

y = []
x = []
origin_txt = []
with open('../data/umich-sentiment-train.txt', 'rb') as f:
    for line in f:
        _label, _sentence = line.decode('utf8').strip().split('\t')
        origin_txt.append(_sentence)
        y.append(int(_label))
        words = nltk.word_tokenize(_sentence.lower())
        _seq = []
        for word in words:
            if word in word2idx.keys():
                _seq.append(word2idx[word])
            else:
                _seq.append(word2idx['UNK'])
        if len(_seq) < MAX_SENTENCE_LENGTH:
            _seq.extend([0] * ((MAX_SENTENCE_LENGTH) - len(_seq)))
        else:
            _seq = _seq[:MAX_SENTENCE_LENGTH]
        x.append(_seq)
		
		
def one_hot(x, vocab_size):
    res = np.zeros(shape = (vocab_size))
    res[x] = 1
    return res
	
x_1 = np.array([np.array([one_hot(word, MAX_FEATURES) for word in example]) for example in x])

tr_idx = np.random.choice(range(x_1.shape[0]), int(x_1.shape[0] * .8))
va_idx = [x for x in range(x_1.shape[0]) if x not in tr_idx]

tr_x = x_1[tr_idx, :]
tr_y = [y[i] for i in tr_idx]
va_x = x_1[va_idx, :]
va_y = [y[i] for i in va_idx]

import mxnet as mx
batch_size = 32 
train_data = mx.io.NDArrayIter(data=[tr_x, tr_y], batch_size=batch_size, shuffle = False)
valid_data = mx.io.NDArrayIter(data=[va_x, va_y], batch_size=batch_size, shuffle = False)

from mxnet import gluon, autograd, nd
from mxnet.gluon import nn
context = mx.gpu()

class RN_Classifier(nn.HybridBlock):
    def __init__(self, SENTENCE_LENGTH, VOCABULARY, **kwargs):
        super(RN_Classifier, self).__init__(**kwargs)
        self.SENTENCE_LENGTH = SENTENCE_LENGTH
        self.VOCABULARY = VOCABULARY
        with self.name_scope():
            self.g_fc1 = nn.Dense(256, activation='relu')
            self.g_fc2 = nn.Dense(256, activation='relu')
            self.g_fc3 = nn.Dense(256, activation='relu')
            self.g_fc4 = nn.Dense(256, activation='relu')


            self.fc1 = nn.Dense(128, activation = 'relu') # 256 * 128
            self.fc2 = nn.Dense(2) # 128 * 2
            # 1253632 param : approx 20MB
    def hybrid_forward(self, F, x):
        # (x_i, x_j)의 pair를 만들기
        # 64 배치를 가정하면
        #print('x shape = {}'.format(x.shape))
        x_i = x.expand_dims(1) # 64 * 1* 40 * 2000* : 0.02GB
        #print('x_i shape = {}'.format(x_i.shape))
        x_i = F.repeat(x_i,repeats= self.SENTENCE_LENGTH, axis=1) # 64 * 40 * 40 * 2000: 1.52GB
        #print('x_i shape = {}'.format(x_i.shape))
        x_j = x.expand_dims(2) # 64 * 40 * 1 * 2000
        #print('x_j shape = {}'.format(x_j.shape))
        x_j = F.repeat(x_j,repeats= self.SENTENCE_LENGTH, axis=2) # 64 * 40 * 40 * 2000: 1.52GB
        #print('x_j shape = {}'.format(x_j.shape))
        x_full = F.concat(x_i,x_j,dim=3) # 64 * 40 * 40 * 4000: 3.04GB
        #print('x_full shape = {}'.format(x_full.shape))
        
        #print('x_i type = {}'.format(np.finfo(x_i[0][0][0][0]).dtype))
        
        # batch*sentence_length*sentence_length개의 batch를 가진 2*VOCABULARY input을 network에 feed
        _x = x_full.reshape((-1, 2 * self.VOCABULARY))
        #print('_x type = {}'.format(np.finfo(_x[0][0]).dtype))
        _x = self.g_fc1(_x) # (64 * 40 * 40) * 256: .1GB 추가메모리는 안먹나?
        
        _x = self.g_fc2(_x) # (64 * 40 * 40) * 256: .1GB (reuse)
        _x = self.g_fc3(_x) # (64 * 40 * 40) * 256: .1GB (reuse)
        _x = self.g_fc4(_x) # (64 * 40 * 40) * 256: .1GB (reuse)

        # sentence_length*sentence_length개의 결과값을 모두 합해서 sentence representation으로 나타냄
        x_g = _x.reshape((-1, self.SENTENCE_LENGTH * self.SENTENCE_LENGTH,256)) # (64, 40*40, 256) : .1GB
        sentence_rep = x_g.sum(1) # (64, 256): ignorable
        
        # 여기서부터는 classifier
        clf = self.fc1(sentence_rep)
        clf = self.fc2(clf)
        return clf
		
		
		
rn = RN_Classifier(MAX_SENTENCE_LENGTH, MAX_FEATURES)
rn.collect_params().initialize(mx.init.Xavier(), ctx = context)

# Fake data test :  Max 3.2GB OK
#z = np.random.uniform(size = (32, 40, 2000))
#z1 = nd.array(z, ctx =context) # , dtype = np.float16)
#res = rn(z1)
#print(res.shape)

loss = gluon.loss.SoftmaxCELoss()
trainer = gluon.Trainer(rn.collect_params(), 'adam', {'learning_rate': 1e-3})

n_epoch = 10
from tqdm import tqdm

for epoch in tqdm(range(n_epoch), desc = 'epoch'):
    ## Training
    train_data.reset()
    n_obs = 0
    _total_los = 0
    pred = []
    label = []
    for i, batch in enumerate(train_data):
        _dat = batch.data[0].as_in_context(context)
        _label = batch.data[1].as_in_context(context)
        with autograd.record():
            _out = rn(_dat)
            _los = nd.sum(loss(_out, _label)) # 배치의 크기만큼의 loss가 나옴
            _los.backward()
        trainer.step(_dat.shape[0])
        n_obs += _dat.shape[0]
        _total_los += nd.sum(_los).asnumpy()
        # Epoch loss를 구하기 위해서 결과물을 계속 쌓음
        pred.extend(nd.softmax(_out)[:,1].asnumpy()) # 두번째 컬럼의 확률이 예측 확률
        label.extend(_label.asnumpy())
    tr_acc = accuracy_score(label, [round(p) for p in pred])
    tr_loss = _total_los/n_obs
    
    ### Evaluate training
    valid_data.reset()
    n_obs = 0
    _total_los = 0
    pred = []
    label = []
    for i, batch in enumerate(valid_data):
        _dat = batch.data[0].as_in_context(context)
        _label = batch.data[1].as_in_context(context)
        _out = rn(_dat)
        _pred_score = nd.softmax(_out)
        n_obs += _dat.shape[0]
        _total_los += nd.sum(loss(_out, _label)).asnumpy()
        pred.extend(nd.softmax(_out)[:,1].asnumpy())
        label.extend(_label.asnumpy())
    va_acc = accuracy_score(label, [round(p) for p in pred])
    va_loss = _total_los/n_obs
    tqdm.write('Epoch {}: tr_loss = {}, tr_acc= {}, va_loss = {}, va_acc= {}'.format(epoch, tr_loss, tr_acc, va_loss, va_acc))
