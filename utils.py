import pandas as pd
import torch
import numpy as np
import re
import nltk

def loadQuora(data_path):
    data_quora = pd.read_csv(data_path+'quora_duplicate_questions.tsv', sep='\t')
    data_quora = data_quora[['question1', 'question2', 'is_duplicate']]
    data_quora.rename(columns={'is_duplicate': 'paraphrase', 'question1': 'utt1', 'question2': 'utt2'}, inplace=True)
    data_quora.dropna(inplace=True)

    test = data_quora[-10000:]
    val = data_quora[:-10000][-10000:]
    train = data_quora[:-20000]

    train = train.reset_index()
    test = test.reset_index()
    val = val.reset_index()
    
    return train, test, val

def loadMsr(data_path):
    with open(data_path+'msr-paraphrase-corpus/msr_paraphrase_train.txt', 'r') as file:
        lines = file.readlines() 
    data_msr = pd.DataFrame(data={'utt1': [line.split('\t')[3] for line in lines[1:]],
                                  'utt2': [line.split('\t')[4] for line in lines[1:]],
                                  'paraphrase': [line.split('\t')[0] for line in lines[1:]]})
    with open(data_path+'msr-paraphrase-corpus/msr_paraphrase_test.txt', 'r') as file:
        lines = file.readlines() 
    data_msr_test = pd.DataFrame(data={'utt1': [line.split('\t')[3] for line in lines[1:]],
                                       'utt2': [line.split('\t')[4] for line in lines[1:]],
                                       'paraphrase': [line.split('\t')[0] for line in lines[1:]]})
  
    split = int(0.8*len(data_msr))
    train = data_msr[:split]
    val = data_msr[split:]
    val = val.reset_index()
    test = data_msr_test

    return train, test, val

def loadTwitter(data_path):
    data_twitter = pd.read_csv(data_path+'twitter/Twitter_URL_Corpus_train.txt', sep='\t', header=None)
    data_twitter.columns = ['utt1', 'utt2', 'paraphrase', 'link']
    data_twitter = data_twitter[['utt1', 'utt2', 'paraphrase']]
    data_twitter['paraphrase'] = data_twitter['paraphrase'].apply(lambda x: int(re.search('(.),', x).group(1)))
    # original paper removes entries where annotators are split
    data_twitter = data_twitter[data_twitter['paraphrase']!=3] 
    data_twitter['paraphrase'] = data_twitter['paraphrase'].apply(lambda x: '1' if x>=4 else '0')

    data_twitter_test = pd.read_csv(data_path+'twitter/Twitter_URL_Corpus_test.txt', sep='\t', header=None)
    data_twitter_test.columns = ['utt1', 'utt2', 'paraphrase', 'link']
    data_twitter_test = data_twitter_test[['utt1', 'utt2', 'paraphrase']]
    data_twitter_test['paraphrase'] = data_twitter_test['paraphrase'].apply(lambda x: int(re.search('(.),', x).group(1)))
    # original paper removes entries where annotators are split
    data_twitter_test = data_twitter_test[data_twitter_test['paraphrase']!=3] 
    data_twitter_test['paraphrase'] = data_twitter_test['paraphrase'].apply(lambda x: '1' if x>=4 else '0')

    split = int(0.8*len(data_twitter))
    batch_size = 16
    train = data_twitter[:split]
    val = data_twitter[split:]
    val = val.reset_index()
    train = train.reset_index()
    test = data_twitter_test
        
    return train, test, val

def loadPaws(data_path):
    train = pd.read_csv(data_path+'PAWS/final/train.tsv', sep='\t', skiprows=1, names=['id', 'utt1', 'utt2', 'paraphrase'])
    test = pd.read_csv(data_path+'PAWS/final/test.tsv', sep='\t', skiprows=1, names=['id', 'utt1', 'utt2', 'paraphrase'])
    val = pd.read_csv(data_path+'PAWS/final/dev.tsv', sep='\t', skiprows=1, names=['id', 'utt1', 'utt2', 'paraphrase'])
    
    return train, test, val

def loadPawsQqp(data_path):
    train = pd.read_csv(data_path+'PAWS/paws_qqp/train.tsv', sep='\t', skiprows=1, names=['id', 'utt1', 'utt2', 'paraphrase'])
    dev_test = pd.read_csv(data_path+'PAWS/paws_qqp/dev_and_test.tsv', sep='\t', skiprows=1, names=['id', 'utt1', 'utt2', 'paraphrase'])
    val = dev_test[:int(len(dev_test)/2)]
    test = dev_test[int(len(dev_test)/2):]
    val = val.reset_index()
    test = test.reset_index()
    
    return train, test, val
        
def loadData(data_path, src):
    if src == 'quora':
        train, test, val = loadQuora(data_path)
    elif src == 'msr':
        train, test, val = loadMsr(data_path)
    elif src == 'twitter':
        train, test, val = loadTwitter(data_path)
    elif src == 'paws':
        train, test, val = loadPaws(data_path)
    elif src == 'paws_qqp':
        train, test, val = loadPawsQqp(data_path)
    else:
        raise Exception("Invalid data source!",src)
    
    return train[:12000], test[:2000], val[:2000]

def getPos(text):
    text = nltk.word_tokenize(text)
    tags = nltk.pos_tag(text)
    return ' '.join([t[1] for t in tags])

def loadAllData(data_path, pos=False):
    tasks=('msr', 'quora', 'twitter', 'paws', 'paws_qqp') 
    taskLabel = {'msr':0, 'quora':1, 'twitter':2, 'paws':3, 'paws_qqp':4}
    train_l = []
    test_l = []
    val_l = []
    for task in tasks:
        train, test, val = loadData(data_path, task)
        train['label'] = taskLabel[task]
        test['label'] = taskLabel[task]
        val['label'] = taskLabel[task]
        train_l.append(train)
        test_l.append(test)
        val_l.append(val)
    train = pd.concat(train_l)
    test = pd.concat(test_l)
    val = pd.concat(val_l)
        
    train = train.reset_index()
    test = test.reset_index()
    val = val.reset_index()
    
    if pos:
        train['utt1'] = train['utt1'].apply(lambda x: getPos(x))
        train['utt2'] = train['utt2'].apply(lambda x: getPos(x))
        test['utt1'] = test['utt1'].apply(lambda x: getPos(x))
        test['utt2'] = test['utt2'].apply(lambda x: getPos(x))
        val['utt1'] = val['utt1'].apply(lambda x: getPos(x))
        val['utt2'] = val['utt2'].apply(lambda x: getPos(x))

    return train, test, val