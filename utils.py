import pandas as pd
import torch
import numpy as np
import re


def loadData(data_path, src):
    if src == 'quora':
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

    elif src == 'msr':
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

    elif src == 'twitter':
        #twitter data
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
        
    else:
        raise Exception("Invalid data source!")
    
    return train, test, val