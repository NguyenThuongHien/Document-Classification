import os
from pyvi import ViTokenizer
from gensim.models import Word2Vec
import pandas as pd

#################################################

def get_tokenizer(link):
    '''
        read the text and tokenizer its
    '''
    with open(link,'r',encoding='utf-8') as f:
        sentense = f.read()
        sentense = ViTokenizer.tokenize(sentense)
    
    return sentense.strip().split()

def get_file_name_and_label(root_data):
    '''
        return file path and it's label
    '''
    dataset = []
    for index, folder in enumerate(os.listdir(root_data)):
        for files in os.listdir(os.path.join(root_data,folder)):
            link = os.path.join(root_data,folder,files)
            sentense = get_tokenizer(link)
            label = index
            data = [sentense,label]
            dataset.append(data)
            
    return dataset

def training_word2vec(root_path):
    dataset = get_file_name_and_label(root_path)
    sentenses = []
    for i in range(len(dataset)):
        sentenses.append(dataset[i][0])
    
    model = Word2Vec(sentenses,min_count=1)
    vocab = list(model.wv.vocab)
    
    model.save('pretrain_data.bin')
    return vocab

def get_word_embedding(pretrained_data, word):
    model = Word2Vec.load(pretrained_data)
    return model.wv[word]

def save_directory_embedding(pretrained, vocab):
    model = Word2Vec.load(pretrained)
    dir = {}
    for word in vocab:
        dir[word] = model.wv[word]
    
    dataframe = pd.DataFrame(dir)
    
    dataframe.to_csv('dataset.csv',encoding='utf-8')


def main():
    
   path = 'data' # thuc muc goc chua dataset
   vocab = training_word2vec(path)
   save_directory_embedding(pretrained='pretrain_data.bin',vocab = vocab)

    
main()