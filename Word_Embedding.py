import os
from pyvi import ViTokenizer
from gensim.models import Word2Vec

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

def training_word2vec(dataset):
    sentenses = []
    for i in range(len(dataset)):
        sentenses.append(dataset[i][0])
    
    model = Word2Vec(sentenses,min_count=1)
    model.save('pretrain_data.bin')



def main():
    root_path = 'dulieu_train'
    dataset = get_file_name_and_label(root_path)
    training_word2vec(dataset)

main()