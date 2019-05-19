import os
from pyvi import ViTokenizer
from gensim.models import Word2Vec
import pandas as pd

##--------------------------------------------------------##
stop_words =[ 'bị',
'bởi',
'cả',
'các',
'cái',
'cần',
'càng',
'chỉ',
'chiếc',
'cho',
'chứ',
'chưa',
'chuyện',
'có',
'có_thể',
'cứ',
'của',
'cùng',
'cũng',
'đã',
'đang',
'đây',
'để',
'đến_nỗi',
'đều',
'điều',
'do',
'đó',
'được',
'dưới',
'gì',
'khi',
'không',
'là',
'lại',
'lên',
'lúc',
'mà',
'mỗi',
'một_cách',
'này',
'nên',
'nếu',
'ngay',
'nhiều',
'như',
'nhưng',
'những',
'nơi',
'nữa',
'phải',
'qua',
'ra'
'rằng',
'rằng',
'rất',
'rất',
'rồi',
'sau',
'sẽ',
'so',
'sự',
'tại',
'theo',
'thì',
'trên',
'trước',
'từ',
'từng',
'và',
'vẫn',
'vào',
'vậy',
'vì',
'việc',
'với',
'vừa',
'.',',',
'/','\\','%','*','$','~','`','!','#','^','&','(',')','_','+','-','=',';',':','"',"'",'{','}','[',']'
]

def delete_stop_words(sentense, stop_word = stop_words):
    for index,ele in enumerate(sentense):
        if ele in stop_word:
            del sentense[index]

def get_tokenizer(link):
    '''
        read the text then tokenizer
    '''
    with open(link,'r',encoding='utf-8') as f:
        sentense = f.read()
        sentense = sentense.lower()
        sentense = ViTokenizer.tokenize(sentense)
    temp = sentense.strip().split()
    delete_stop_words(temp)
    return temp

def get_file_name_and_label(root_data):
    '''
        return file path and it's label
    '''
    max_sentenses = -1
    dataset = []
    for folder in os.listdir(root_data):
        for files in os.listdir(os.path.join(root_data,folder)):
            link = os.path.join(root_data,folder,files)
            sentense = get_tokenizer(link)
            max_sentenses = max(max_sentenses,len(sentense))
            data = sentense
            dataset.append(data)
    
    with open('max_lenght_of_sentenses.txt','w+') as f:
        f.write(str(max_sentenses))
   
    return dataset

def training_word2vec(root_path):
    print('Embedding...')
    dataset = get_file_name_and_label(root_path)
   
    model = Word2Vec(dataset,min_count=1)
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
    print('Saving to file...')
    dataframe.to_csv('dataset.csv',encoding='utf-8')
    
   
   
if __name__ == '__main__':
    path = 'train_data'
    vocab = training_word2vec(path)
    save_directory_embedding(pretrained='pretrain_data.bin',vocab = vocab)