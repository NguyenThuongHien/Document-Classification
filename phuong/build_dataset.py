import numpy as np
from gensim.models import Word2Vec
import os
from pyvi import ViTokenizer
import torch

##--------------------------------------------##
numclass = 10
root_path = 'train_data'
path_max_lenght = 'max_lenght_of_sentenses.txt'
pretrained = 'pretrain_data.bin'

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

def get_max_lenght_sentenses(path = path_max_lenght):
    '''
        the max lenght of sentenses is stored in file
    '''
    with open(path,'r') as f:
        return int(f.read())

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


def get_word_embedding(sentense,max_len):
    temp = [i*0 for i in range(100)]
    model = Word2Vec.load(pretrained)
    X_data = []
    lenSent = len(sentense)
    if lenSent > max_len:
        for i in range(max_len):
            X_data.append(model.wv[sentense[i]])
    else:
        for i in range(max_len):
            if i < lenSent:
                X_data.append(model.wv[sentense[i]])
            else:
                X_data.append(temp)
    return X_data

def get_embedding_and_label(root_path = root_path):
    print('Building train data...')
    max_len = get_max_lenght_sentenses()
    X_data = []
    Y_data = []
    for index, folder in enumerate(os.listdir(root_path)):
        for file in os.listdir(os.path.join(root_path,folder)):
            link = os.path.join(root_path,folder,file)
            sentense = get_tokenizer(link)
            sentense_embedding = get_word_embedding(sentense,max_len = max_len)
            x = torch.Tensor([sentense_embedding])
            y = torch.Tensor([index]).type(torch.LongTensor)
            
            X_data.append(x)
            Y_data.append(y)

            

    return X_data, Y_data


if __name__ == '__main__':
    
    X_train, Y_train  = get_embedding_and_label()
    