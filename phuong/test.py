from model import Model
from gensim.models import Word2Vec
from pyvi import ViTokenizer
import torch
##----------------------------------------------##
pretrained = 'pretrain_data.bin'
path_max_lenght = 'max_lenght_of_sentenses.txt'
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


def convert_out_to_class(out):
    '''
        out is a pytorch tensor
    '''
    out = out.detach()[0].tolist()
    out_max = max(out)
    return out.index(out_max)

if __name__ == '__main__':
    
    input_size = 100
    hidden_size = 40
   
    num_layer = 30
    num_class = 5
    filepath = 'pretrained'
    path_sentense = 'test.txt'
    max_len = get_max_lenght_sentenses()
    sentense = get_tokenizer(path_sentense)

    embedding_sentense = get_word_embedding(sentense = sentense,max_len = max_len)

    model = Model(input_size = input_size, hidden_size = hidden_size, num_layer = num_layer,num_class=num_class)

    model.load_state_dict(torch.load(filepath))

    model.eval()

    output = model.forward(embedding_sentense)

    the_class = convert_out_to_class(output)
    print(the_class)

    pass