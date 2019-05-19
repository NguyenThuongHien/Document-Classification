from model import Model
from build_dataset import get_embedding_and_label
import torch
from torch import nn, optim

##-------------------------------------##

def get_max_lenght_sentenses(path):
    '''
        the max lenght of sentenses is stored in file
    '''
    with open(path,'r') as f:
        return int(f.read())


if __name__ == '__main__':
    path_max_lenght = 'max_lenght_of_sentenses.txt'
    input_size = 100
    hidden_size = 20
    sequence_lenght = get_max_lenght_sentenses(path_max_lenght)
    batch_size = 1
    num_layer = 20
    num_class = 2
    epochs = 10
    

    model = Model(input_size = input_size, hidden_size = hidden_size, num_layer = num_layer,num_class=num_class)

    X_train, Y_train = get_embedding_and_label()
    
    loss = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters())
    print('training...')
    for e in range(epochs):
        l = 0
        for x, y in zip(X_train,Y_train):
            opt.zero_grad()
            out = model.forward(x)
            l += loss(out,y)
            l.backward(retain_graph=True)
            opt.step()
        print('epoch {} - loss {}'.format(e,l))
    
    