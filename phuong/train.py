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

def convert_out_to_class(out):
    '''s
        out is a pytorch tensor
    '''
    out = out.detach()[0].tolist()
    out_max = max(out)
    return out.index(out_max)

if __name__ == '__main__':
    path_max_lenght = 'max_lenght_of_sentenses.txt'
    input_size = 100
    hidden_size = 40
    sequence_lenght = get_max_lenght_sentenses(path_max_lenght)
    batch_size = 1
    num_layer = 30
    num_class = 5
    epochs = 30
    

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
    
    filepath = 'pretrained'

    torch.save(model.state_dict(),filepath)


    
    