import torch
from torch import nn, optim


class Model(nn.Module):
    def __init__(self,input_size,hidden_size,num_layer,num_class):
        super(Model,self).__init__()
        self.num_class = num_class
        self.cell = nn.LSTM(input_size,hidden_size,num_layer)
        self.linear = nn.Linear(hidden_size,num_class)
    
    def forward(self, input):
        # input size: batch_size = 1, sequence lenght, input_size (1,n,100)
        batch_size = input.size()[0]

        out,_ = self.cell(input)
        last_out = out[-1][-1]

        fc = self.linear(last_out)

        return fc.view(batch_size,self.num_class)

if __name__ == '__main__':
    model = Model(100,20,2,2)
    
