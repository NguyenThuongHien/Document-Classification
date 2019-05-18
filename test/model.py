import torch
from torch import nn, optim

class Model(nn.Module):
    def __init__(self,input_size, hidden_size, numLayers):
        super(Model,self).__init__()
        self.cell = nn.GRU(input_size = input_size, hidden_size = hidden_size,numLayers)
        

    def forward(self, *input):
        return super().forward(*input)
    def init_hidden(self):
        pass