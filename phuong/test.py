import torch
from torch import nn, optim
from torch.autograd import Variable

# sequence_lenght = 3140
# input_size = 100
# batch_size = 1
# number layers =  20


'''
lstm = nn.LSTM(5,3,3)
linear = nn.Linear(3,2)
inputs = torch.rand(1,3,5)

out, _ = lstm(inputs)

last_out = out[-1][-1]
print(last_out.size())
mul = linear(last_out)
print(mul)


inputs = torch.rand(1,2)
print(inputs)
x =  torch.Tensor([1]).type(torch.LongTensor)
loss = nn.CrossEntropyLoss()
l = loss(inputs,x)
print(l)
'''

a = [1,1,3]

a = torch.Tensor(a).type(torch.LongTensor)
print(a.type())

