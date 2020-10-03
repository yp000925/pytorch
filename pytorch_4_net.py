import torch
from torch import nn

class Block(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x, x_input):
        x_out = torch.max((1+x)*x_input[:,-1,:],torch.tensor(0.0))
        return x_out

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        #3层lstm
        self.lstm = nn.LSTM(input_size=3,hidden_size=3, num_layers=5, batch_first=True)
        self.linear = nn.Linear(3,3)
        self.block = Block()

    def forward(self,x_input):
        x = self.lstm(x_input)[0][:,-1,:]
        x = self.linear(x)
        y = self.block(x,x_input)

        return y

