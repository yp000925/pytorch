import torch
import unicodedata
import string
from io import open
import glob
import os
from torch import nn

## 数据准备

def findfiles(path):
    return glob.glob(path)

all_letters = string.ascii_letters + ",.;'"
n_letters = len(all_letters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

all_country = []
dis_by_country = {}

def readlines(filename):
    names = open(filename,encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(name) for name in names]

for filename in findfiles('data/names/*.txt'):
    country_name = os.path.splitext(os.path.basename(filename))[0]
    all_country.append(country_name)
    names = readlines(filename)
    dis_by_country[country_name]=names

n_country = len(all_country)

def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1,n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def nameToTensor(name):
    tensor = torch.zeros(len(name), 1, n_letters)
    for i,letter in enumerate(name):
        tensor[i][0][letterToIndex(letter)] = 1
    return tensor

# print(nameToTensor('Jone').size())
import random
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def dataloader(dataset, batch_size=100):
    data = []
    for i in range(batch_size):
        country = randomChoice(list(dataset.keys()))
        name = randomChoice(dataset[country])
        country_tensor = torch.tensor([all_country.index(country)],dtype=torch.long)
        name_tensor = nameToTensor(name)
        data.append((country, name, country_tensor, name_tensor))
    return data

# def dataloader(dataset):
#     country = randomChoice(all_country)
#     name = randomChoice(dataset[country])
#     country_tensor = torch.tensor([all_country.index(country)],dtype=torch.long)
#     name_tensor = nameToTensor(name)
#     return (country, name, country_tensor, name_tensor)

class LSTM_net(nn.Module):

    def __init__(self,inputsize,hiddensize,outputsize):
        super().__init__()
        self.hiddensize = hiddensize
        self.lstm = nn.LSTM(input_size=inputsize,hidden_size=hiddensize)
        self.h2o = nn.Linear(hiddensize,outputsize)
        self.softmax = nn.Softmax(dim=2)

    def forward(self,input,hidden):
        out,hidden = self.lstm(input.view(1,1,-1),hidden)
        output = self.h2o(hidden[0])
        output = self.softmax(output)
        return output.view(1,-1), hidden

    def init_hidden(self):
        #one represent the hidden state, the other represent the hidden cell state
        return (torch.zeros(1,1,self.hiddensize),torch.zeros(1,1,self.hiddensize))


# class LSTM_net(nn.Module):
#     def __init__(self,inputsize,hiddensize,outputsize):
#         super().__init__()
#         self.hidden_size = hiddensize
#         self.lstm = nn.LSTM(input_size=inputsize, hidden_size=hiddensize)
#         self.h2o = nn.Linear(hiddensize, outputsize)
#         self.softmax = nn.Softmax()
#
#     def forward(self,input):
#         h_t = torch.zeros(input.size()[0], self.hidden_size,dtype=torch.double)
#         c_t = torch.zeros(input.size()[0], self.hidden_size,dtype=torch.double)
#
#         for i,input_t in enumerate(input.chunk(input.size(1),dim=1)):
#             h_t,c_t = self.lstm(input_t,(h_t,c_t))
#             output = self.h2o(h_t)
#             output = self.softmax(output)
#             return output

model = LSTM_net(inputsize=n_letters,hiddensize=128,outputsize=n_country)

optimizer = torch.optim.SGD(model.parameters(),lr = 0.01, momentum=0.9)


def train_step(model, dataset, batch_size = 128):

    hidden = model.init_hidden()  # init hidden states
    model.zero_grad()
    data_ = dataloader(dataset, batch_size)
    total_loss = 0

    for  country, name, country_tensor, name_tensor in data_:

        for i in range(name_tensor.size()[0]):
            output, hidden = model(name_tensor[i],hidden)

        loss = nn.NLLLoss()(output, country_tensor)

        loss.backward(retain_graph=True)

        total_loss += loss

    optimizer.step()

    return output, total_loss/batch_size, country, name




def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_country[category_i], category_i


import time
import math

n_iters =10
print_every = 1
plot_every = 1

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):

    output, loss, category, name = train_step(model, dis_by_country, batch_size=128)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (
        iter, iter / n_iters * 100, timeSince(start), loss, name, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0


