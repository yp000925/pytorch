# from __future__ import unicode_literals, print_function, division
"""
对名字分类，找出名字来自哪个地区
A character-level RNN reads words as a series of characters -
outputting a prediction and "hidden state" at each step, feeding its
previous hidden state into each next step. We take the final prediction
to be the output, i.e. which class the word belongs to.

Specifically, we'll train on a few thousand surnames from 18 languages
of origin, and predict which language a name is from based on the
spelling

"""



import unicodedata
import string
from io import open
import glob
import os
import torch


"""

1. read data and store them into the dictionary data type 
2. transform the name into one-hot vector and then tensor [name_length x 1 x 57] dim=1 表示batch_size 是1  
"""
def findFiles(path): return glob.glob(path)

print(findFiles('data/names/*.txt'))

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

# transfer the name into one-hot vector
def letterToIndex(letter):
    return all_letters.find(letter)

# def letterToTensor(letter):
#     tensor = torch.zeros(1,n_letters)
#     tensor[0][letterToIndex(letter)] = 1
#     return tensor

# transfer a name into tensor [name_length x 1 x 57] one-hot

def nameToTensor(name):
    tensor = torch.zeros(len(name),1,n_letters)
    for i, letter in enumerate(name):
        tensor[i][0][letterToIndex(letter)] = 1
    return tensor


'''
generate training example
'''

import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    name = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    name_tensor = nameToTensor(name)
    return category, name, category_tensor, name_tensor

# for i in range(10):
#     category, line, category_tensor, line_tensor = randomTrainingExample()
#     print('category =', category, '/ line =', line)


"""
define the RNN network 
using the linear layer 
"""
from torch import nn
class RNN_net(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size+hidden_size,hidden_size)
        self.h2o = nn.Linear(input_size+hidden_size,output_size)
        self.softmax = nn.Softmax()

    def forward(self,input, hidden):
        combined = torch.cat((input,hidden),1) # concat along dim=1
        hidden =self.i2h(combined)
        output = self.h2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1,self.hidden_size)


hidden_size = 128
net = RNN_net(n_letters,hidden_size,n_categories)


#define a function to transfer the output into one category which has the greatest value using torch.topk

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def train_step(model,name,category,learning_rate=0.005):
    hidden = model.initHidden()  # init hidden states
    model.zero_grad()

    for i in range(name.size()[0]):
        output, hidden = model(name[i],hidden)

    loss = nn.NLLLoss()(output,category)

    loss.backward()

    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

def evaluate(model,name_tensor):
    hidden = model.initHidden()

    for i in range(name_tensor.size()[0]):
        output, hidden = model(name_tensor[i], hidden)

    return output


def predict(model, name, n_predictions=3):
    print('\n> %s' % name)
    with torch.no_grad():
        output = evaluate(model,nameToTensor(name))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

'''
train the network 
'''
import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000



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
    category, name, category_tensor, name_tensor = randomTrainingExample()
    output, loss = train_step(net,name_tensor,category_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, name, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

print("finish traring , make prediction: ")
predict(net,'Dovesky')
predict(net,'Jackson')
predict(net,'Satoshi')