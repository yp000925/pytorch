# 线性回归举例

import numpy as np
import pandas as pd
import torch
from torch import nn
from matplotlib import pyplot as plt
import datetime

n = 400

X = torch.rand([n,2])*10 - 5.0
w0 = torch.tensor([[2.0],[3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0 +b0+torch.normal(0.0, 2.0, size=[n,1])


def data_iter(features,labels,batch_size):
    num_example = len(features)
    indices = list(range(num_example))
    np.random.shuffle(indices)
    for i in range(0,num_example,batch_size):
        index = torch.LongTensor(indices[i:min(i+batch_size,num_example)])
        yield features.index_select(0,index),labels.index_select(0,index)


class LineraRegression:
    def __init__(self):
        self.w = torch.randn_like(w0,requires_grad=True)
        self.b = torch.zeros_like(b0,requires_grad=True)
    def forward(self,x):
        return x@self.w+self.b

    def loss_func(self,y_pred,y_true):
        return torch.mean((y_pred-y_true)**2/2)

model = LineraRegression()


def train_step(model,features,labels):
    predictions = model.forward(features)
    loss = model.loss_func(predictions,labels)

    loss.backward()

    with torch.no_grad():
        model.w -= 0.001*model.w.grad
        model.b -= 0.001*model.b.grad

        model.w.grad.zero_()
        model.b.grad.zero_()
    return loss

def train_model(model,epoches):

    for epoch in range(1,epoches+1):

        for features,labels in data_iter(X,Y,10):
            loss = train_step(model,features,labels)

        if epoch % 200 == 0:
            print("-----------"*8,datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'))
            print("epoch=",epoch,"loss = ",loss.item())
            print("model.w = ",model.w.data)
            print("model.b = ",model.b.data)


train_model(model,2000)

#可视化结果

plt.figure(figsize = (12,5))
ax1 = plt.subplot(121)
ax1.scatter(X[:,0].numpy(),Y[:,0].numpy(), c = "b",label = "samples")
ax1.plot(X[:,0].numpy(),(model.w[0].data*X[:,0]+model.b[0].data).numpy(),"-r",linewidth = 5.0,label = "model")
ax1.legend()
plt.xlabel("x1")
plt.ylabel("y",rotation = 0)


ax2 = plt.subplot(122)
ax2.scatter(X[:,1].numpy(),Y[:,0].numpy(), c = "g",label = "samples")
ax2.plot(X[:,1].numpy(),(model.w[1].data*X[:,1]+model.b[0].data).numpy(),"-r",linewidth = 5.0,label = "model")
ax2.legend()
plt.xlabel("x2")
plt.ylabel("y",rotation = 0)

plt.show()