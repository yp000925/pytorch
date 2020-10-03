# DNN二分类举例

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import datetime

#preprare the data
num_pos = 2000
num_neg = 2000

r_pos = 5.0+torch.normal(0.0,1.0,size = [num_pos,1])
theta = 2*np.pi*torch.rand([num_pos,1])
X_p = torch.cat([r_pos*torch.cos(theta),r_pos*torch.sin(theta)], dim = 1)
Y_p = torch.ones_like(r_pos)

r_neg = 8.0+torch.normal(0.0,1.0,size = [num_neg,1])
theta = 2*np.pi*torch.rand([num_neg,1])
X_n = torch.cat([r_neg*torch.cos(theta),r_neg*torch.sin(theta)], dim = 1)
Y_n = torch.zeros_like(r_neg)

X = torch.cat([X_p,X_n],dim = 0)
Y = torch.cat([Y_p,Y_n],dim = 0)
#
# plt.figure(figsize= (6,6))
# plt.scatter(X_p[:,0].numpy(),X_p[:,1].numpy(),c = 'r')
# plt.scatter(X_n[:,0].numpy(),X_n[:,1].numpy(),c = 'g')
# plt.legend(['positive','negative'])
# plt.show()


# get data iterator
def data_iter(features,labels,batch_size = 10):
    num_feature = len(features)
    indexes = list(range(num_feature))
    np.random.shuffle(indexes)
    for i in range(0,num_feature,batch_size):
        index = torch.LongTensor(indexes[i:min(i+batch_size,num_feature)])
        yield features.index_select(0,index),labels.index_select(0,index)

#define DNN model

class DNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(2,4))
        self.b1 = nn.Parameter(torch.zeros(1,4))
        self.w2 = nn.Parameter(torch.randn(4,8))
        self.b2 = nn.Parameter(torch.zeros(1,8))
        self.w3 = nn.Parameter(torch.randn(8,1))
        self.b3 = nn.Parameter(torch.zeros(1,1))

    def forward(self,x):
        x = torch.relu(x@self.w1 + self.b1)
        x = torch.relu(x@self.w2 + self.b2)
        y = torch.sigmoid(x@self.w3 +self.b3)
        return y

    def loss_func(self,y_pred,y_true):
        eps = 1e-7
        y_pred = torch.clamp(y_pred,eps,1.0-eps)
        bce = -y_true*torch.log(y_pred)-(1-y_true)*torch.log(1-y_pred)
        return torch.mean(bce)

    def eval_metric(self,y_pred,y_true):
        label_pred = torch.where(y_pred>0.5,torch.ones_like(y_pred,dtype=torch.float32),
                                 torch.zeros_like(y_pred,dtype=torch.float32))
        acc = torch.mean(1-torch.abs(y_true-label_pred))
        return acc

model = DNNModel()

def train_step(model,features,labels):
    prediction = model(features)
    loss = model.loss_func(prediction,labels)
    metric = model.loss_func(prediction,labels)

    loss.backward()

    for param in model.parameters():
        param.data = param.data - 0.01*param.grad.data

    model.zero_grad()
    return loss.item(),metric.item()


def train_model(model,epoches):
    for epoch in range(1,epoches):
        loss_list,metric_list = [],[]
        for features,labels in data_iter(X,Y):
            loss,metric = train_step(model,features,labels)
            loss_list.append(loss)
            metric_list.append(metric)
        if epoch%100 == 0:
            print("-----------"*8,datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'))
            print("epoch = ",epoch,"loss = ", np.mean(loss_list),"metric = ",np.mean(metric_list))


train_model(model,1000)
#
# fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize = (12,5))
# ax1.scatter(X_p[:,0],X_p[:,1], c="r")
# ax1.scatter(X_n[:,0],X_n[:,1],c = "g")
# ax1.legend(["positive","negative"]);
# ax1.set_title("y_true");
#
# Xp_pred = X[torch.squeeze(model.forward(X)>=0.5)]
# Xn_pred = X[torch.squeeze(model.forward(X)<0.5)]
#
# ax2.scatter(Xp_pred[:,0],Xp_pred[:,1],c = "r")
# ax2.scatter(Xn_pred[:,0],Xn_pred[:,1],c = "g")
# ax2.legend(["positive","negative"])
# ax2.set_title("y_pred")
# plt.show()