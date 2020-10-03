import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset,TensorDataset
import torch
from pytorch_4_net import Net
import torchkeras

df = pd.read_csv("data/covid-19.csv", sep='\t')
# df.plot(x = "date", y= ['confirmed_num',"cured_num","dead_num"],figsize = (10,6))
# plt.xticks(rotation = 60)
dfdata = df.set_index("date")
dfdiff = dfdata.diff(periods=1).dropna()
dfdiff = dfdiff.reset_index("date")

dfdiff = dfdiff.drop("date", axis=1).astype("float32")

WINDOWSIZE = 8

class CovidDataset(Dataset):

    def __len__(self):
        return len(dfdiff)-WINDOWSIZE

    def __getitem__(self, i):
        x = dfdiff.loc[i:i+WINDOWSIZE-1, :]
        feature = torch.tensor(x.values)
        y = dfdiff.loc[i+WINDOWSIZE, :]
        label = torch.tensor(y.values)
        return (feature,label)

ds_train = CovidDataset()
dl_train = DataLoader(ds_train, batch_size=38)

net = Net()


def mspe(y_pred,y_true):
    err_percent = (y_true-y_pred)**2/(torch.max(y_true**2,torch.tensor(1e-7)))
    return torch.mean(err_percent)

# model = net
# model.optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1)
# model.loss_func = mspe
# 
# def train_step(model,features,labels):
#     model.train()
#     model.optimizer.zero_grad()
#
#     predicts = model(features)
#     loss = model.loss_func(predicts,labels)
#
#     loss.backward()
#     model.optimizer.step()
#
#     return loss.item()
#
# def train_model(model,epochs, dl_train,log_step_freq):
#     dfhistory = pd.DataFrame(columns=['epoch','loss'])
#     print("Start training")
#     print("========="*8)
#
#     for epoch in range(1,epochs+1):
#         loss_sum = 0.0
#         step = 1
#         for step, (features,labels) in enumerate(dl_train,1):
#             loss = train_step(model,features,labels)
#             loss_sum += loss
#             if step % log_step_freq == 0:
#                 print(("step: %d, loss: %.3f")%(step,loss_sum/step))
#
#         info = (epoch,loss_sum/step)
#         dfhistory.loc[epoch-1] = info
#
#         print(("epoch: %d, loss: %.3f")%info)
#         print("========="*8)
#
#     print("finish training")
#     return dfhistory
#
# dfhistory=train_model(model,3,dl_train,log_step_freq=1)
#
#
#


model = torchkeras.Model(net)
model.compile(loss_func=mspe,optimizer=torch.optim.Adagrad(model.parameters(),lr=0.1))
dfhistory = model.fit(2,dl_train,log_step_freq=1)

