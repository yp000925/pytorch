import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,TensorDataset
import datetime
from torchvision import transforms,datasets
from pytorch_2_net import *
from sklearn.metrics import roc_auc_score

train_transfroms = transforms.Compose([transforms.ToTensor()])
valid_transforms = transforms.Compose([transforms.ToTensor()])

ds_train = datasets.ImageFolder("data/cifar2/train",transform=train_transfroms,
                                target_transform= lambda t: torch.tensor([t]).float()
                                )
ds_valid = datasets.ImageFolder("data/cifar2/test",transform=valid_transforms,
                                target_transform= lambda t: torch.tensor([t]).float()
                                )

dl_train = DataLoader(ds_train,batch_size=50,shuffle=True,num_workers=3)
dl_valid = DataLoader(ds_valid,batch_size=50,shuffle=True,num_workers=3)


net = Net()
model = net
model.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
model.loss_func = torch.nn.BCELoss()
model.metric_func = lambda ypred,ytrue: roc_auc_score(ytrue.data.numpy(),ypred.data.numpy())
model.metric_name = 'auc'

def train_step(model,features,labels):
    # 训练模式，dropout层发生作用
    model.train()

    #梯度清零
    model.optimizer.zero_grad()

    #正向传播求梯度
    predictions = model(features)
    loss = model.loss_func(predictions,labels)
    metric = model.metric_func(predictions,labels)

    #反向传播求梯度
    loss.backward()
    model.optimizer.step()

    return loss.item(), metric

def valid_step(model,features,labels):
    # 测试模式，dropout层不发生作用
    model.eval()
    # 关闭梯度计算
    with torch.no_grad():
        predictions = model(features)
        loss = model.loss_func(predictions,labels)
        metric = model.metric_func(predictions,labels)

    return loss.item(), metric

def train_model(model,epochs, dl_train, dl_valid, log_step_freq):
    metric_name = model.metric_name
    dfhistory = pd.DataFrame(columns= ['epoch','loss',metric_name,'val_loss','val_'+metric_name])
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("start training...")
    print("========="*8 + "%s" % nowtime)

    for epoch in range(1,epochs+1):

        # 训练循环
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        for step, (feature,labels) in enumerate(dl_train,1):
            loss, metric = train_step(model,feature,labels)

            loss_sum+=loss
            metric_sum+=metric
            if step % log_step_freq == 0:
                print(('[step = %d] loss: %.3f, '+ metric_name + ": %.3f")% (step,loss_sum/step,metric_sum/step))

        # 验证循环
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step =1

        for val_step,(feature,labels) in enumerate(dl_valid,1):
            val_loss, val_metric = valid_step(model,feature,labels)
            val_loss_sum += val_loss
            val_metric_sum += val_metric

        info = (epoch,loss_sum/step,metric_sum/step,val_loss_sum/val_step,val_metric_sum/val_step)
        dfhistory.loc[epoch-1]=info

        print(("\nEPOCH = %d, loss = %.3f," + metric_name + ": %.3f, val_loss = %.3f,"+"val_"+metric_name+"= %.3f")%info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n"+"========="*8 + "%s" % nowtime)
    print("Finished Training....")

    return dfhistory

epochs = 2
dfhistory = train_model(model,epochs,dl_train,dl_valid,log_step_freq=50)



def predict(model,dl):
    model.eval()
    with torch.no_grad():
        result = torch.cat([model.forward(t[0]) for t in dl])
    return result.data

#预测概率
y_pred_probs = predict(model,dl_valid)

#预测类别
y_pred_labels = torch.where(y_pred_probs>0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))

# #保存和加载模型 pkl文件
# torch.save(model.state_dict(),"./model_parameter.pkl")
#
# net_clone = Net()
# net_clone.load_state_dict(torch.load("./model_parameter.pkl"))
