# 自定义损失函数

⚠️ pytorch中的损失函数为 y_pred在前，y_true在后，和tensorflow相反

自定义损失函数需要接收两个张量y_pred，y_true作为输入参数，并输出一个标量作为损失函数值。

## 内置losses

对于回归模型，通常使用的内置损失函数是均方损失函数nn.MSELoss 。

对于二分类模型，通常使用的是二元交叉熵损失函数`nn.BCELoss` *(输入已经是sigmoid激活函数之后的结果)*
或者 `nn.BCEWithLogitsLoss` *(输入尚未经过nn.Sigmoid激活函数)* 。

对于多分类模型，一般推荐使用交叉熵损失函数 `nn.CrossEntropyLoss`。
(y_true需要是一维的，是类别编码。*y_pred未经过`nn.Softmax`激活。*) 

此外，如果多分类的y_pred经过了`nn.LogSoftmax`激活，可以使用`nn.NLLLoss`损失函数(The negative log likelihood loss)。
这种方法和直接使用nn.CrossEntropyLoss等价。
常用的一些内置损失函数说明如下。


* nn.MSELoss（均方误差损失，也叫做L2损失，用于回归）

* nn.L1Loss （L1损失，也叫做绝对值误差损失，用于回归）

* nn.SmoothL1Loss (平滑L1损失，当输入在-1到1之间时，平滑为L2损失，用于回归)

* nn.BCELoss (二元交叉熵，用于二分类，输入已经过nn.Sigmoid激活，对不平衡数据集可以用weigths参数调整类别权重)

* nn.BCEWithLogitsLoss (二元交叉熵，用于二分类，输入未经过nn.Sigmoid激活)

* nn.CrossEntropyLoss (交叉熵，用于多分类，要求label为稀疏编码，输入未经过nn.Softmax激活，对不平衡数据集可以用weigths参数调整类别权重)

* nn.NLLLoss (负对数似然损失，用于多分类，要求label为稀疏编码，输入经过nn.LogSoftmax激活)

* nn.CosineSimilarity(余弦相似度，可用于多分类)

* nn.AdaptiveLogSoftmaxWithLoss (一种适合非常多类别且类别分布很不均衡的损失函数，会自适应地将多个小类别合成一个cluster)

内置的损失函数一般有类的实现和函数的实现两种形式。

如：nn.BCE 和 F.binary_cross_entropy 都是二元交叉熵损失函数，前者是类的实现形式，后者是函数的实现形式。

实际上类的实现形式通常是调用函数的实现形式并用nn.Module封装后得到的。

一般我们常用的是类的实现形式。它们封装在torch.nn模块下，并且类名以Loss结尾。


更多损失函数的介绍参考如下知乎文章：

《PyTorch的十八个损失函数》

https://zhuanlan.zhihu.com/p/61379965


## 自定义loss
自定义损失函数接收两个张量y_pred,y_true作为输入参数，并输出一个 *标量* 作为损失函数值。

也可以对nn.Module进行子类化，重写forward方法实现损失的计算逻辑，从而得到损失函数的类的实现

Example: Focal loss

详见《5分钟理解Focal Loss与GHM——解决样本不平衡利器》

https://zhuanlan.zhihu.com/p/80594704

```python
import torch
import torch.functional as F
from torch import nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0,alpha=0.75):
        super().__init__()
        self.gamma = torch.tensor(gamma)
        self.alpha = torch.tensor(alpha)
        
    def forward(self,y_pred,y_true):
        p_t = y_true*(1-y_pred)+ (1-y_true)*y_pred
        focal_weight = (y_true*self.alpha+(1-y_true)*(1-self.alpha))*torch.pow(p_t,self.gamma)
        bce = nn.BCELoss(reduction = 'none')(y_pred,y_true)
        loss = torch.mean(bce*focal_weight)
        return loss
        
```
```python
#困难样本
y_pred_hard = torch.tensor([[0.5],[0.5]])
y_true_hard = torch.tensor([[1.0],[0.0]])

#容易样本
y_pred_easy = torch.tensor([[0.9],[0.1]])
y_true_easy = torch.tensor([[1.0],[0.0]])

focal_loss = FocalLoss()
bce_loss = nn.BCELoss()

print("focal_loss(hard samples):", focal_loss(y_pred_hard,y_true_hard))
print("bce_loss(hard samples):", bce_loss(y_pred_hard,y_true_hard))
print("focal_loss(easy samples):", focal_loss(y_pred_easy,y_true_easy))
print("bce_loss(easy samples):", bce_loss(y_pred_easy,y_true_easy))

#可见 focal_loss让容易样本的权重衰减到原来的 0.0005/0.1054 = 0.00474
#而让困难样本的权重只衰减到原来的 0.0866/0.6931=0.12496
# 因此相对而言，focal_loss可以衰减容易样本的权重。
```

## 正则化
一般来说，监督学习的目标函数由损失函数和正则化项组成。(Objective = Loss + Regularization)

通常认为L1 正则化可以产生稀疏权值矩阵，即产生一个稀疏模型，可以用于特征选择。

而L2 正则化可以防止模型过拟合（overfitting）。一定程度上，L1也可以防止过拟合。

将正则化项添加到目标损失函数来训练模型

Example：

```python
import torch

def L2Loss(model, alpha):
    l2_loss = torch.tensor(0.0,requires_grad=True)
    for name,param in model.named_parameters():#一般不对bias做正则化    
        if 'bias' not in name:
            l2_loss = l2_loss+ (0.5 * alpha * torch.sum(torch.pow(param, 2)))
    return l2_loss
    
def L1Loss(model,beta):
    L1_loss = torch.tensor(0.0,requires_grad=True)
    for name,param in model.named_parameters():#一般不对bias做正则化 
        if 'bias' not in name:
            l1_loss = l1_loss + beta * torch.sum(torch.abs(param))
    return l1_loss

# 将L2正则和L1正则添加到FocalLoss损失，一起作为目标函数
def focal_loss_with_regularization(y_pred,y_true):
    focal = FocalLoss()(y_pred,y_true) 
    l2_loss = L2Loss(model,0.001) #注意设置正则化项系数
    l1_loss = L1Loss(model,0.001)
    total_loss = focal + l2_loss + l1_loss
    return total_loss
```


> 如果仅仅需要使用L2正则化，那么也可以利用优化器的weight_decay参数来实现。

> weight_decay参数可以设置参数在训练过程中的衰减，这和L2正则化的作用效果等价


```
before L2 regularization:

gradient descent: w = w - lr * dloss_dw 

after L2 regularization:

gradient descent: w = w - lr * (dloss_dw+beta*w) = (1-lr*beta)*w - lr*dloss_dw

so （1-lr*beta）is the weight decay ratio.
```


Pytorch的优化器支持一种称之为Per-parameter options的操作，就是对每一个参数进行特定的学习率，权重衰减率指定，以满足更为细致的要求。

```python
import torch 

weight_params = [param for name, param in model.named_parameters() if "bias" not in name]
bias_params = [param for name, param in model.named_parameters() if "bias" in name]

optimizer = torch.optim.SGD([{'params': weight_params, 'weight_decay':1e-5},
                             {'params': bias_params, 'weight_decay':0}],
                            lr=1e-2, momentum=0.9)

```