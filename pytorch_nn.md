#torch.nn 一些常用函数和操作积累和记录贴

## torch.nn.functional

> `import torch.nn.funtional as F`

`F.cross_entropy` 作为loss function，可以实现log_softmax和nll loss 的结合，故输入的pred时候不用再经过activation层（log_softmax）

## torch.nn.Module 
*
1. 可用于控制集中控制网络的参数，梯度等，包含了`.parameters()` `.zero_grad()` `.no_grad()`等attribute 

2. 其中`no_grad` 
> * 将推断（inference）的代码包裹在 `with torch.no_grad():` 之中，以达到*暂时*不追踪网络参数中的导数的目的，为了减少可能存在的计算和内存消耗
> * 作为修饰器放在eval()之前
>> ```python
>> @torch.no_grad()
>> def eval():
>>     ...
>> ```

3\. `model.eval()` `model.train()` 用来控制 `nn.BatchNorm2d` and `nn.Dropout`等在不同phases中有不同表现的layers

## torch.utils.data 

`torch.utils.data.Dataset` `torch.utils.data.TensorDataset()` 自定义`__getitem__(): return`

PyTorch’s TensorDataset is a Dataset wrapping tensors. By defining a length and way of indexing, this also gives us a way to *iterate, index, and slice* along the first dimension of a tensor. 

This will make it easier to access both the independent and dependent variables in the same line as we train


`torch.utils.data.DataLoader` 自定义`__iter__(): yield `

Pytorch’s DataLoader is responsible for managing batches.You can create a DataLoader from any Dataset. 

The DataLoader gives us each mini-batch automatically.

##  Lambda 层

一般用作于和torch.nn.Sequential一起简化模型定义

Example： 定义`view()`层
```python
from torch import nn

class Lambda(nn.Module):
    def __init__(self,func):
        super().__init__()
        self.func = func
    def forward(self,x):
        return self.func(x)
        
def preprocess(x):
    return x.view(-1,1,28,28)


model = nn.Sequential(
    Lambda(preprocess),
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),
)
```

另外一种实现对batch数据一起操作的方法是加入在自定义dataloader里面

```python
def preprocess(x, y):
    return x.view(-1, 1, 28, 28), y


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))
                        
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True),
valid_dl = DataLoader(valid_ds, batch_size=bs * 2),        
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)
```

