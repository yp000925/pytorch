#  Pytorch 的一些基本概念和操作

### 对于tensor的基本操作
* 数据类型 `dtype` (除了str类型其余都有) 
    * 可以使用  `tensor.type_as(x)` 进行数据类型的转换
* 维度 `tensor.dim()`
* 尺寸（每一维的长度）`tensor.reshape()` or `tensor.contiguous().view()` 可以用来改变尺寸    


* 数组与tensor的转换：
    * 可以用numpy方法从Tensor得到numpy数组，也可以用torch.from_numpy从numpy数组得到Tensor。
    
            * 这两种方法关联的Tensor和numpy数组是共享数据内存的 -> 如果改变其中一个，另外一个的值也会发生改变。
            * 如果有需要，可以用张量的clone方法拷贝张量，中断这种关联。

    * 此外，还可以使用item方法从*标量*张量得到对应的Python数值。使用tolist方法从张量得到对应的Python数值列表。

* 数据运算 
    * **标量运算** 一些基本的运算符 + `torch.clamp()` `torch.fmod()` `torch.remainder()`
    * **向量运算** 在一个特定轴上运算 
        * `mean() max() sum() prod() min() std() var() median()`
        * `svd() qr() eig() norm() trace() inverse() ` 
        

Some commands:
*  generate random int:
    `randint = torch.floor(minval+(maxval-minval)*torch.randn([4,10,7])).int()`
*  tensor 的切片提取 `torch.index_select` `torch.take` `torch.gather` `torch.masked_select` ⚠️：不可更改值
*  若想更改数值，则使用 `torch.where` `torch.index_fill` `torch.masked_fill`
*  改变张量维度：  ⚠️ torch默认的图片格式为 **Batch,Height,Width,Channel**
    * `torch.reshape` or `torch.view`
    * `torch.squeeze` 减少维度
    * `torch.unsqueeze` 增加维度
    * `torch.transpose` 交换维度
* 合并： `torch.cat` `torch.stack`  分割：`torch.split` 

### 自动微分机制
* 对于*标量*张量来说`tensor.backward()`可以自动计算和该tensor相关的所有`requires_grad=True`的张量微分梯度，并存储在这些张量的grad属性下
* 对于非标量即维度不为0的张量，则要传入一个和它*同形状*的gradient参数张量，使其变成一个*标量*张量之后再进行计算
        
        Example： y.backward(gradient=gradient) #y和gradient为同形状的张量
        等价于： gradient = torch.tensor([[1.0,1.0],[1.0,1.0]])
                z = torch.sum(y*gradient)
                
* 还可以利用`torch.autograd.grad(y,x,create_graph=True)`求梯度（`create_graph=True`属性可以允许创建更高阶的导数）
```python
import numpy as np 
import torch 
x1 = torch.tensor(1.0,requires_grad = True) # x需要被求导
x2 = torch.tensor(2.0,requires_grad = True)

y1 = x1*x2
y2 = x1+x2

# 允许同时对多个自变量求导数
(dy1_dx1,dy1_dx2) = torch.autograd.grad(outputs=y1,inputs = [x1,x2],retain_graph = True)
print(dy1_dx1,dy1_dx2)

# 如果有多个因变量，相当于把多个因变量的梯度结果求和
(dy12_dx1,dy12_dx2) = torch.autograd.grad(outputs=[y1,y2],inputs = [x1,x2])
print(dy12_dx1,dy12_dx2)
```

Example: 利用自动微分和优化器求极值
```python
import numpy as np
import torch

# f(x) = a*x**2 + b*x + c的最小值

x = torch.tensor(0.0, requires_grad=True)  # x需要被求导
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)

optimizer = torch.optim.SGD(params=[x], lr=0.01)


def f(x):
    result = a * torch.pow(x, 2) + b * x + c
    return (result)


for i in range(500):
    optimizer.zero_grad()
    y = f(x)
    y.backward()
    optimizer.step()
    if i%50 == 0: 
        print("y=", f(x).data, ";", "x=", x.data)
```

```
y= tensor(0.9604) ; x= tensor(0.0200)
y= tensor(0.1274) ; x= tensor(0.6431)
y= tensor(0.0169) ; x= tensor(0.8700)
y= tensor(0.0022) ; x= tensor(0.9527)
y= tensor(0.0003) ; x= tensor(0.9828)
y= tensor(3.9399e-05) ; x= tensor(0.9937)
y= tensor(5.2452e-06) ; x= tensor(0.9977)
y= tensor(7.1526e-07) ; x= tensor(0.9992)
y= tensor(1.1921e-07) ; x= tensor(0.9997)
y= tensor(0.) ; x= tensor(0.9999)
```

### 利用tensorboard 可视化

可以利用 torch.utils.tensorboard 将计算图导出到 TensorBoard进行可视化。

```python
from torch import nn 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.w = nn.Parameter(torch.randn(2,1))
        self.b = nn.Parameter(torch.zeros(1,1))

    def forward(self, x):
        y = x@self.w + self.b
        return y

net = Net()

```

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./data/tensorboard')
writer.add_graph(net,input_to_model = torch.rand(10,2))
writer.close()

```

```python
%load_ext tensorboard
#%tensorboard --logdir ./data/tensorboard

```

```python
from tensorboard import notebook
notebook.list() 

```

```python
#在tensorboard中查看模型
notebook.start("--logdir ./data/tensorboard")
```


### 利用nn.Module 管理模块
`nn.Parameter() nn.ParameterList() nn.ParameterDict()`可以用来自定义参数（default requires_grad=True）

`nn.Module` 提供了一些方法管理自模块（可以想像成不同的layer）：

* children() 方法: 返回生成器，包括模块下的所有子模块。

* named_children()方法：返回一个生成器，包括模块下的所有子模块，以及它们的名字。

* modules()方法：返回一个生成器，包括模块下的所有各个层级的模块，包括模块本身。

* named_modules()方法：返回一个生成器，包括模块下的所有各个层级的模块以及它们的名字，包括模块本身。

其中 *chidren()* 方法和 *named_children()* 方法较多使用。

modules()方法和named_modules()方法较少使用，其功能可以通过多个named_children()的嵌套使用实现。


```python
# 冻结制定层（embedding层）参数
children_dict = {name:module for name,module in net.named_children()}

embedding = children_dict["embedding"]
embedding.requires_grad_(False) #冻结其参数

```

#### pytorch 的hook机制
[参考链接](https://zhuanlan.zhihu.com/p/87853615) 

pytorch中包含forward和backward两个钩子注册函数，用于获取forward和backward中输入和输出

> `nn.Module.register_forward_hook(hook: Callable[[...], None]) nn.Module.register_backward_hook(hook: Callable[[...], None])` 

[举例forward并利用forward写summary封装](./pytorch_forward_hook.py)


The hook will be called every time after forward() has computed an output. It should have the following signature:
> hook(module, input, output) -> None or modified output

hook（）函数是register_forward_hook()函数必须提供的参数，用户可以自行决定拦截了中间信息之后要做什么


### 使用GPU训练

当数据准备过程还是模型训练时间的主要瓶颈时，我们可以使用更多进程来准备数据。

当参数迭代过程成为训练时间的主要瓶颈时，我们通常的方法是应用GPU来进行加速。

<!-- #region -->
Pytorch中使用GPU加速模型非常简单，只要将模型和数据移动到GPU上。核心代码只有以下几行。

```python
# 定义模型
... 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device) # 移动模型到cuda

# 训练模型
...

features = features.to(device) # 移动数据到cuda
labels = labels.to(device) # 或者  labels = labels.cuda() if torch.cuda.is_available() else labels
...
```

如果要使用多个GPU训练模型，也非常简单。只需要在将模型设置为数据并行风格模型。
则模型移动到GPU上之后，会在每一个GPU上拷贝一个副本，并把数据平分到各个GPU上进行训练。核心代码如下。

```python
# 定义模型
... 

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model) # 包装为并行风格模型

# 训练模型
...
features = features.to(device) # 移动数据到cuda
labels = labels.to(device) # 或者 labels = labels.cuda() if torch.cuda.is_available() else labels
...
```
<!-- #endregion -->


### 保存和加载模型
A common PyTorch convention is to save models using either a .pt or .pth file extension.

Remember too, that you must call *model.eval()* to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.

```python
# Specify a path
PATH = "state_dict_model.pt"

# Save
torch.save(net.state_dict(), PATH)

# Load
model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()
```

or the entire model saved:
```python
# Specify a path
PATH = "entire_model.pt"

# Save
torch.save(net, PATH)

# Load
model = torch.load(PATH)
model.eval()
```

More information saved such as a general checkpoint:

```python
EPOCH = 5
PATH = "model.pt"
LOSS = 0.4

torch.save({
            'epoch': EPOCH,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)
            
            
model = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()
```