# Tensorboard 使用

首先在Pytorch中指定一个目录创建一个torch.utils.tensorboard.SummaryWriter日志写入器。

然后根据需要可视化的信息，利用日志写入器将相应信息日志写入我们指定的目录。

最后就可以传入日志目录作为参数启动TensorBoard，然后就可以在TensorBoard中看到预设的信息。

* 可视化模型结构： writer.add_graph

* 可视化指标变化： writer.add_scalar  常用于在训练过程中实时动态地查看loss和各种metric的变化曲线（*标量*）
 
* 可视化参数分布： writer.add_histogram 常用于模型的参数(一般非标量)在训练过程中的变化可视化, 能够观测张量值分布的直方图随训练步骤的变化趋势。

* 可视化原始图像： writer.add_image 或 writer.add_images

* 可视化人工绘图： writer.add_figure

```python
from torch.utils.tensorboard import SummaryWriter

summary_path = './tensorboard'
writer = SummaryWriter(summary_path)
writer.add_graph(model,input_to_model=torch.rand(1,3,2,2))

for step in range(steps):
    # training process
    # ... 
    writer.add_histogram("weight",w, step)
    writer.add_scalar("loss",loss.item(),step) #日志中记录x在第step i 的值
    writer.add_scalar("eval_metric",metric.item(),step)
writer.close()
```

***对于图片的可视化***

> 如果只写入一张图片信息，可以使用writer.add_image。  
> 如果要写入多张图片信息，可以使用writer.add_images.  
> 也可以用 torchvision.utils.make_grid将多张图片拼成一张图片，然后用writer.add_image写入.  
> 注意，传入的是代表图片信息的Pytorch中的张量数据。
```python

# 仅查看一张图片
writer.add_image('images[0]', images[0])
writer.close()

# 将多张图片拼接成一张图片，中间用黑色网格分割
# create grid of images
img_grid = torchvision.utils.make_grid(images)
writer.add_image('image_grid', img_grid)
writer.close()

# 将多张图片直接写入
writer.add_images("images",images,global_step = 0)
writer.close()
```

可视化人工绘图
> 使用 add_figure  
> 注意，和writer.add_image不同的是，writer.add_figure需要传入matplotlib的figure对象。

```python
figure = plt.figure(figsize=(8,8))
writer.add_figure('figure',figure,global_step=0)
writer.close()

```