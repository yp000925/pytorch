#Pytorch API
##低阶
自己写优化步骤,网络参数等
举例：

example1：[线性回归](./pytorch_5.py)

example2：[DNN二分类](./pytorch_6.py)

##中阶
利用已经封装好的优化器和网络层，并且使用数据管道 `torch.utils.data.Dataset`,`torch.utils.data.DataLoader`,`torch.utils.data.TensorDataset`

在这里一般会用到`nn.functional.*`,注意该操作一般存在于`forward`方法中，而相应的同等功效的`nn.layers`则是已经定义好的层（里面的训练参数都是已经设置好的），
可以认为是继承了`nn.Module` 将相应的 `nn.functional.*` 操作进行进一步封装，所以一般含有训练参数，例如卷积层我们都用`nn.conv2d`而不用`nn.functinal.conv2d`(后者仅仅为卷积操作)，但`nn.functional.relu`一般替代`nn.Relu`
* 着重注意dropout层的处理，因为该层虽然没有训练参数，但是在model.eval()和model.train()的表现是不一致的，如果直接定义层会自动设置，但如果只用了操作，则要手动设置`F.dropout(x, training=self.training)`，所以建议使用`nn.dropout`
