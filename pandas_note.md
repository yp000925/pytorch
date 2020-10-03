#Pandas
>主要收集一些利用pandas进行数据预处理的方法

>`pd.get_dummies()`可以用作进行one-hot编码

>`enumerate(iterable,start=0)` 一般用作循环中迭代计数
> ```python
> for iter,data in enumerate(dataloader,start=1):
>     foo
> ```

###长度不同的数据打包成为batch
在训练RNN网络时经常遇到不同长度的样本，进行mini-batch操作的时候没有办法封装到一起输入网络，这时候可以使用：
* `torch.nn.utils.rnn.pad_sequence()`：用于给 mini-batch 中的数据加 padding，让 mini-batch 中所有 sequence 的长度等于该 mini-batch 中最长的那个 sequence 的长度。
* `torch.nn.utils.rnn.pack_padded_sequence()`和 `torch.nn.utils.rnn.pad_packed_sequence()`结合使用：用于提高效率，避免 LSTM 前向传播时，把加入在训练数据中的 padding 考虑进去
        
在Datasetloader 中专门有个函数`collate_fn`来将Dataset的返回值拼接为一个tensor，不额外设置的时候会调用default的函数。当数据长度不一致的时候就需要修改这个函数。
    
```python
import torch.nn.utils.rnn as rnn_utils

def collate_fn(data):
  data.sort(key=lambda x: len(x), reverse=True)
  data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
  return data
data_loader = DataLoader(data, batch_size=3, shuffle=True, 
                         collate_fn=collate_fn)  
```
*注意* 这时候因为数据是由 0 作为后续padding的，而这本身在运算时没有任何意义，为了防止占用计算资源，`pack_padded_sequence()`和`pad_packed_sequence()`相结合使用。

* `pack_padded_sequence` 有三个参数：`input,lengths,batch_first`. `input`是上一步padding 过的数据，`lengths`是每条数据实际的有效长度，`batch_first`是指的数据是否按照[batch_size, sequence_length, data_dim] 排布。
输出有两部分，`data`和`batch_sizes`,`data`为原来的数据按照time step重新排列,而 padding 的部分，直接空过了。batch_sizes则是每次实际读入的数据量  
> 即 RNN 把一个 mini-batch sequence 又重新划分为了很多小的 batch，每个小 batch 为所有 sequence 在当前 time step 对应的值，如果某 sequence 在当前 time step 已经没有值了，那么，就不再读入填充的 0，而是降低 batch_size。batch_size相当于是对训练数据的重新划分。  
这也是为什么前面在 collate_fn中我们要对 mini-batch 中的 sequence 按照长度降序排列，是为了方便我们取每个 time step 的batch，防止中间夹杂着 padding。

```python
def collate_fn(data):
    data.sort(key=lambda x: len(x), reverse=True)
    data_length = [len(sq) for sq in data]
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    return data, data_length
    #注意这时的data需要为[batch_size, sequence_length, data_dim]格式
    
data_loader = DataLoader(data, batch_size=3, shuffle=True,
                         collate_fn=collate_fn)
net = nn.LSTM(1, 10, 2, batch_first=True)
    h0 = torch.rand(2, 3, 10)
    c0 = torch.rand(2, 3, 10)
batch_x, batch_x_len = iter(data_loader).next()
batch_x_pack = rnn_utils.pack_padded_sequence(batch_x,
                                              batch_x_len, batch_first=True)
out, (h1, c1) = net(batch_x_pack, (h0, c0))
```
