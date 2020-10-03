#
# import torch
# from torch import nn
#
# #%%
# class TestNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.edges = torch.arange(10 + 1).float() / 10
#         self.edges[-1]+=1e-6
#
#     def forward(self,x):
#         num_in_bin=[]
#         for i in range(10):
#             inds = (x >= self.edges[i]) & (x < self.edges[i+1])
#             num_in_bin.append(inds.sum().item())
#         return num_in_bin
#
# input = torch.tensor([[1],[0.21],[0.21],[0.21],[0.34],[0.76]])
# net = TestNet()
# net(input)


#%%
import torchvision
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# class ImageTranspose():
#     def __call__(self, img):
#         img = img.permute([0,2,1])
#         return img
#
#
# transform = torchvision.transforms.Compose([ImageTranspose(), torchvision.transforms.ToTensor()])
#
# img1 = torch.rand([3,224,224])

class TestDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.basepath = "data/cifar2/train/0_airplane/"

    def __getitem__(self, index):
        img_path = self.basepath + str(int(index))+'.jpg'
        img = Image.open(img_path)
        img = img.rotate(90)
        img_arr = np.array(img)
        # data = torch.tensor(img_arr)
        return img_arr

fig = plt.figure()
img_data = TestDataset()

for i in range(4):
    sample = img_data[i]
    print(i, sample.shape)
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    plt.imshow(sample)

    if i == 3:
        plt.show()
        break