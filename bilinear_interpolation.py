import numpy as np
import torch

# numpy version

def bilinear_interpolate_numpy(im, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return (Ia.T*wa).T + (Ib.T*wb).T + (Ic.T*wc).T + (Id.T*wd).T

# tensor version

dtype = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor


def bilinear_interpolate_torch(im, x, y):
    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1

    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0][0]
    Ib = im[y1, x0][0]
    Ic = im[y0, x1][0]
    Id = im[y1, x1][0]

    wa = (x1.type(dtype) - x) * (y1.type(dtype) - y)
    wb = (x1.type(dtype) - x) * (y - y0.type(dtype))
    wc = (x - x0.type(dtype)) * (y1.type(dtype) - y)
    wd = (x - x0.type(dtype)) * (y - y0.type(dtype))

    return torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(
        torch.t(Id) * wd)



#%%
# use nn.functional.grid_sample
'''
nn.functional.grid_sample(input, grid,mode='mode='bilinear', padding_mode='zeros'):
   :param
        input : feature maps , the shape should be NxCxIHxIW
        grid:   flow-field of size (N x OH x OW x 2) , should be normalized to the range [-1,1],
                x: 1, y: 1 is the right-bottom pixel of the input
   :return
        output Tensor
    
'''
import torch.nn.functional
import torch
dtype = torch.FloatTensor
dtype_long = torch.LongTensor

def bilinear_interpolate_torch_gridsample(image, samples_x, samples_y):
                                                # input image is: W x H x C
    image = image.permute(2,0,1)                # change to:      C x W x H
    image = image.unsqueeze(0)                  # change to:  1 x C x W x H
    samples_x = samples_x.unsqueeze(2)
    samples_x = samples_x.unsqueeze(3)
    samples_y = samples_y.unsqueeze(2)
    samples_y = samples_y.unsqueeze(3)
    samples = torch.cat([samples_x, samples_y],3)
    samples[:,:,:,0] = (samples[:,:,:,0]/(W-1)) # normalize to between  0 and 1
    samples[:,:,:,1] = (samples[:,:,:,1]/(H-1)) # normalize to between  0 and 1
    samples = samples*2-1                       # normalize to between -1 and 1
    return torch.nn.functional.grid_sample(image, samples)

# Correctness test
W, H, C = 5, 5, 1
test_image = torch.ones(W,H,C).type(dtype)
test_image[3,3,:] = 4
test_image[3,4,:] = 3

test_samples_x = torch.FloatTensor([[3.2]]).type(dtype)
test_samples_y = torch.FloatTensor([[3.4]]).type(dtype)

print(bilinear_interpolate_torch_gridsample(test_image, test_samples_x, test_samples_y))
