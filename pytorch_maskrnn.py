#%%
import os
import numpy as np
import torch
from PIL import Image
import torch
from torch.utils.data import DataLoader,Subset


class PennFundanDataset():
    '''
    this is a class to load the pennfundan dataset for object detection
    :returns
        image: a PIL image with (HxW)
        target: a dict contains
                    boxes(FloatTensor[N,4]): the coordinates for bounding box [xmin,ymin,xmax,ymax]
                    labels(Int64Tensor[N]): the label for each bounding box, 0 represents the background
                    image_id(Int64Tensor[1]): an image identifier
                    area(Tensor[N]): The area of the bounding box, seperate the evaluation metric scores between small, medium and large boxes
                    iscrowd(UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation
                    (optionally) masks(UInt8Tensor[N,H,W] : The segmentation masks for each object
                    (optionally) keypoints(FloatTensor[N,K,3]): For each object, it contains K keypoints in [x,y,visibility] format
    '''
    def __init__(self,root, transform):
        self.root = root
        self.transform = transform
        self.imgs = list(sorted(os.listdir(os.path.join(root,'PNGImages'))))
        self.masks = list(sorted(os.listdir(os.path.join(root,'PedMasks'))))

    def __getitem__(self, idx):
        # load image and masks
        img_path = os.path.join(self.root,'PNGImages',self.imgs[idx])
        mask_path = os.path.join(self.root,'PedMasks',self.masks[idx])
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path) #do not need to convert into RBG, different values means different instance, 0 is for background
        mask = np.array(mask)
        obj_idx = np.unique(mask)
        obj_idx = obj_idx[1:] # remove the 0 value which stands for the background

        #split the color-encoded mask into a set of binary masks
        masks = mask == obj_idx[:, None, None]

        #get bounding box coords for each mask
        num_objs = len(obj_idx)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin,ymin,xmax,ymax])

        #convert into a torch tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class in this example for people
        labels = torch.ones((num_objs),dtype=torch.int64)
        masks = torch.as_tensor(masks,dtype=torch.uint8)

        img_id = torch.tensor([idx])
        area = (boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])
        #suppose all objects are not crowded
        iscrowd = torch.zeros((num_objs,),dtype=torch.int64)

        target= {}
        target['boxes'] = boxes
        target['masks'] = masks
        target['labels'] = labels
        target['iscrowd'] = iscrowd
        target['area'] = area
        target['image_id'] = img_id

        if self.transform is not None:
            img,target = self.transform(img,target)

        return img,target

    def __len__(self):
        return len(self.imgs)


#%%
# finetune the mask RCNN model from coco pretrained one for our task (person detection)

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)#只替换了最顶层的分类网络（全连接）

# #%%
# 暂时有问题
# # modify it with a different backbone
# from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection.rpn import AnchorGenerator
# import torchvision
#
# # load a pre-trained net
# backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# backbone.out_channels = 1280 #defined in mobilenet_v2, this parameter is needed to be set for FasterRCNN use
#
# anchor_generator = AnchorGenerator(sizes=((32,64,128,256,512),), aspect_ratios=((0.5,1.0,2.0)))
#
# roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2) #sample ratio指的是取哪几个周围的点做bilinear
#
# model = FasterRCNN(backbone, num_classes=2, box_roi_pool=roi_pooler)

#%%
import pytorch_detection_example_transforms as T


def get_transform(train):
    transform = []
    transform.append(T.ToTensor())
    if train:
        transform.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transform)

def collate_fn(batch):
    return tuple(zip(*batch))


#%%
import pytorch_detection_example_utils as utils
import math
import sys

def train_one_epoch(model,optimizer,data_loader,device,epoch,print_freq):
    model.train()
    header = 'Epoch:{}'.format(epoch)
    # metric_logger = utils.MetricLogger(delimiter = '  ')
    # metric_logger.add_meter('lr',utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    for i,(images, targets) in enumerate(data_loader,1):

        images = list(image.to(device) for image in images)
        targets = [{k:v.to(device) for k,v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = utils.reduce_dict(loss_dict) #分布式的时候可以用到

        loss_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = loss_reduced.item()

        if not math.isfinite(loss_value):
            print("loss is {}, stop training!".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        if i % print_freq == 0:
            print(header)
            loss_str = []
            for name, meter in loss_dict_reduced.items():
                loss_str.append(
                    "{}: {}".format(name, str(meter))
                )
            print('Iteration:{:d}, Meters: {loss_dict}'.format(i, loss_dict=str('   '.join(loss_str))))

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    return



device = torch.device('cpu')

num_classes = 2
dataset = PennFundanDataset('data/Pennfudanped', get_transform(train=True))
dataset_test = PennFundanDataset('data/Pennfudanped', get_transform(train=False))

indices = torch.randperm(len(dataset)).tolist()
dataset = Subset(dataset, indices[:-50])
dataset_test = Subset(dataset_test, indices[-50:])

data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
data_loader_test  = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn = collate_fn)

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params,lr=0.005,momentum=0.9,weight_decay=0.0005)

epoch_num=10
for epoch in range(epoch_num):
    train_one_epoch(model,optimizer,data_loader,device,epoch,print_freq=10)
    break






