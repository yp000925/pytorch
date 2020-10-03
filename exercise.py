import torch

def log_softmax(x):
    return x-x.exp().sum(-1).log().unsqueeze(-1)

x = torch.tensor([0,0,0,1,3],dtype=torch.float32)
log_softmax(x)

def nll(preds,labels):
    # here the preds is from logSoftmax
    # Loss = -log(Softmax).sum() 对于label 为1
    return -preds[range(labels.shape(0)),labels].mean()

def accuracy(out,labels):
    preds = torch.argmax(out,dim=1)
    return (preds==labels).float().mean()

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

print()

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])