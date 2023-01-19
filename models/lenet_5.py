import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Lenet_5(nn.Module):
    def __init__(self,num_classes =10):
        super(Lenet_5,self).__init__()

        self.conv1 = nn.Conv2d(in_channels= 1,out_channels=6,kernel_size=5,stride=1)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1)

        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,num_classes)
        self.name = "Lenet-5"

    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = F.relu(self.conv2(out))
        out = F.avg_pool2d(out, kernel_size=2, stride=2)

        out = out.view(out.size(0),-1)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out

if __name__=="__main__":
    net = Lenet_5()
    for name, parameters in net.named_parameters():
        print(name, ';', parameters.size())

    
    print(net.name)
    summary(net,input_size = (1,32,32), device = 'cpu')
        



