import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import os

class AlexNet(nn.Module):
    def __init__(self,num_classes=1000):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,padding=1)

        self.fc1 = nn.Linear(6*6*256,4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096,num_classes)

        self.name = "AlexNet"

    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out,kernel_size=3, stride=2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, kernel_size =3, stride= 2)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.max_pool2d(out, kernel_size=3, stride=2)

        out = out.view(out.size(0),-1)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        # out = nn.Softmax(out)
        
        return out
if __name__=="__main__":
    net = AlexNet()
    for name, parameters in net.named_parameters():
        print(name,":",parameters.size())
    
    print(net.name)
    summary(net,input_size = (3,227,227), device = 'cpu')





        

