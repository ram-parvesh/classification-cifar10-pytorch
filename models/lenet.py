import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Lenet(nn.Module):
    def __init__(self):
        super(Lenet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16*5*5 , 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84, 10)
        self.name ="Lenet"

    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out , 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out , 2)

        out = out.view(out.size(0),-1)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out

if __name__== "__main__":
    net = Lenet()

    # net.named_parameters() #It is also an iterable object, which can only call out the specific parameters of the network, and also has name information
    for name, parameters in net.named_parameters():
        print(name, ';', parameters.size())

    
    print(net.name)
    summary(net,input_size = (3,32,32), device = 'cpu')
        

