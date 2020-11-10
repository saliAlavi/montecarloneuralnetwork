import torch
import torch.nn as nn
import torch.nn.functional as F



class LinearNet(torch.nn.Module):
    def __init__(self,input_size):
        super(LinearNet, self).__init__()

        self.fc0 = nn.Linear(input_size,96)
        self.fc1 = nn.Linear(96, 128)
        self.fc2 = nn.Linear(128, 256)

        self.fct2 = nn.Linear(256, 128)
        self.fct1 = nn.Linear(128, 96)
        self.fct0 = nn.Linear(96, input_size)

        self.fco1 = nn.Linear(input_size,32)
        self.fco2 = nn.Linear(32,1)

    def forward(self,x):
        res0=x
        res1 =F.relu(self.fc0(res0))
        res2 =F.relu(self.fc1(res1))
        res3=F.relu(self.fc2(res2))
        out=F.relu(self.fct2( res3))
        out=F.relu(out+ res2)
        out=F.relu(self.fct1(out ))
        out=F.relu(out+ res1)
        out=F.relu(self.fct0(out))
        out =F.relu(out+ res0)
        out= out.reshape(x.size(0),-1)
        out = self.fco1(out)
        out = F.relu(out)
        out = self.fco2(out)

        return out



class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size,kernel,padding=0,stride=1):
        super(ConvBlock, self).__init__()
        self.conv=nn.Conv1d(input_size,output_size,kernel_size=kernel,padding=padding, stride=stride)

    def forward(self,x):
        x = self.conv(x)
        x = F.relu(x)
        return x


class ConvTransposeBlock(torch.nn.Module):
    def __init__(self,input_size, output_size,kernel,padding=0,stride=1):
        super(ConvTransposeBlock, self).__init__()
        self.convtransp=nn.ConvTranspose1d(input_size, output_size,kernel_size=kernel,padding=padding,stride=stride)

    def forward(self,x):
        x = self.convtransp(x)
        x  = F.relu(x)
        return x

