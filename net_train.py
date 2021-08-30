import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import cv2
T=60
BC=0.1
XLLC=600
train_dataset = datasets.MNIST(root='./data/',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = datasets.MNIST(root='./data/',train=False,transform=transforms.ToTensor(),download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=T,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=60,
                                          shuffle=False)
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.c1=nn.Conv2d(1,10,3,1,1,bias=False)
        self.c2=nn.Conv2d(10,28,3,1,1,bias=False)
        self.c3=nn.Conv2d(28,28,3,1,1,bias=False)
        self.l1=nn.Linear(28*3*3,28*3*2)
        self.l2=nn.Linear(28*3*2,100)
        self.pool=nn.MaxPool2d(2,2)
        self.bn10=nn.BatchNorm2d(10)
        self.bn28= nn.BatchNorm2d(28)
        self.bn28_1=nn.BatchNorm2d(28)

    def forward(self,x):
        x = self.pool(F.relu(self.bn10(self.c1(x))))
        x = self.pool(F.relu(self.bn28(self.c2(x))))
        x = self.pool(F.relu(self.bn28_1(self.c3(x))))
        x=x.view(-1,28*3*3)
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        return x
class Net1(nn.Module):
    def __init__(self):
        super(Net1,self).__init__()
        self.l2_=nn.Linear(100,28*3*3)
        self.l3=nn.Linear(28*3*3,600)
        self.l4=nn.Linear(600,2700)
        self.c4=nn.Conv2d(3,28,3,1,0,bias=False)
        self.c5=nn.Conv2d(28,1,1,1,0,bias=False)
        self.pool=nn.MaxPool2d(2,2)
        self.bn28_2=nn.BatchNorm2d(28)

    def forward(self,x):
        x=F.relu(self.l2_(x))
        x=F.relu(self.l3(x))
        x=F.relu(self.l4(x))
        x=x.view(-1,3,30,30)
        x=F.relu(self.bn28_2(self.c4(x)))
        x=F.relu(self.c5(x))
        return x
mod=Net()
mod1=Net1()
mod=mod.cuda()
mod1=mod1.cuda()
loss=nn.MSELoss()
loss1=nn.CrossEntropyLoss()
optimizer=optim.SGD(mod.parameters(),lr=BC)
schedulerD = MultiStepLR(optimizer, milestones=[50,150,300,450], gamma=0.8)
optimizer1=optim.SGD(mod1.parameters(),lr=BC)
schedulerD1 = MultiStepLR(optimizer1, milestones=[50,150,300,450], gamma=0.8)
params=list(mod.parameters())
params1=list(mod1.parameters())
pj=torch.zeros(10,100)
for i in range(10):
    for j in range(10):
        pj[i,10*i+j]=1
pj=pj.cuda()
def func(y):
    o=torch.zeros(T,100)
    o=o.cuda()
    for i in range(T):
        k=y[i].int()
        o[i,:]=pj[k,:]
    return o
def func1(w,y):
    o=torch.zeros(T,100)
    o=o.cuda()
    for i in range(T):
        k=y[i].int()
        o[i,:]=w[i]*pj[k,:]
    return o
def func2(w):
    a=torch.zeros(T,10)
    a=a.cuda()
    for i in range(T):
        for j in range(10):
            a[i,j]=-loss(w[i],pj[j,:])
    return a
for i in range(XLLC):
    print(i)
    train_loss=0
    train_loss1=0
    train_loss2=0
    q1=torch.sum(params[0])
    q11 = torch.sum(params1[0])
    print(q1,q11)
    for x,y in train_loader:
        x = Variable(x)
        x = x.cuda()
        y = y.cuda()
        mod.requires_grad_(True)
        mod.train()
        mod1.requires_grad_(False)
        mod1.eval()
        out=mod(x)
        outs=func2(out)
        out1=mod1(func(y)*out)
        lossvaluex1=loss(out1,x)
        lossvaluex2 = loss1(outs, y)
        lossvalue=lossvaluex1+lossvaluex2
        optimizer.zero_grad()
        lossvalue.backward()
        optimizer.step()
        schedulerD.step()
        mod1.requires_grad_(True)
        mod1.train()
        mod.requires_grad_(False)
        mod.eval()
        out_=mod(x)
        out2=mod1(func1(out_,y))
        lossvalue1=loss(out2,x)
        optimizer1.zero_grad()
        lossvalue1.backward()
        optimizer1.step()
        schedulerD1.step()
        train_loss+=lossvalue
        train_loss1+=lossvalue1
        train_loss2 += lossvaluex2
    print(train_loss,train_loss1,train_loss2)
torch.save(mod.state_dict(),'mod-lx1-100-1.pt')
torch.save(mod1.state_dict(),'mod-lx2-100-1.pt')








