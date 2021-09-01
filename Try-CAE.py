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
T=1
train_dataset = datasets.MNIST(root='./data/',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = datasets.MNIST(root='./data/',train=False,transform=transforms.ToTensor(),download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=T,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=T,
                                          shuffle=True)
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
loss1=nn.CrossEntropyLoss()
loss2=nn.MSELoss()
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
            a[i,j]=-loss2(w[i],pj[j,:])
    return a
mod.load_state_dict(torch.load('mod-lx1-100-1.pt'))
mod1.load_state_dict(torch.load('mod-lx2-100-1.pt'))
mod.eval()
mod1.eval()
i=0
j=0
def func_(w,k):
    a=torch.zeros(10)
    a=a.cuda()
    for i in range(10):
        a[i]=w[10*i]+w[10*i+1]+w[10*i+2]+w[10*i+3]+w[10*i+4]+w[10*i+5]+w[10*i+6]+w[10*i+7]+w[10*i+8]+w[10*i+9]

    aa,aaa=torch.sort(a)
    if a[k]==aa[9]:
        return aa[9]-aa[8]
    else:
        return aa[8]-aa[9]

def func_1(w):
    a=torch.zeros(10)
    a=a.cuda()
    for i in range(10):
        a[i]=loss2(w[0],pj[i,:])
    aa,aaa=torch.sort(a)
    return aaa[0]
def func11(w,y):
    o=torch.zeros(T,100)
    o=o.cuda()
    for i in range(T):
        o[i,:]=w[i]*pj[y,:]
    return o
def funh(w):
    o=torch.mm(w,pj.t())
    return o
def nce(x,w,a):
    y=torch.zeros(1)
    y=y.cuda()
    for i in range(10):
        xx=mod1(func1(w,y))
        lossx=loss2(xx,x)
        if lossx<a:
            return 0
        if lossx>a:
            y+=1
    return 1
Problem_image_by_S2=0
Image_by_S3_with_correct_label=0
Image_by_S3_with_wrong_label=0
Problem_image_by_S4=0
Problem_image_by_S5=0
Image_by_S5_with_correct_label=0
Image_by_S5_with_wrong_label=0
mod.eval()
for x,y in test_loader:
    x=x.cuda()
    y=y.cuda()
    k=int(y)
    w=mod(x)
    qq = func_1(w)
    iq = int(qq)
    if abs(func_(w[0],iq))<80:
        Problem_image_by_S2+=1
    else:
        xx = mod1(func11(w, iq))
        l=loss2(x,xx)
        if l<0.04:
            if iq==int(y):
                Image_by_S3_with_correct_label+=1
            else:
                Image_by_S3_with_wrong_label+=1
        if l>0.09:
            Problem_image_by_S4+=1
        if 0.04<l and l<0.09:
            ln=nce(x,w,l)
            if ln==0:
                Problem_image_by_S5+=1
            else:
                if iq == int(y):
                    Image_by_S5_with_correct_label += 1
                else:
                    Image_by_S5_with_wrong_label += 1

print("Problem_image_by_S2" + '='+str(Problem_image_by_S2))
print("Image_by_S3_with_correct_label" + '='+str(Image_by_S3_with_correct_label))
print("Image_by_S3_with_wrong_label" + '='+str(Image_by_S3_with_wrong_label))
print("Problem_image_by_S4" + '='+str(Problem_image_by_S4))
print("Problem_image_by_S5" + '='+str(Problem_image_by_S5))
print("Image_by_S5_with_correct_label" + '='+str(Image_by_S5_with_correct_label))
print("Image_by_S5_with_wrong_label" + '='+str(Image_by_S5_with_wrong_label))
