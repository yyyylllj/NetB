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
                                           shuffle=False)
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
def cg(x):
    a=torch.sum(x)/784
    b=torch.zeros(x.size())
    for i in range(28):
        for j in range(28):
            if x[0][0][i][j]>a:
                b[0][0][i][j]+=1
    b=b.cuda()
    return b
def xsd(x,y):
    a=abs(x-y)
    b=torch.sum(a)
    c=torch.sum(x)
    d=torch.sum(y)
    return b/(c+d)
def nce(x,w):
    cgx=cg(x)
    y=torch.zeros(1)
    y=y.cuda()
    L=torch.zeros(10)
    L=L.cuda()
    for i in range(10):
        xx=mod1(func1(w,y))
        cgxx=cg(xx)
        L[i]=loss2(xx,x)*xsd(cgx,cgxx)
        y+=1
    return L


def Po(x):
    y=-14*14*x*x
    z=y.exp()
    return z

def pzd(L):
    w = torch.zeros(10)
    w = w.cuda()
    for i in range(10):
        w[i]=1/L[i]
    #a=torch.sum(w)
    #for i in range(10):
    #    if (1-p)*w[i]/a.item()<0.1:
    #        w[i]=w[i]-w[i]
    return w

def Pi(l,p):
    ww=pzd(l)
    cxx=[]
    sw=torch.sum(ww)
    w = torch.zeros(10)
    w = w.cuda()
    for i in range(10):
         w[i]=(1-p)*ww[i]/sw
         if w[i]>0.1:
            cxx.append((i))
    return w,cxx
wx=0
wxx=torch.zeros(10)
Num_of_Label=torch.zeros(10)
Num_of_Label_correct=torch.zeros(10)
Num_of_Label_wrong=torch.zeros(10)
Outlier=0

q=0
for x,y in test_loader:
    print(q)
    q+=1
    x=x.cuda()
    xc=cg(x)
    y=y.cuda()
    #print(y)
    k=int(y)
    w=mod(x)
    L=nce(x,w)
    #print(L)
    L1,L2=torch.sort(L)
    #print(L1)
    Poz=Po(L1[0])
    Pox=(1-Poz)/(1+Poz)
    if Pox>0.1:
        Outlier+=1
    tx, ty = Pi(L, Pox.item())
    l = len(ty)
    Num_of_Label[l]+=1
    j=0
    for i in range(l):
        if k==int(ty[i]):
            j=1
            break
    if j==1:
        Num_of_Label_correct[l]+=1
    else:
        Num_of_Label_wrong[l]+=1




print("There"+' '+"are"+' '+str(Outlier)+' '+"image(s)"+' '+"are"+' '+"given"+' '+"label"+' '+"Outlier")
for i in range(10):
    print("There"+' '+"are"+" "+str(int(Num_of_Label[i]))+" "+"image(s)"+" "+"are"+" "+"given"+" "+str(i)+" "+"label(s)")
    print(str(int(Num_of_Label_correct[i]))+" "+"of"+" "+"them"+" "+"have"+" "+"the"+" "+"correct"+" "+"label")
    print(str(int(Num_of_Label_wrong[i]))+" "+"of"+" "+"them"+" "+"don't"+" "+"have"+" "+"correct"+" "+"label")



