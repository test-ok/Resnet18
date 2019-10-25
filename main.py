import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import argparse
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter


# use cuda or cpu
def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 100
NUM_EPOCHS = 20

transf = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
transform_train = transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))])
trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=False,transform=transform_test)
testloader = torch.utils.data.DataLoader(testset,batch_size=1,shuffle=False,num_workers=2)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

def conv3x3(input_channel,output_channel,stride):
    return nn.Conv2d(input_channel,output_channel,kernel_size=3,stride=stride,padding=1,bias=False)

class ResBlock(nn.Module):
    def __init__(self,input_channel,output_channel,stride,downSample=None):
        super(ResBlock,self).__init__()
        self.conv1 = conv3x3(input_channel,output_channel,stride)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(output_channel,output_channel,stride=1)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.downSample = downSample
        if(input_channel!=output_channel) or (stride!=1):
            self.downSample = nn.Sequential(nn.Conv2d(input_channel,output_channel,kernel_size=1,stride=stride,bias=False),nn.BatchNorm2d(output_channel))

    def forward(self,x):
        residual = x
        result = self.conv1(x)
        result = self.bn1(result)
        result = self.relu(result)
        result = self.conv2(result)
        result = self.bn2(result)
        if self.downSample is not None:
            residual = self.downSample(x)
        result = result + residual
        result = self.relu(result)
        return result

class ResNet18(nn.Module):
    def __init__(self,block):
        super(ResNet18,self).__init__()
        # self.input_channel = 32
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3,bias=False),
        nn.BatchNorm2d(64),nn.ReLU())
        self.maxpooling = nn.MaxPool2d(kernel_size=3,stride=2,padding=1,return_indices=True)
        self.conv2 = nn.Sequential(block(64,64,stride=1,downSample=None),block(64,64,stride=1,downSample=None))
        self.conv3 = nn.Sequential(block(64,128,stride=2),block(128,128,stride=1))
        self.conv4 = nn.Sequential(block(128,256,stride=2),block(256,256,stride=1))
        self.conv5 = nn.Sequential(block(256,512,stride=2),block(512,512,stride=1))
        self.avgpool = nn.AvgPool2d((1,1))
        self.fc = nn.Linear(512,10)
    def get_conv1_bn(self,x):
        output = self.conv1(x)
        return output
    def get_conv1(self,x):
        output = self.conv1[0](x)
        output = self.conv1[2](output)
        return output
    def get_conv5(self,x):
        temp = self.conv1(x)
        temp, _ = self.maxpooling(temp)
        temp = self.conv2(temp)
        temp = self.conv3(temp)
        temp = self.conv4(temp)
        output = self.conv5(temp)
        return output
    def get_indice(self,x):
        temp = self.conv1(x)
        temp, indice = self.maxpooling(temp)
        return indice
    def forward(self,x):
        result = self.conv1(x)
        result, _ = self.maxpooling(result)
        result = self.conv2(result)
        result = self.conv3(result)
        result = self.conv4(result)
        result = self.conv5(result)
        result = self.avgpool(result)
        result = result.view(result.size(0),-1)
        result = self.fc(result)
        return result


resnet = ResNet18(ResBlock)

if torch.cuda.is_available():
    resnet = resnet.cuda()
learn_rate = 0.001
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(),lr=learn_rate)

print("done")

if __name__ == '__main__':
    print("Start Training!")
    iteration = 0
    writer = SummaryWriter()
    for epoch in range(NUM_EPOCHS):
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, (images,labels) in enumerate(trainloader):
            length = len(trainloader)
            images = get_variable(images)
            labels = get_variable(labels)
            optimizer.zero_grad()
            outputs = resnet(images)

            loss = loss_function(outputs,labels)
            
            loss.backward()
            optimizer.step()
            iteration = i+1+epoch*length
            
            sum_loss = sum_loss + loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            correct = correct + predicted.eq(labels.data).cpu().sum()
            if(i+1) % 10 == 0:
                print("epoch [%d/%d], Iteration [%d], Loss: %.6f, Acc: %.6f%%" % (epoch+1, NUM_EPOCHS, (i+1+epoch*length), sum_loss/(i+1),100.*correct/total))
            writer.add_scalar('Loss/train', sum_loss/(i+1), iteration)

            writer.add_scalar('Accuracy/train', correct/total, iteration)
   
        print('Waiting Test!')
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                resnet.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = resnet(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                total = total + labels.size(0)
                correct = correct + (predicted == labels).sum()
            accuracy_test = 100 * correct / total
            print('correct rate: %.3f%%' % accuracy_test)
            writer.add_scalar('Accuracy/test', (accuracy_test * 0.01), iteration)
    writer.close()
    torch.save(resnet.state_dict(),'./resnet18_%d_lr_0_001.pth' %(iteration))





