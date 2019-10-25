import torch
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from main import ResNet18
from main import ResBlock

model = ResNet18(ResBlock)
model.load_state_dict(torch.load("./resnet18_10000_rate_0_001.pth"))

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)),cmap='Greys')
    plt.show()

if __name__=='__main__':
    tensor = model.conv1[0].weight.cpu().data.clone()
    transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))])
    testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=False,transform=transform_test)        
    testloader = torch.utils.data.DataLoader(testset,batch_size=1,shuffle=False,num_workers=2)
    for data in testloader:
        model.eval()
        images, labels = data
        feature_conv5 = model.get_conv5(images).cpu()
        feature_conv1 = model.get_conv1(images).cpu()
        feature_conv1_bn = model.get_conv1_bn(images).cpu()
        feature_conv1 = feature_conv1.permute(1,0,2,3)
        feature_conv1_bn = feature_conv1_bn.permute(1,0,2,3)
        feature_conv5 = feature_conv5.permute(1,0,2,3)

        Map1 = torchvision.utils.make_grid(feature_conv1,nrow=8,padding=2,normalize=False)
        Map2 = torchvision.utils.make_grid(feature_conv1_bn,nrow=8,padding=2,normalize=False)
        img1 = Map1.data.clone()
        img2 = Map2.data.clone()

        nrow = 8
        padding = 1
        rows = np.min((feature_conv5.shape[0] // nrow + 1, 64))    
        grid = torchvision.utils.make_grid(feature_conv5.data, nrow=nrow, normalize=True, padding=padding)  
        plt.figure( figsize=(nrow,rows) )
        plt.imshow(grid.numpy().transpose((1, 2, 0)), cmap = plt.get_cmap('gray_r'))
        plt.show() 
        # Map3 = torchvision.utils.make_grid(feature_conv5,nrow=8,padding=2,normalize=False)
        # img3 = Map3.data.clone()
        imshow(img1)
        imshow(img2)
        # imshow(img3)
        imshow(torchvision.utils.make_grid(images))
        break # just test one image