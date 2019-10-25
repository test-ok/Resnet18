import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import matplotlib.image as plimg
from main import ResNet18
from main import ResBlock


model = ResNet18(ResBlock)
model.load_state_dict(torch.load("./resnet18_10000_rate_0_001.pth"))
tensor = model.conv1[0].weight.cpu().data.clone()
reconv1 = nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=7,stride=2,padding=3,bias=False,output_padding=1)
reconv1.weight.data = tensor
reconv1.eval()

def imshow(img,channel):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)),cmap='Greys')
    #saving...
    # plt.savefig("./feature_reconv/feature_reconv1_channel_%d.png" % channel)
    plt.show()

def preprocess(output,i):


    out_temp = output.clone()
    if i !=0 and i != 63:
        out_temp[:,0:i,:,:] = 0
        out_temp[:,i+1:,:,:] = 0
    elif i == 63:
        out_temp[:,0:-1,:,:] = 0
    elif i == 0:
        out_temp[:,1:,:,:] = 0

    out_temp = F.relu(out_temp)
    out_temp = reconv1(out_temp,output_size=(32,32))
    out = out_temp.detach().data

    imshow(torchvision.utils.make_grid(out),i)

if __name__ == "__main__":
    
    transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))])
    testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=False,transform=transform_test)        
    testloader = torch.utils.data.DataLoader(testset,batch_size=1,shuffle=False,num_workers=2)
    
    for data in testloader:
        model.eval()
        images, labels = data
        with torch.no_grad():
            output = model.get_conv1_bn(images).cpu()
            for i in range(64):
                preprocess(output,i)

            out_temp = F.relu(output)
            out_temp = reconv1(output,output_size=(32,32))
            out = out_temp.detach().data
            imshow(torchvision.utils.make_grid(out),1001)
        imshow(torchvision.utils.make_grid(images),1000)
        print("finnished!")
        break # just read first image




