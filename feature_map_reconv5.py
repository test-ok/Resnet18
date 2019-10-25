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
tensor_conv1 = model.state_dict()['conv1.0.weight'].cpu().data.clone()
tensor_conv2_0_v1 = model.state_dict()['conv2.0.conv1.weight'].cpu().data.clone()
tensor_conv2_0_v2 = model.state_dict()['conv2.0.conv2.weight'].cpu().data.clone()
tensor_conv2_1_v1 = model.state_dict()['conv2.1.conv1.weight'].cpu().data.clone()
tensor_conv2_1_v2 = model.state_dict()['conv2.0.conv2.weight'].cpu().data.clone()
tensor_conv3_0_v1 = model.state_dict()['conv3.0.conv1.weight'].cpu().data.clone()
tensor_conv3_0_v2 = model.state_dict()['conv3.0.conv2.weight'].cpu().data.clone()
tensor_conv3_0_downsample = model.state_dict()['conv3.0.downSample.0.weight'].cpu().data.clone()
tensor_conv3_1_v1 = model.state_dict()['conv3.1.conv1.weight'].cpu().data.clone()
tensor_conv3_1_v2 = model.state_dict()['conv3.1.conv2.weight'].cpu().data.clone()
tensor_conv4_0_v1 = model.state_dict()['conv4.0.conv1.weight'].cpu().data.clone()
tensor_conv4_0_v2 = model.state_dict()['conv4.0.conv2.weight'].cpu().data.clone()
tensor_conv4_0_downsample = model.state_dict()['conv4.0.downSample.0.weight'].cpu().data.clone()
tensor_conv4_1_v1 = model.state_dict()['conv4.1.conv1.weight'].cpu().data.clone()
tensor_conv4_1_v2 = model.state_dict()['conv4.1.conv2.weight'].cpu().data.clone()
tensor_conv5_0_v1 = model.state_dict()['conv5.0.conv1.weight'].cpu().data.clone()
tensor_conv5_0_v2 = model.state_dict()['conv5.0.conv2.weight'].cpu().data.clone()
tensor_conv5_0_downsample = model.state_dict()['conv5.0.downSample.0.weight'].cpu().data.clone()
tensor_conv5_1_v1 = model.state_dict()['conv5.1.conv1.weight'].cpu().data.clone()
tensor_conv5_1_v2 = model.state_dict()['conv5.1.conv2.weight'].cpu().data.clone()

reconv5_1_v2 = nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
reconv5_1_v1 = nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
reconv5_0_v2 = nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
reconv5_0_v1 = nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=3,stride=2,padding=1)
reconv5_0_downsample = nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=1,stride=2,output_padding=1)
reconv4_1_v2 = nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
reconv4_1_v1 = nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
reconv4_0_v2 = nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
reconv4_0_v1 = nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=3,stride=2,padding=1)
reconv4_0_downsample = nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=1,stride=2,output_padding=1)
reconv3_1_v2 = nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
reconv3_1_v1 = nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
reconv3_0_v2 = nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
reconv3_0_v1 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,stride=2,padding=1)
reconv3_0_downsample = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=1,stride=2,output_padding=1)
reconv2_1_v2 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
reconv2_1_v1 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
reconv2_0_v2 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
reconv2_0_v1 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
reconv1 = nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=7,stride=2,padding=3,bias=False,output_padding=1)

reconv5_1_v2.weight.data = tensor_conv5_1_v2
reconv5_1_v1.weight.data = tensor_conv5_1_v1
reconv5_0_v2.weight.data = tensor_conv5_0_v2
reconv5_0_v1.weight.data = tensor_conv5_0_v1
reconv5_0_downsample.weight.data = tensor_conv5_0_downsample
reconv4_1_v2.weight.data = tensor_conv4_1_v2
reconv4_1_v1.weight.data = tensor_conv4_1_v1
reconv4_0_v2.weight.data = tensor_conv4_0_v2
reconv4_0_v1.weight.data = tensor_conv4_0_v1
reconv4_0_downsample.weight.data = tensor_conv4_0_downsample
reconv3_1_v2.weight.data = tensor_conv3_1_v2
reconv3_1_v1.weight.data = tensor_conv3_1_v1
reconv3_0_v2.weight.data = tensor_conv3_0_v2
reconv3_0_v1.weight.data = tensor_conv3_0_v1
reconv3_0_downsample.weight.data = tensor_conv3_0_downsample
reconv2_1_v2.weight.data = tensor_conv2_1_v2
reconv2_1_v1.weight.data = tensor_conv2_1_v1
reconv2_0_v2.weight.data = tensor_conv2_0_v2
reconv2_0_v1.weight.data = tensor_conv2_0_v1
reconv1.weight.data = tensor_conv1

reconv5_1_v2.eval()
reconv5_1_v1.eval()
reconv5_0_v2.eval()
reconv5_0_v1.eval()
reconv5_0_downsample.eval()
reconv4_1_v2.eval()
reconv4_1_v1.eval()
reconv4_0_v2.eval()
reconv4_0_v1.eval()
reconv4_0_downsample.eval()
reconv3_1_v2.eval()
reconv3_1_v1.eval()
reconv3_0_v2.eval()
reconv3_0_v1.eval()
reconv3_0_downsample.eval()
reconv2_1_v2.eval()
reconv2_1_v1.eval()
reconv2_0_v2.eval()
reconv2_0_v1.eval()
reconv1.eval()

def imshow(img,channel):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)),cmap='Greys')
    # saving...
    # plt.savefig("./feature_reconv5/feature_reconv5_channel_%d.png" % channel)
    plt.show()

def preprocess(output,i):
    

    out_temp = output.clone()
    if i !=0 and i != 511:
        out_temp[:,0:i,:,:] = 0
        out_temp[:,i+1:,:,:] = 0
    elif i == 511:
        out_temp[:,0:-1,:,:] = 0
    elif i == 0:
        out_temp[:,1:,:,:] = 0
    # reconv5_x
    out_temp = F.relu(out_temp)
    out_temp_v5 = out_temp.clone()
    out_temp = reconv5_1_v2(out_temp,output_size=(1,1))
    out_temp = reconv5_1_v1(out_temp,output_size=(1,1))
    out_temp = reconv5_0_v2(out_temp,output_size=(1,1))
    out_temp = reconv5_0_v1(out_temp,output_size=(2,2))
    out_temp_v5 = reconv5_0_downsample(out_temp_v5)
    out_temp = out_temp + out_temp_v5
    #reconv4_x
    out_temp = F.relu(out_temp)
    out_temp_v4 = out_temp.clone()
    out_temp = reconv4_1_v2(out_temp,output_size=(2,2))
    out_temp = reconv4_1_v1(out_temp,output_size=(2,2))
    out_temp = reconv4_0_v2(out_temp,output_size=(2,2))
    out_temp = reconv4_0_v1(out_temp,output_size=(4,4))
    out_temp_v4 = reconv4_0_downsample(out_temp_v4)
    out_temp = out_temp + out_temp_v4
    #reconv3_x
    out_temp = F.relu(out_temp)
    out_temp_v3 = out_temp.clone()
    out_temp = reconv3_1_v2(out_temp,output_size=(4,4))
    out_temp = reconv3_1_v1(out_temp,output_size=(4,4))
    out_temp = reconv3_0_v2(out_temp,output_size=(4,4))
    out_temp = reconv3_0_v1(out_temp,output_size=(8,8))
    out_temp_v3 = reconv3_0_downsample(out_temp_v3)
    out_temp = out_temp + out_temp_v3
    #reconv2_x
    out_temp = F.relu(out_temp)
    out_temp_v2 = out_temp.clone()
    out_temp = reconv2_1_v2(out_temp,output_size=(8,8))
    out_temp = reconv2_1_v1(out_temp,output_size=(8,8))
    out_temp = reconv2_0_v2(out_temp,output_size=(8,8))
    out_temp = reconv2_0_v1(out_temp,output_size=(8,8))
    out_temp = out_temp + out_temp_v2
    return out_temp


if __name__ == "__main__":
    transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))])
    testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=False,transform=transform_test)        
    testloader = torch.utils.data.DataLoader(testset,batch_size=1,shuffle=False,num_workers=2)
    
    for data in testloader:
        model.eval()
        images, labels = data
        with torch.no_grad():
            indice = model.get_indice(images)
            initial = model.get_conv5(images).cpu()
            unpool = nn.MaxUnpool2d(kernel_size=3,stride=2,padding=1)
            
            for i in range(512):
                out_temp = preprocess(initial,i)
                out_temp = unpool(out_temp,indices=indice,output_size=(16,16))
                out_temp = F.relu(out_temp) 
                out_temp = reconv1(out_temp,output_size=(32,32))
                out = out_temp.detach().data
                imshow(torchvision.utils.make_grid(out),i)
        break # just test one image


    