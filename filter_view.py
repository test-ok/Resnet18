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

if __name__=='__main__':
    tensor = model.conv1[0].weight.cpu().data.clone()
    nrow = 8
    padding = 1
    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)  
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)), cmap = plt.get_cmap('gray_r'))
    plt.show() 