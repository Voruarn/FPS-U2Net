import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms#, utils

from model.FPSU2Net import FPSU2Net

if __name__ == "__main__":
    print('test FPSU2Net!')
    torch.cuda.empty_cache()
   
    input=torch.rand((1,3,256,256)).cuda()
    model = FPSU2Net(3,1).cuda()
    print('input.shape:',input.shape)
    
   
    output=model(input)
    
    i=0
    for opt in output:
        print('d{}.shape:'.format(i),opt.shape)
        i+=1
