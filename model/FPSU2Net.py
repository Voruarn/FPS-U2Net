import torch
import torch.nn as nn
import torch.nn.functional as F
from model.u2net import *
from model.modules import *


## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.interpolate(src,size=tar.shape[2:],mode='bilinear')

    return src



### FPSU2Net ###
class FPSU2Net(nn.Module):

    def __init__(self,in_ch=3,out_ch=1, **kwargs):
        super(FPSU2Net,self).__init__()
      
        self.stage1 = RSU7(in_ch,16,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,16,64)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(64,16,64)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(64,16,64)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(64,16,64)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.ma1=MAM1(64,64)
        self.ma2=MAM(64,64)
        self.ma3=MAM(64,64)
        self.ma4=MAM(64,64)
        self.ma5=MAM5(64,64)

        self.stage6 = RSU4F(64,16,64)
        
        # decoder
        self.stage5d = RSU4F(128,16,64)
        self.stage4d = RSU4(128,16,64)
        self.stage3d = RSU5(128,16,64)
        self.stage2d = RSU6(128,16,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(64,out_ch,3,padding=1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx) # RSU7
        hx = self.pool12(hx1)
      
        #stage 2
        hx2 = self.stage2(hx) # RSU6
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx) # RSU5
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx) # RSU4
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx) # RSU4F
        hx = self.pool56(hx5)
 
        #stage 6
        hx6 = self.stage6(hx) # Bottleneck
        hx6up = _upsample_like(hx6,hx5)

        #decoder
        c5=self.ma5(hx4, hx5)
        hx5d = self.stage5d(torch.cat((hx6up,c5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        c4=self.ma4(hx3, hx4, hx5)
        hx4d = self.stage4d(torch.cat((hx5dup,c4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        c3=self.ma3(hx2,hx3,hx4)
        hx3d = self.stage3d(torch.cat((hx4dup,c3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        c2=self.ma2(hx1,hx2,hx3)
        hx2d = self.stage2d(torch.cat((hx3dup,c2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        c1=self.ma1(hx1,hx2)
        hx1d = self.stage1d(torch.cat((hx2dup,c1),1))

        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        return d1,d2,d3,d4,d5,d6, torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)




if __name__=="__main__":
    print('test FPSU2NetMR!')
   
 