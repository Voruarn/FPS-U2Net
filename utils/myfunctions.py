import torch.nn as nn
import torch.nn.functional as F
import torch 

from torch.autograd import Variable

def _iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)


bce_loss = nn.BCELoss(size_average=True)
iou_loss = IOU(size_average=True)


def bce_iou_loss(pred,target):
	bce_out = bce_loss(pred,target)
	iou_out = iou_loss(pred,target)
	loss = bce_out + iou_out
	return loss


def multi_biou_loss(d0, d1, d2, d3, d4, d5, d6, mask):

    loss0 = bce_iou_loss(d0, mask)
    loss1 = bce_iou_loss(d1, mask)
    loss2 = bce_iou_loss(d2, mask)
    loss3 = bce_iou_loss(d3, mask)
    loss4 = bce_iou_loss(d4, mask)
    loss5 = bce_iou_loss(d5, mask)
    loss6 = bce_iou_loss(d6, mask)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return loss0, loss


# bce loss in original U2Netp
def multi_bce_loss(d0, d1, d2, d3, d4, d5, d6, labels_v):
    
	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	# print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

	return loss0, loss

# ------- 1. define loss function --------
def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def multi_ppa_loss(d0, d1, d2, d3, d4, d5, d6, mask):

    loss0 = structure_loss(d0, mask)

    loss1 = structure_loss(d1, mask)
    loss2 = structure_loss(d2, mask)
    loss3 = structure_loss(d3, mask)
    loss4 = structure_loss(d4, mask)
    loss5 = structure_loss(d5, mask)
    loss6 = structure_loss(d6, mask)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return loss0, loss
