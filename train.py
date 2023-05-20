import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
import utils
import sys
import numpy as np
import argparse

from model import U2NET, U2NETP
from model import FPSU2Net, FPSU2NetV2, FPSU2NetV3, FPSU2NetV4
from dataset import ext_transforms as et
from dataset.LC8_FP_SEG_V4 import LC8FPSegDataset
from tqdm import tqdm

from metrics import StreamSegMetrics
from torch.utils.tensorboard import SummaryWriter
import random

def get_argparser():
    parser = argparse.ArgumentParser()
    
    # Datset Options
    parser.add_argument("--trainset_path", type=str, 
        default='D:/2023_Files/Dataset/LC8_FPS512_AugV3/Train',
        help="path to Dataset")
    parser.add_argument("--testset_path", type=str, 
        default='D:/2023_Files/Dataset/LC8_FPS512_AugV3/Test',
        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='LC8FPS', help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: None)")
    parser.add_argument("--heads", type=int, default=8,
                        help="heads of BoT MHSA")
   
    parser.add_argument("--model", type=str, default='FPSU2Net',
            help='range:[u2net, u2netp, FPSU2Net, FPSU2NetV2, FPSU2NetV3, FPSU2NetV4 ]')
    
    parser.add_argument("--threshold", type=float, default=0.6,
                     help='threshold to predict foreground')

    # Train Options    
    parser.add_argument("--epochs", type=int, default=150,
                        help="epoch number (default: 30k)")

    parser.add_argument("--total_itrs", type=int, default=150,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')

    parser.add_argument("--batch_size", type=int, default=8,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=8,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=512)
    # RandomCrop
    parser.add_argument("--random_crop_size", type=int, default=512)

    parser.add_argument("--n_cpu", type=int, default=8,
                        help="download datasets")

    parser.add_argument("--ckpt", type=str,
        default=None, 
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=True)
    # default='cross_entropy',
   
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    #  default=100,
    parser.add_argument("--val_interval", type=int, default=5,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    return parser

def get_dataset(opts):
    # Dataset And Augmentation
    
    train_transform = et.ExtCompose([
        # et.ExtResize(size=opts.crop_size),
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(opts.random_crop_size, opts.random_crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    if opts.crop_val:
        val_transform = et.ExtCompose([
            # et.ExtResize(opts.crop_size),
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            # et.ExtResize(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    train_dst = LC8FPSegDataset(is_train=True,voc_dir=opts.trainset_path, 
                                transform=train_transform)
    val_dst = LC8FPSegDataset(is_train=False,voc_dir=opts.testset_path,
                            transform=val_transform)
    return train_dst, val_dst

def validate(opts, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
   
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader)):
            images, labels = data['image'], data['label']

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            d0, d1, d2, d3, d4, d5, d6 = model(images)
            outputs = d0
            
            outputs=outputs.squeeze()
            outputs[outputs>=opts.threshold]=1
            outputs[outputs<opts.threshold]=0

            preds = outputs.detach().cpu().numpy()
            
            preds=preds.astype(int)
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)   
            
        score = metrics.get_results()
    return score

bce_loss = nn.BCELoss(size_average=True)

# bce loss in original U2Netp
def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    
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

def muti_ppa_loss(d0, d1, d2, d3, d4, d5, d6, mask):
    mask=mask.unsqueeze(dim=1)

    loss0 = structure_loss(d0, mask)

    loss1 = structure_loss(d1, mask)
    loss2 = structure_loss(d2, mask)
    loss3 = structure_loss(d3, mask)
    loss4 = structure_loss(d4, mask)
    loss5 = structure_loss(d5, mask)
    loss6 = structure_loss(d6, mask)

    loss   = loss0+loss1/2+loss2/4+loss3/8+loss4/16+loss5/32+loss6/64
    return loss0, loss

if __name__=="__main__":
    if not os.path.exists('./CHKP/'):
        os.makedirs('./CHKP/')

    tb_writer = SummaryWriter()

    opts = get_argparser().parse_args()
    
    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst = get_dataset(opts)
    opts.total_itrs=opts.epochs * (len(train_dst) // opts.batch_size)
    print('opts:',opts)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.n_cpu,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=opts.n_cpu)
    print("Dataset: %s, Train set: %d, Val set: %d" %
            (opts.dataset, len(train_dst), len(val_dst)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "epoch": epoch+1,
            "model_state": net.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, path)
        print("Model saved as %s" % path)

    # ------- 3. define model --------
    if(opts.model=='u2net'):
        net = U2NET(3, 1)
    elif(opts.model=='u2netp'):
        net = U2NETP(3,1)
    elif opts.model=='FPSU2Net':
        net= FPSU2Net(3,1,(opts.crop_size, opts.crop_size), heads=opts.heads)
    elif opts.model=='FPSU2NetV2':
        net= FPSU2NetV2(3,1,(opts.crop_size, opts.crop_size), heads=opts.heads)
    elif opts.model=='FPSU2NetV3':
        net= FPSU2NetV3(3,1,(opts.crop_size, opts.crop_size), heads=opts.heads)
    elif opts.model=='FPSU2NetV4':
        net= FPSU2NetV4(3,1,(opts.crop_size, opts.crop_size), heads=opts.heads)
   

    if torch.cuda.is_available():
        net.cuda()

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=opts.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    cur_epoch=0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint["model_state"])
        net = nn.DataParallel(net)
        net.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_epoch = checkpoint["epoch"]
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        net = nn.DataParallel(net)
        net.to(device)
  
    metrics = StreamSegMetrics(opts.num_classes)

    # ------- 5. training process --------
    print("---start training...")

    for epoch in range(cur_epoch, opts.epochs):
        net.train()

        data_loader = tqdm(train_loader, file=sys.stdout)
        cur_itrs=0
        running_loss = 0.0
        running_tar_loss = 0.0

        for data in data_loader:
            cur_itrs += 1
            
            inputs, labels = data['image'], data['label']
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
            
            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            
            loss0, loss=muti_ppa_loss(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss0.data.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss0, loss
            
            data_loader.desc = "Epoch {}/{}, train_loss={:.4f}, tar_loss={:.4f}".format(epoch, opts.epochs,
                                    running_loss / cur_itrs, running_tar_loss / cur_itrs)
            scheduler.step()
            
       
        if (epoch+1) % opts.val_interval == 0:
            save_ckpt('CHKP/{}_{}_ep{}.pth'.format(opts.model, 
                                opts.dataset,  epoch+1))
                
        print("validation...")
        net.eval()
        val_score = validate(
            opts=opts, model=net, loader=val_loader, device=device, metrics=metrics)

        print('val_score:',val_score)

        tags = ["train_loss", "learning_rate","Mean_Acc","Mean_IoU","Fire_IoU"]

        tb_writer.add_scalar(tags[0], (running_loss/cur_itrs), epoch)
        tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[2], val_score['Mean Acc'], epoch)
        tb_writer.add_scalar(tags[3], val_score['Mean IoU'], epoch)
        tb_writer.add_scalar(tags[4], val_score['Class IoU'][1], epoch)
