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
from model.u2net import U2NET, U2NETP
from model.fpsu2net import FPSU2Net
from dataset import ext_transforms as et
from dataset.LC8_FP_SEG_V4 import LC8FPSegDataset
from tqdm import tqdm

from metrics import StreamSegMetrics
from torch.utils.tensorboard import SummaryWriter
import random
from utils.myfunctions import multi_bce_loss, multi_biou_loss


def get_argparser():
    parser = argparse.ArgumentParser()
    
    # Datset Options
    parser.add_argument("--trainset_path", type=str, 
        default='../Train',
        help="path to Dataset")
    parser.add_argument("--testset_path", type=str, 
        default='../Test',
        help="path to Dataset")
    
    parser.add_argument("--dataset", type=str, default='LC8FPS', help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: None)")
   
    parser.add_argument("--model", type=str, default='FPSU2Net',
            help='range:[ FPSU2Net]')

    parser.add_argument("--epochs", type=int, default=60,
                        help="epoch number ")

    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10)
    
    parser.add_argument("--batch_size", type=int, default=8,
                        help='batch size (default: 16)')
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--n_cpu", type=int, default=4,
                        help="download datasets")

    parser.add_argument("--ckpt", type=str,
        default=None, 
        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=True)
    # default='cross_entropy',
    parser.add_argument("--loss_type", type=str, default='bi',
                        help="loss_type:[bce, bi]")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--val_interval", type=int, default=5,
                        help="epoch interval for eval (default: 100)")

    return parser


def get_dataset(opts):

    train_transform = et.ExtCompose([
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
 
    val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    train_dst = LC8FPSegDataset(is_train=True,voc_dir=opts.trainset_path, 
                                transform=train_transform)
    val_dst = LC8FPSegDataset(is_train=False,voc_dir=opts.testset_path,
                            transform=val_transform)
    return train_dst, val_dst


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
        val_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.n_cpu)
    print("Dataset: %s, Train set: %d, Val set: %d" %
            (opts.dataset, len(train_dst), len(val_dst)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def save_ckpt(path):
        torch.save({
            "epoch": epoch+1,
            "model_state": net.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, path)
        print("Model saved as %s" % path)


    net= eval(opts.model)(3,1)
 
    optimizer = optim.Adam(net.parameters(), lr=opts.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=opts.weight_decay)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    if opts.loss_type=='bce':
        criterion=multi_bce_loss
    elif opts.loss_type=='bi':
        criterion=multi_biou_loss

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
            
            inputs, labels = data
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
            
            labels_v=labels_v.unsqueeze(dim=1)
            loss0, loss=criterion(d0, d1, d2, d3, d4, d5, d6, labels_v)

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
            save_ckpt('CHKP/latest_{}_{}_{}.pth'.format(opts.model, 
                                opts.dataset, opts.loss_type))
           
