from tqdm import tqdm
import utils
import os
import random
import argparse
import numpy as np
import sys

from torch.utils import data
from dataset.ActiveFireDataset import ActiveFireDataset
from metrics import StreamSegMetrics
from model.FPSU2Net import FPSU2Net
import torch.nn.functional as F
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
import pytorch_iou


def get_argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--trainset_path", type=str, 
        default='..',
        help="path to Dataset")
    parser.add_argument("--testset_path", type=str, 
        default='..',
        help="path to Dataset")
    
    parser.add_argument("--dataset", type=str, default='AFDOce', help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=2,
                        help='num_classes')
    parser.add_argument("--threshold", type=float, default=0.5,
                     help='threshold to predict foreground')

    parser.add_argument("--model", type=str, default='FPSU2Net',
        help='model name:[FPSU2Net]')
    parser.add_argument("--epochs", type=int, default=80,
                        help="epoch number (default: 60)")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="total_itrs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10)
  
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size ')
    parser.add_argument("--trainsize", type=int, default=256)

    parser.add_argument("--n_cpu", type=int, default=2,
                        help="download datasets")
    
    parser.add_argument("--ckpt", type=str,
            default=None, help="restore from checkpoint")
    parser.add_argument("--loss_type", type=str, default='bi', 
                        help="loss type:[bce, bi]")
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--val_interval", type=int, default=5,
                        help="epoch interval for eval (default: 100)")

    return parser


def get_dataset(opts):
    train_dst = ActiveFireDataset(is_train=True,voc_dir=opts.trainset_path)
    val_dst = ActiveFireDataset(is_train=False,voc_dir=opts.testset_path)
    return train_dst, val_dst

def validate(opts, model, loader, device, metrics):
    metrics.reset()
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            s1,s2,s3,s4,s5,s6, s1_sig,s2_sig,s3_sig,s4_sig,s5_sig,s6_sig= model(images)
              
            outputs=s1_sig
            outputs=outputs.squeeze()
            outputs[outputs>=opts.threshold]=1
            outputs[outputs<opts.threshold]=0
            
            preds = outputs.detach().cpu().numpy()
            
            preds=preds.astype(int)
            targets = labels.cpu().numpy()
            if targets.shape[0]==1:
                targets=targets.squeeze()
           
            metrics.update(targets, preds)   

        score = metrics.get_results()
    return score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CE = torch.nn.BCEWithLogitsLoss()
MSE = torch.nn.MSELoss()
IOU = pytorch_iou.IOU(size_average = True)



def main():
    if not os.path.exists('CHKP'):
        utils.mkdir('CHKP')

    opts = get_argparser().parse_args()

    tb_writer = SummaryWriter()
    
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    
    print("Device: %s" % device)

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst = get_dataset(opts)
    opts.total_itrs=opts.epochs * (len(train_dst) // opts.batch_size)
    print('opts:',opts)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.n_cpu,
        drop_last=True)  
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.n_cpu)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))


    model = eval(opts.model)()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=opts.weight_decay)
    metrics = StreamSegMetrics(opts.num_classes)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    def save_ckpt(path):
        torch.save({
            "epoch": epoch+1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, path)
        print("Model saved as %s" % path)  
        
    cur_epoch=0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model=model.to(device)
        
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_epoch = checkpoint["epoch"]   
        
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model=model.to(device)


    for epoch in range(cur_epoch,opts.epochs):
        model.train()
        cur_itrs=0
        data_loader = tqdm(train_loader, file=sys.stdout)
        running_loss = 0.0
        
        for (images, gts) in data_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            gts = gts.to(device, dtype=torch.float32)

            optimizer.zero_grad()
           
            s1,s2,s3,s4,s5,s6,s1_sig,s2_sig,s3_sig,s4_sig,s5_sig,s6_sig= model(images)
            
            loss1 = CE(s1, gts) + IOU(s1_sig, gts)
            loss2 = CE(s2, gts) + IOU(s2_sig, gts)
            loss3 = CE(s3, gts) + IOU(s3_sig, gts)
            loss4 = CE(s4, gts) + IOU(s4_sig, gts)
            loss5 = CE(s5, gts) + IOU(s5_sig, gts)
            loss6 = CE(s6, gts) + IOU(s6_sig, gts)
            
            total_loss = loss1 + loss2/2 + loss3/4 +loss4/8 + loss5/16 + loss6/32
            
            running_loss += total_loss.data.item()

            total_loss.backward()
            optimizer.step()

            data_loader.desc = "Epoch {}/{}, loss={:.4f}".format(epoch, opts.epochs, running_loss/cur_itrs)
            
            scheduler.step()

        if (epoch+1) % opts.val_interval == 0:
            save_ckpt('CHKP/latest_{}_{}.pth'.format(opts.model, opts.dataset))
    
        print("validation...")
        model.eval()
        val_score = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
        
        print('val_score:',val_score)
    
        tags = ["train_loss", "learning_rate","Mean_Acc","Mean_IoU","Fire_IoU"]

        tb_writer.add_scalar(tags[0], (running_loss/cur_itrs), epoch)
        tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[2], val_score['Mean Acc'], epoch)
        tb_writer.add_scalar(tags[3], val_score['Mean IoU'], epoch)
        tb_writer.add_scalar(tags[4], val_score['Class IoU'][1], epoch)


if __name__ == '__main__':
    main()
