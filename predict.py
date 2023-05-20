import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms#, utils
import numpy as np
from PIL import Image

from torch.utils import data

from model import U2NET, U2NETP
from model import FPSU2Net, FPSU2NetV2, FPSU2NetV3, FPSU2NetV4
from dataset import ext_transforms as et
from dataset.LC8_FP_SEG_Testset import LC8FPSegDataset
import argparse
from tqdm import tqdm
import collections

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir,size=(512,512)):
    resolution=list(size)
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    imo = im.resize((resolution[0], resolution[1]),resample=Image.BILINEAR)
    pb_np = np.array(imo)
    imo.save(d_dir+image_name+'.png')

def get_argparser():
    parser = argparse.ArgumentParser()
 
    # Datset Options
    parser.add_argument("--prediction_dir", type=str, 
        default='./preds/',
        help="path to Dataset")
    
    parser.add_argument("--testset_path", type=str, 
        default='D:/2023_Files/Dataset/LC8_FPS1024_AugV3/Test',
        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='LC8FPS',
                        choices=['LC8FPS'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: None)")
    parser.add_argument("--heads", type=int, default=8,
                        help="heads of BoT MHSA")

    parser.add_argument("--model", type=str, default='FPSU2Net',
            help='range:[u2net, u2netp, FPSU2Net, FPSU2NetV2, FPSU2NetV3, FPSU2NetV4 ]')
    
    parser.add_argument("--threshold", type=float, default=0.6,
                     help='threshold to predict foreground')
    
    parser.add_argument("--batch_size", type=int, default=1,
                        help='batch size (default: 16)')
    parser.add_argument("--crop_val", action='store_true', default=True,
                        help='crop validation (default: False)')
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--n_cpu", type=int, default=1,
                        help="download datasets")
    parser.add_argument("--ckpt", type=str,
    default=None, help="restore from checkpoint")

    return parser

def get_dataset(opts):
    if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                # et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    val_dst = LC8FPSegDataset(is_train=False,voc_dir=opts.testset_path,
                            transform=val_transform)
    return val_dst

def main():
    torch.cuda.empty_cache()

    opts = get_argparser().parse_args()

    test_dst = get_dataset(opts)
    
    opts.prediction_dir='./LC8FPS_preds/'+opts.model+'/'
    print('opts:',opts)

    test_loader = data.DataLoader(
        test_dst, batch_size=opts.batch_size, shuffle=False, num_workers=opts.n_cpu)
    print("Dataset: %s, Test set: %d" %
            (opts.dataset, len(test_dst)))
   
    # --------- 3. model define ---------
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
   
    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
    try:
        net.load_state_dict(checkpoint["model_state"])
        print('try: load pth from:', opts.ckpt)
    except:
        dic = collections.OrderedDict()
        for k, v in checkpoint["model_state"].items():
            #print( k)
            mlen=len('module')+1
            newk=k[mlen:]
            # print(newk)
            dic[newk]=v
        net.load_state_dict(dic)
        print('except: load pth from:', opts.ckpt)

    net=net.cuda()
    net.eval()
    # --------- 4. inference for each image ---------
    for data_test in tqdm(test_loader):
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        inputs_test = Variable(inputs_test).cuda()

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)
  
        # save results to test_results folder
        if not os.path.exists(opts.prediction_dir):
            os.makedirs(opts.prediction_dir, exist_ok=True)
        save_output(data_test['img_name'][0], pred, opts.prediction_dir, 
                    size=(opts.input_size, opts.input_size))

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
