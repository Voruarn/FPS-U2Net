import matplotlib.pyplot as plt
import os
import torch
import torchvision
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import argparse
import sys


class ActiveFireDataset(torch.utils.data.Dataset):
    """一个用于加载 ActiveFire 数据集的自定义数据集"""

    def __init__(self, is_train, voc_dir, Btype='B766', trainsize=256):
        self.trainsize=trainsize
        
        IMG_dir='Images'+Btype
        PNGImages=voc_dir+'/'+IMG_dir

        images=[i.split('.')[0] for i in os.listdir(PNGImages)]

        features, labels=[], []
        for fname in tqdm(images):
            if fname=='':
                continue
            features.append(voc_dir+'/'+IMG_dir+'/'+f'{fname}.png')
            labels.append(voc_dir+'/Masks/'+f'{fname}.png')
        
        self.features=features
        self.labels=labels

        self.img_transform = transforms.Compose([
            # transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.gt_transform = transforms.Compose([
            # transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        
        mode='training'
        if not is_train:
            mode='test'
        print('read ' + str(len(self.features)) + ' examples for '+mode)

    def __getitem__(self, idx):
        image=Image.open(self.features[idx]).convert('RGB')
        gt=Image.open(self.labels[idx]).convert('L')

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        return image, gt
       
    def __len__(self):
        return len(self.features)
    


# def get_argparser():
#     parser = argparse.ArgumentParser()
    
#     # Datset Options
#     parser.add_argument("--trainset_path", type=str, 
#         default='D:/2023_Files/ActiveFire/AFD_OceB766/Train',
#         help="path to Dataset")
#     parser.add_argument("--testset_path", type=str, 
#         default='D:/2023_Files/ActiveFire/AFD_OceB766/Train',
#         help="path to Dataset")

#     parser.add_argument("--dataset", type=str, default='AFDOce',
#         choices=['AFDOce'], help='Name of dataset')

#     parser.add_argument("--batch_size", type=int, default=2,
#                         help='batch size (default: 16)')
#     parser.add_argument("--n_cpu", type=int, default=2,
#                         help="download datasets")
#     return parser


# def get_dataset(opts):
#     train_dst = ActiveFireDataset(is_train=True,voc_dir=opts.trainset_path)
#     val_dst = ActiveFireDataset(is_train=False,voc_dir=opts.testset_path)
#     return train_dst, val_dst


# if __name__=='__main__':
    
#     opts = get_argparser().parse_args()
#     print(opts)
#     train_dst, val_dst = get_dataset(opts)
#     train_loader = torch.utils.data.DataLoader(
#         train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.n_cpu,
#         drop_last=True)  # drop_last=True to ignore single-image batches.
       
#     data_loader = tqdm(train_loader, file=sys.stdout)
    
#     for (images, gt) in data_loader:
#         print('images.shape:',images.shape)
#         print('gt.shape:',gt.shape)

#         print(gt[gt>0])
#         break











