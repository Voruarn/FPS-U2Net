import matplotlib.pyplot as plt
import os
import torch
import torchvision
from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
import torch
import torchvision.transforms as transforms

class LC8FPSDataset(torch.utils.data.Dataset):
    """一个用于加载 LC8FPSDataset  测试集的自定义数据集"""

    def __init__(self, is_train, voc_dir,  trainsize=256):
        self.trainsize=trainsize

        IMG_dir='Images'
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
        self.names=images

        self.img_transform = transforms.Compose([
            #transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.gt_transform = transforms.Compose([
            #transforms.Resize((self.trainsize, self.trainsize)),
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

        return {'img':image, 'label':gt, 'name':self.names[idx]}
       
    def __len__(self):
        return len(self.features)
    










