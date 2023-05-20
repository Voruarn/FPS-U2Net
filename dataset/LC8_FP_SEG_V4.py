import matplotlib.pyplot as plt
import os
import torch
import torchvision
from d2l import torch as d2l
from tqdm import tqdm
import numpy as np
from PIL import Image
import dataset.ext_transforms as et
# import ext_transforms as et
import argparse
import torch
import random


#Pascal VOC2012语义分割数据集
"""
D:/2023_Files/Dataset/LC8_FP_SEG_S512_Aug_V2/Train
"""
voc_dirV2='D:/2023_Files/Dataset/LC8_FP_SEG_S512_Aug_V2/Train'

def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注"""
    #将所有输入的图像和标签读入内存
    PNGImages=voc_dir+'/PNGImages'

    images=[i.split('.')[0] for i in os.listdir(PNGImages)]

    mode=torchvision.io.image.ImageReadMode.RGB
 
    features, labels=[], []
    for fname in tqdm(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'PNGImages',f'{fname}.png')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass', f'{fname}.png'), mode))
    return features, labels

#列举RGB颜色值和类名

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0]]

#@save
VOC_CLASSES = ['background', 'fire']

def voc_colormap2label():#构建RGB->VOC类别的索引
    """构建从RGB到VOC类别索引的映射"""
    colormap2label=torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    #获取colormap在colormap2label的value
    """将VOC标签中的RGB值映射到它们的类别索引"""
    colormap=colormap.permute(1, 2, 0).numpy().astype('int32')
    idx=((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
         + colormap[:, :, 2])
    return colormap2label[idx]

#预处理数据
def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    rect=torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature=torchvision.transforms.functional.crop(feature, *rect)
    label=torchvision.transforms.functional.crop(label, *rect)
    return feature, label


def label2image(pred):
    # pred: [320,480]
    colormap = torch.tensor(VOC_COLORMAP,dtype=int)
    x = pred.long()
    return (colormap[x,:]).data.cpu().numpy()

def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


#自定义语义分割数据集类
class LC8FPSegDataset(torch.utils.data.Dataset):
    """一个用于加载LC8_FP_SEG数据集的自定义数据集"""
    cmap = voc_cmap()
    def __init__(self, is_train, voc_dir, transform=None):
        pass
        self.transform=transform

        PNGImages=voc_dir+'/PNGImages'

        images=[i.split('.')[0] for i in os.listdir(PNGImages)]

        features, labels=[], []
        for fname in tqdm(images):
            if fname=='':
                continue
            features.append(voc_dir+'/PNGImages/'+f'{fname}.png')
            labels.append(voc_dir+'/SegmentationClass/'+f'{fname}.png')
        
        self.features=features
        self.labels=labels

        self.colormap2label=voc_colormap2label()
        mode='training'
        if not is_train:
            mode='test'
        print('read ' + str(len(self.features)) + ' examples for '+mode)

    def __getitem__(self, idx):
        img=Image.open(self.features[idx]).convert('RGB')
        target=Image.open(self.labels[idx])

        # img.show()

        if self.transform is not None:
            img, target = self.transform(img, target)

        target=target.permute(2,0,1)
        target=voc_label_indices(target, self.colormap2label)

        # print(img.shape)
        # print(target.shape)
        sample = {'image':img, 'label':target}
        return sample
       
    
    def __len__(self):
        return len(self.features)
    
    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]
