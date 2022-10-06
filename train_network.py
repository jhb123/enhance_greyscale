#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 17:42:00 2022

@author: josephbriggs
"""
import math
from pathlib import Path
import torch
from torchvision.io import read_image
import torchvision.transforms as transforms

from torch.utils.data import Dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt

#%%
class imageDataset(Dataset):
    def __init__(self, hr_img_dir, lr_img_dir):
        self.hr_img_dir = Path(hr_img_dir)
        self.lr_img_dir = Path(lr_img_dir)

    def __len__(self):
        return len(self.hr_img_dir.glob('*.png'))

    def __getitem__(self, idx):
        fname = f"{str(idx):0>7}.png"
        hr_image = read_image(str(self.hr_img_dir.joinpath(fname)))
        lr_image = read_image(str(self.lr_img_dir.joinpath(fname)))
        return hr_image,lr_image

def show_image(image):
    image_np = image.numpy()    
    cv2.imshow("image",np.transpose(image_np,axes = [2,1,0]))
    cv2.waitKey()

def train_network():
    '''


    Returns
    -------
    None.

    '''


def test_network():
    '''


    Returns
    -------
    None.

    '''

def export_network():
    '''


    Returns
    -------
    None.

    '''


def main():
    
    images = imageDataset('gs_imgs_hr_x2','gs_imgs_lr_x2')
    
    imgs_to_show = 3
    fig,ax = plt.subplots(2,imgs_to_show)
    ax = ax.ravel()
    for i in range(imgs_to_show):
        hr_img,lr_img = images[i]
        hr_img = hr_img.numpy()
        lr_img = lr_img.numpy()
        ax[i].imshow(hr_img[0,:,:], vmin=0, vmax=255,cmap = 'gray')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i+imgs_to_show].imshow(lr_img[0,:,:], vmin=0, vmax=255,cmap = 'gray')
        ax[i+imgs_to_show].set_xticks([])
        ax[i+imgs_to_show].set_yticks([])
if __name__ == "__main__":
    main()