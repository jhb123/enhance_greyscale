#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:56:55 2022

@author: josephbriggs
"""
from pathlib import Path
import glob
import pathlib
import argparse
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image,write_png

def image_data(img_dir,res):
    stride = 17
    
    path = Path(img_dir)
    for image_path in path.glob('*.png'):
        image = read_image(str(image_path))
        
        crop_width = image.shape[1] % (stride*res)
        crop_height = image.shape[2] % (stride*res)
        
        image = transforms.functional.crop(image,0,0,
                                           image.shape[1]-crop_width,
                                           image.shape[2]-crop_height)
        
        hr_image = transforms.Grayscale()(image)
        
        width = hr_image.shape[1]
        height = hr_image.shape[2]
        
        width_lr = int(width/res)
        height_lr = int(height/res)
        
        LR_transfrom = torch.nn.Sequential(
        transforms.Grayscale(),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=1),
        transforms.Resize((width_lr,height_lr)))
        
        
        lr_image = LR_transfrom(image)
        
        for i in range(0,width_lr,stride):
            for j in range(0,height_lr,stride):
                crop_hr = hr_image[:,i*res:i*res+stride*res,
                                   j*res:j*res+stride*res]            
                crop_lr = lr_image[:,i:i+stride,j:j+stride]       

                yield crop_hr,crop_lr



def main():
    '''
    Converts files to greyscale.
    '''

    parser = argparse.ArgumentParser(description='Convert files to greyscale.')
    parser.add_argument('--input_path', "-i", type=str,
                        help='path to the image or directory of images. \
                If converting a directory, use *')
    parser.add_argument('--output_path', "-o", type=str,
                        help='prefix of path to the where the greyscale images will be saved.')
    parser.add_argument('--res', "-r", type=int,
                    help='downscale factor.')

    args = parser.parse_args()

    pathlib.Path(args.output_path+f'_hr_x{args.res}').mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output_path+f'_lr_x{args.res}').mkdir(parents=True, exist_ok=True)

    images = image_data(args.input_path,res = args.res)

    for i,(hr,lr) in enumerate(images):
        write_png(hr, args.output_path+f'_hr_x{args.res}/{str(i):0>7}.png')
        write_png(lr, args.output_path+f'_lr_x{args.res}/{str(i):0>7}.png')

    print('converted files to greyscale')





if __name__ == "__main__":

    main()
