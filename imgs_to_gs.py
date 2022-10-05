#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:56:55 2022

@author: josephbriggs
"""
import glob
import pathlib
import argparse
from tqdm import tqdm
import cv2


def main():
    '''
    Converts files to greyscale.
    '''

    parser = argparse.ArgumentParser(description='Convert files to greyscale.')
    parser.add_argument('--input_path', "-i", type=str,
                        help='path to the image or directory of images. \
                If converting a directory, use *')
    parser.add_argument('--output_path', "-o", type=str,
                        help='path to the where the greyscale images will be saved.')
    args = parser.parse_args()

    files = glob.glob(args.input_path)
    pathlib.Path(args.output_path).mkdir(parents=True, exist_ok=True)

    for file in tqdm(files):
        img = cv2.imread(file)
        gs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'{args.output_path}/{file.split("/")[-1]}', gs_img)

    print('converted files to greyscale')


if __name__ == "__main__":

    main()
