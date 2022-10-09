#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:14:00 2022

@author: josephbriggs
"""
import argparse
import numpy as np
import torch
import cv2


def main():
    '''
    A script for enhancing greyscale images.
    '''

    parser = argparse.ArgumentParser(description="upscale a greyscale image.")
    parser.add_argument("--input_image", "-i", type=str,
                        help="path to the images.")
    parser.add_argument("--output_image", "-o", type=str,
                        help="output image name.")
    parser.add_argument("--model_path", "-m", type=str, default="enhance_x2.pt",
                        help="path to a pre-trained model. enhance_x2 by default")
    args = parser.parse_args()

    model = torch.load(args.model_path)
    model.eval()

    image = cv2.imread(args.input_image)
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # the netowrk works with 32bit floats between 0 and 1.
    image_gs_norm = image_gs.astype(np.float32)/255
    super_res = model(torch.tensor(image_gs_norm).unsqueeze(0))

    cv2.imwrite(args.output_image, super_res.squeeze(0).detach().numpy()*255)


if __name__ == "__main__":
    main()
