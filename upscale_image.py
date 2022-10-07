#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:14:00 2022

@author: josephbriggs
"""

import numpy as np
import torch
import cv2


def main():

    model = torch.load('enhance_x2.pt')
    model.eval()

    test_image = cv2.imread(r"crossword.jpeg")[1000:1500, 0:800]
    test_image_gs = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    test_image_gs_norm = test_image_gs.astype(np.float32)/255

    super_res = model(torch.tensor(test_image_gs_norm).unsqueeze(0))

    cv2.imshow("input", test_image_gs)
    cv2.imshow("super res", super_res.squeeze(0).detach().numpy())

    cv2.waitKey()


if __name__ == "__main__":
    main()
