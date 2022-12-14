#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 17:34:03 2022

@author: josephbriggs
"""

import torch
import torch.nn as nn

class GreyscaleSuperResModel(nn.Module):

    def __init__(self,res):
        '''
        Implementation of a neural network inspired by
        https://arxiv.org/pdf/1609.05158.pdf

        Returns
        -------
        Model.
        '''
        super(GreyscaleSuperResModel, self).__init__()
        self.res = res
        # three layers

        # shape: n_(l-1) * n_l * k_l * k_l
        # n_l = number of features at layer l
        # k_l = filter size at layer l
        # f_l function applied to matrix elementwise


        self.conv1 = nn.Conv2d(1, 500, 5,padding='same',padding_mode='replicate') # 5x5 kernel, 64 features
        self.conv2 = nn.Conv2d(500, 32, 3,padding='same',padding_mode='replicate') # 3x3 kernel, 32 features
        self.conv3 = nn.Conv2d(32,res**2,3,padding = 'same',padding_mode='replicate') #

        
    def forward(self,x):

        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
   
        pixel_shuffle = nn.PixelShuffle(self.res)
        x = pixel_shuffle(x)
        return x



    def cost(self, high_res_generate, high_res_original):
        '''
        return losses and signal to noise ratio.
        Returns
        -------
        None.

        '''
