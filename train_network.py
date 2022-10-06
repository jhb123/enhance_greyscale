#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 17:42:00 2022

@author: josephbriggs
"""
from pathlib import Path
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import cv2
import numpy as np
import enhance_greyscale_network


class ImageDataset(Dataset):
    def __init__(self, high_res_img_dir, low_res_img_dir):
        self.high_res_img_dir = Path(high_res_img_dir)
        self.low_res_img_dir = Path(low_res_img_dir)
        # for i,_ in enumerate(self.low_res_img_dir.glob('*.png')):
        #     pass
        # self.length = i

    def __len__(self):
        return 10000  # self.length

    def __getitem__(self, idx):
        fname = f"{str(idx):0>7}.png"
        high_res_image = read_image(
            str(self.high_res_img_dir.joinpath(fname)))/255
        low_res_image = read_image(
            str(self.low_res_img_dir.joinpath(fname)))/255
        return high_res_image, low_res_image


def show_image(low_res_img, high_res_img, learned_img):

    _, ax = plt.subplots(1, 3)
    high_res_img = high_res_img.numpy()
    low_res_img = low_res_img.numpy()
    learned_img = learned_img.numpy()

    ax[0].imshow(low_res_img[0, :, :], vmin=0, vmax=1, cmap='gray')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('LR')
    ax[1].imshow(high_res_img[0, :, :], vmin=0, vmax=1, cmap='gray')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('HR')
    ax[2].imshow(learned_img[0, :, :], vmin=0, vmax=1, cmap='gray')
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_title('Learned')


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    print(size)
    for batch, (high_res, low_res) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(low_res)
        loss = loss_fn(pred, high_res)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(low_res)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for high_res, low_res in dataloader:
            pred = model(low_res)
            test_loss += loss_fn(pred, high_res).item()

    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")


def main():

    images = ImageDataset('gs_imgs_hr_x2', 'gs_imgs_lr_x2')

    learning_rate = 1e-2
    batch_size = 10
    epochs = 1

    train_count = int(0.7 * len(images))
    valid_count = int(0.2 * len(images))
    test_count = len(images) - train_count - valid_count

    train_dataset, valid_dataset, test_dataset =  \
        torch.utils.data.random_split(
            images, (train_count, valid_count, test_count))

    train_dataset_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True)

    test_dataset_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    model = enhance_greyscale_network.GreyscaleSuperResModel(2)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(train_dataset_loader, model, loss, optimizer)
        test_loop(test_dataset_loader, model, loss)
    print("Done!")

    test_img = images[0]
    learned = model(test_img[1]).detach()
    low_res = test_img[1].detach()
    high_res = test_img[0].detach()

    show_image(low_res, high_res, learned)

    test_image = cv2.imread(r"crossword.jpeg")[1000:1500, 0:800]
    test_image_gs = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    test_image_gs_norm = test_image_gs.astype(np.float32)/255

    super_res = model(torch.tensor(test_image_gs_norm).unsqueeze(0))

    cv2.imshow("input", test_image_gs)
    cv2.imshow("super res", super_res.squeeze(0).detach().numpy())

    cv2.waitKey()


if __name__ == "__main__":
    main()
