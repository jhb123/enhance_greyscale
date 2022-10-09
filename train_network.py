#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 17:42:00 2022

@author: josephbriggs
"""
import argparse
from pathlib import Path
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import enhance_greyscale_network
from torchmetrics import PeakSignalNoiseRatio


class ImageDataset(Dataset):
    def __init__(self, high_res_img_dir, low_res_img_dir, length=-1):
        self.high_res_img_dir = Path(high_res_img_dir)
        self.low_res_img_dir = Path(low_res_img_dir)
        if length == -1:
            for i, _ in enumerate(self.low_res_img_dir.glob('*.png')):
                pass
            self.length = i
        else:
            self.length = length

    def __len__(self):
        return self.length

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
    plt.show()


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    psnr = PeakSignalNoiseRatio(data_range=1)

    for batch, (high_res, low_res) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(low_res)
        loss = loss_fn(pred, high_res)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss_f, current = loss.item(), batch * len(low_res)
            noise_metric = psnr(pred, high_res).item()
            print(f"loss: {loss_f:>7f}, psnr = {noise_metric:>7f}  [{current:>6d}/{size:>6d}]")
    return loss.item()


def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for high_res, low_res in dataloader:
            pred = model(low_res)
            test_loss += loss_fn(pred, high_res).item()

    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss


def load_datasets(high_res_dir, low_res_dir, batch_size, length=-1):
    images = ImageDataset(high_res_dir, low_res_dir, length)

    train_count = int(0.9 * len(images))
    test_count = len(images) - train_count

    train_dataset, test_dataset =  \
        torch.utils.data.random_split(
            images, (train_count, test_count))

    train_dataset_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)


    test_dataset_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset_loader, test_dataset_loader, images


def main():

    
    parser = argparse.ArgumentParser(description='train the enhancement neural network.')
    parser.add_argument('--high_res', "-hr", type=str,
                        help='directory of the high resoultion training set.')
    parser.add_argument('--low_res', "-lr", type=str,
                        help='directory of the low resoultion training set.')
    parser.add_argument('--resolution', "-r", type=int,
                        help='upscaling factor.')
    
    parser.add_argument('--learning_rate', "-a", type=float,default = 1e-4,
                        help='learning rate.')
    parser.add_argument('--batch_size', "-b", type=int,default = 10,
                        help='batch size.')
    parser.add_argument('--epochs', "-e", type=int,default = 1,
                        help='number of passes through the data set.')
    parser.add_argument('--datasubset', "-d", type=int,default = -1,
                        help='Amount of data to train with.')
    
    args = parser.parse_args()

    
    
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    dataset_size = args.datasubset
    
    res_factor = args.resolution

    model = enhance_greyscale_network.GreyscaleSuperResModel(res_factor)
    loss = torch.nn.MSELoss()
    
    train_dataset_loader, test_dataset_loader, images = \
        load_datasets(args.high_res, args.low_res,
                      batch_size, dataset_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(train_dataset_loader, model, loss, optimizer)
        test_loop(test_dataset_loader, model, loss)
    print("Done!")

    torch.save(model, f'enhance_x{res_factor}.pt')

    test_img = images[1]
    learned = model(test_img[1]).detach()
    low_res = test_img[1].detach()
    high_res = test_img[0].detach()

    show_image(low_res, high_res, learned)


if __name__ == "__main__":
    main()
