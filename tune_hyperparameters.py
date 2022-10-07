#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 15:54:35 2022

@author: josephbriggs
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import enhance_greyscale_network
import train_network


def tune_learning_rate(epochs, learning_rates,
                       train_dataset_loader, valid_dataset_loader,
                       model, loss):

    error_training = np.zeros(learning_rates.size)
    error_validation = np.zeros(learning_rates.size)

    for i, learning_rate in enumerate(learning_rates):
        for name, module in model.named_children():
            print('resetting ', name)
            module.reset_parameters()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_loss = 0
        validation_loss = 0

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            train_loss += train_network.train_loop(
                train_dataset_loader, model, loss, optimizer)
            validation_loss += train_network.test_loop(
                valid_dataset_loader, model, loss)

        error_training[i] = train_loss/len(train_dataset_loader)
        error_validation[i] = validation_loss/len(train_dataset_loader)

    fig, ax = plt.subplots()
    ax.plot(learning_rates, error_training, label='train')
    ax.plot(learning_rates, error_validation, label='validation')
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Average loss')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()


def tune_batch_size(epochs, batch_sizes, model, loss, dataset_size):

    learning_rate = 1e-2

    error_training = np.zeros(batch_sizes.size)
    error_validation = np.zeros(batch_sizes.size)

    for i, batch_size in enumerate(batch_sizes):
        for name, module in model.named_children():
            print('resetting ', name)
            module.reset_parameters()

        train_dataset_loader, valid_dataset_loader, test_dataset_loader, images = \
            train_network.load_datasets(
                'gs_imgs_hr_x2', 'gs_imgs_lr_x2', int(batch_size), dataset_size)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_loss = 0
        validation_loss = 0

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            train_loss += train_network.train_loop(
                train_dataset_loader, model, loss, optimizer)
            validation_loss += train_network.test_loop(
                valid_dataset_loader, model, loss)

        error_training[i] = train_loss/len(train_dataset_loader)
        error_validation[i] = validation_loss/len(train_dataset_loader)

    fig, ax = plt.subplots()
    ax.plot(batch_sizes, error_training, label='train')
    ax.plot(batch_sizes, error_validation, label='validation')
    ax.set_xlabel('Batch size')
    ax.set_ylabel('Average loss')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(batch_sizes)
    ax.minorticks_off()
    ax.set_xticklabels([str(b) for b in batch_sizes])

    ax.legend()


def tune_training_set_size(epochs, data_set_sizes, model,
                           loss, batch_size, learning_rate):

    error_training = np.zeros(data_set_sizes.size)
    error_validation = np.zeros(data_set_sizes.size)

    for i, data_set_size in enumerate(data_set_sizes):
        for name, module in model.named_children():
            print('resetting ', name)
            module.reset_parameters()

        train_dataset_loader, valid_dataset_loader, test_dataset_loader, images = \
            train_network.load_datasets(
                'gs_imgs_hr_x2', 'gs_imgs_lr_x2', batch_size, data_set_size)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_loss = 0
        validation_loss = 0

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            train_loss += train_network.train_loop(
                train_dataset_loader, model, loss, optimizer)
            validation_loss += train_network.test_loop(
                valid_dataset_loader, model, loss)

        error_training[i] = train_loss/len(train_dataset_loader)
        error_validation[i] = validation_loss/len(train_dataset_loader)

    fig, ax = plt.subplots()
    ax.plot(data_set_sizes, error_training, label='train')
    ax.plot(data_set_sizes, error_validation, label='validation')
    ax.set_xlabel('Training set size')
    ax.set_ylabel('Average loss')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()


def epoch_investigation(epochs, learning_rate,
                        train_dataset_loader, valid_dataset_loader,
                        model, loss):

    error_training = np.zeros(epochs.size)
    error_validation = np.zeros(epochs.size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss = 0
    validation_loss = 0

    for i in epochs:
        print(f"Epoch {i+1}\n-------------------------------")
        train_loss = train_network.train_loop(
            train_dataset_loader, model, loss, optimizer)
        validation_loss = train_network.test_loop(
            valid_dataset_loader, model, loss)

        error_training[i] = train_loss/len(train_dataset_loader)
        error_validation[i] = validation_loss/len(train_dataset_loader)

    fig, ax = plt.subplots()
    ax.plot(epochs, error_training, label='train')
    ax.plot(epochs, error_validation, label='validation')
    ax.set_xlabel('Epoch ')
    ax.set_ylabel('Average loss')
    ax.set_yscale('log')
    ax.legend()


def main():

    epochs = 5
    dataset_size = 1000

    model = enhance_greyscale_network.GreyscaleSuperResModel(2)
    loss = torch.nn.MSELoss()

    # test 1: does batch size affect result.
    # tune_batch_size(epochs,np.power(2,np.arange(5,11)),model,loss,dataset_size)

    batch_size = 10

    train_dataset_loader, valid_dataset_loader, test_dataset_loader, images = \
        train_network.load_datasets(
            'gs_imgs_hr_x2', 'gs_imgs_lr_x2', batch_size, dataset_size)

    # test 2: determine a suitable learning rate.
    # this is a bit pointless, adam optimiser is adaptive.
    # tune_learning_rate(25,np.logspace(-5,-1,4),train_dataset_loader,
    #                         valid_dataset_loader,model,loss)

    learning_rate = 1e-2

    # test 3: investigate the effect of training set size. 
    tune_training_set_size(epochs,np.logspace(1,4,3).astype(int),model,
                                    loss,batch_size,learning_rate)

    # test 4: investigate the effect of epochs
    # epoch_investigation(np.arange(10), learning_rate, train_dataset_loader,
    #                     valid_dataset_loader, model, loss)


if __name__ == "__main__":
    main()
