import os
import time
import argparse
from pathlib import Path

import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from humanfriendly import format_timespan

from models import *
from dataset import NpyDataset


def main():
    # Measure exec time
    start_time = time.time()

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",
                        default='STEAD-STEAD',
                        help="Name of dataset to evaluate on")
    parser.add_argument("--model_name", default='1h6k_test_model',
                        help="Name of model to save")
    parser.add_argument("--model_folder", default='test',
                        help="Folder to save model")
    parser.add_argument("--classifier", default='1h6k',
                        help="Choose classifier architecture")
    parser.add_argument("--train_path", default='Train_data.hdf5',
                        help="HDF5 train Dataset path")
    parser.add_argument("--val_path", default='Validation_data.hdf5',
                        help="HDF5 validation Dataset path")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Size of the batches")
    parser.add_argument("--eval_iter", type=int, default=10,
                        help="Number of batches between validations")
    parser.add_argument("--earlystop", type=int, default=1,
                        help="Early stopping flag, 0 no early stopping")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Adam learning rate")
    parser.add_argument("--wd", type=float, default=0,
                        help="weight decay parameter")
    parser.add_argument("--b1", type=float, default=0.9,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.99,
                        help="adam: decay of first order momentum of gradient")
    args = parser.parse_args()

    # Select training device
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # Train dataset
    train_set = NpyDataset(args.train_path)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=8)

    # Validation dataset
    val_set = NpyDataset(args.val_path)
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=8)

    # Load specified Classifier
    net = get_classifier(args.classifier)
    net.to(device)

    # Count number of parameters
    params = count_parameters(net)

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr,
                           betas=(args.b1, args.b2), weight_decay=args.wd)

    # Train model
    train_model(train_loader, args.dataset_name, val_loader, net,
                device, args.epochs, optimizer, criterion,
                args.earlystop, args.patience, args.eval_iter,
                f'{args.model_folder}', args.model_name)

    # Measure training, and execution times
    train_end = time.time()

    # Training time
    train_time = train_end - start_time

    print(f'Execution details: \n{args}\n'
          f'Number of parameters: {params}\n'
          f'Training time: {format_timespan(train_time)}')


def train_model(train_loader, dataset_name, val_loader, net, device, epochs,
                optimizer, criterion, earlystop, patience,
                eval_iter, model_folder, model_name):

    # Training and validation errors
    tr_accuracies = []
    val_accuracies = []

    # Training and validation losses
    tr_losses = []
    val_losses = []

    # Early stopping
    val_acc = 1
    early = np.zeros(patience).tolist()

    with tqdm.tqdm(total=epochs, desc='Epochs', position=0) as epoch_bar:
        for epoch in range(epochs):

            n_correct, n_total = 0, 0

            # Early stopping
            if all(val_acc <= i for i in early) and earlystop:
                break

            with tqdm.tqdm(total=len(train_loader),
                           desc='Batches', position=1) as batch_bar:

                for i, data in enumerate(train_loader):

                    inputs, labels = data[0].to(device), data[1].to(device)

                    net.train()
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = net(inputs)

                    # pred labels
                    pred = torch.round(outputs)

                    # Calculate accuracy on current batch
                    n_total += labels.size(0)
                    n_correct += (pred == labels).sum().item()
                    train_acc = 100 * n_correct / n_total

                    # Calculate loss
                    loss = criterion(outputs, labels.float())

                    # Brackprop
                    loss.backward()

                    # Optimize
                    optimizer.step()

                    # Check validation accuracy periodically
                    if i % eval_iter == 0:
                        # Switch model to eval mode
                        net.eval()

                        # Calculate accuracy on validation
                        total_val_loss = 0
                        total_val, correct_val = 0, 0

                        with torch.no_grad():
                            for val_data in val_loader:

                                # Retrieve data and labels
                                traces, labels = val_data[0].to(device),\
                                                 val_data[1].to(device)

                                # Forward pass
                                outputs = net(traces)

                                # Calculate loss
                                val_loss = criterion(outputs, labels.float())

                                # Total loss for epoch
                                total_val_loss += val_loss.item()

                                # pred labels
                                pred = torch.round(outputs)

                                # Sum up correct and total validation examples
                                total_val += labels.size(0)
                                correct_val += (pred == labels).sum().item()

                            val_avg_loss = total_val_loss / len(val_loader)

                        # Calculate validation accuracy
                        val_acc = 100 * correct_val / total_val

                        # Save acc for early stopping
                        early.pop(0)
                        early.append(val_acc)

                    # Save loss to list
                    val_losses.append(val_avg_loss)
                    tr_losses.append(loss.cpu().detach().numpy())

                    # Append training and validation accuracies
                    tr_accuracies.append(train_acc)
                    val_accuracies.append(val_acc)

                    batch_bar.update()

                    # Early stopping
                    if all(val_acc <= i for i in early) and earlystop:
                        break

                epoch_bar.update()

    if len(model_folder.split("/")) > 1:
        model_dirname = model_folder.split("/")[-1]

    else:
        model_dirname = ""

    # Plot train and validation accuracies
    learning_curve_acc(tr_accuracies, val_accuracies,
                       f'Figures/Learning_curves/{dataset_name}/'
                       f'Accuracy/{model_dirname}',
                       model_name)

    # Plot train and validation losses
    learning_curve_loss(tr_losses, val_losses,
                        f'Figures/Learning_curves/{dataset_name}/'
                        f'Loss/{model_dirname}',
                        model_name)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder, exist_ok=True)

    torch.save(net.state_dict(),
               f'{model_folder.split("/")[-1]}/{model_name}.pth')


def learning_curve_acc(tr_acc, val_acc, savepath, model_name):

    if not os.path.exists(savepath):
        os.makedirs(savepath, exist_ok=True)

    plt.figure(figsize=(20, 20))
    line_tr, = plt.plot(tr_acc, label='Training accuracy')
    line_val, = plt.plot(val_acc, label='Validation accuracy')
    plt.grid(True)
    plt.xlabel('Batches')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy learning curve model {model_name}')
    plt.legend(handles=[line_tr, line_val], loc='best')
    plt.savefig(f'{savepath}/{model_name}_accuracies.png')


def learning_curve_loss(tr_loss, val_loss, savepath, model_name):

    if not os.path.exists(savepath):
        os.makedirs(savepath, exist_ok=True)

    plt.figure(figsize=(20, 20))
    line_tr, = plt.plot(tr_loss, label='Training Loss')
    line_val, = plt.plot(val_loss, label='Validation Loss')
    plt.grid(True)
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.title(f'Loss learning curve model {model_name}')
    plt.legend(handles=[line_tr, line_val], loc='best')
    plt.savefig(f'{savepath}/{model_name}_losses.png')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_classifier(x):
    if x == 'CRED':
        return CRED()
    elif x == 'ANN':
        return ANN()
    elif x == 'CNN':
        return CNN()


if __name__ == "__main__":
    main()
