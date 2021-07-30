import os
import time
import argparse
from pathlib import Path

import tqdm
import torch
import numpy as np
import pandas as pd
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
    parser.add_argument("--device", type=int, default=3,
                        help="Training gpu device")
    parser.add_argument("--workers", type=int, default=0,
                        help="Dataloader num workers")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Size of the batches")
    parser.add_argument("--eval_iter", type=int, default=1,
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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Train dataset
    train_set = NpyDataset(args.train_path)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers)

    # Validation dataset
    val_set = NpyDataset(args.val_path)
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.workers)

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
          f'Classifier: {args.classifier}\n'
          f'Number of parameters: {params}\n'
          f'Training time: {format_timespan(train_time)}')


def train_model(train_loader, dataset_name, val_loader, net, device, epochs,
                optimizer, criterion, earlystop, patience,
                eval_iter, model_folder, model_name):

    # Training and validation accuracies
    # tr_accuracies = []
    # val_accuracies = []

    # Training and validation fscores
    tr_fscores = []
    val_fscores = []

    # Training and validation losses
    tr_losses = []
    val_losses = []

    # Early stopping counter
    early_counter = 0
    # current_best_acc = 0
    current_best_fscore = 0
    best_n_batches = 0
    total_batches = 0

    with tqdm.tqdm(total=epochs, desc='Epochs', position=0) as epoch_bar:
        for epoch in range(epochs):

            # Number of correctly classfied training examples
            n_total, n_correct = 0, 0

            # Early stopping
            if early_counter >= patience and epoch > 0 and earlystop:
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
                    #n_total += labels.size(0)
                    #n_correct += (pred == labels).sum().item()
                    #train_acc = 100 * n_correct / n_total

                    # Calculate fscore on current batch
                    train_tp, train_fp, train_tn, train_fn = confusion(pred, labels)
                    train_fscore = (2 * train_tp) / ( 2* train_tp + train_fn + train_fp)

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
                        # total_val_loss = 0
                        # total_val, correct_val = 0, 0

                        # Calculate fscore on validation
                        val_tps = 0
                        val_tns = 0
                        val_fps = 0
                        val_fns = 0

                        # Calculate loss on validation
                        total_val_loss = 0

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

                                # Calcular casos de prediccion
                                tp, fp, tn, fn = confusion(pred, labels)
                                val_tps += tp
                                val_tns += tn
                                val_fps += fp
                                val_fns += fn

                                # Sum up correct and total validation examples
                                # total_val += labels.size(0)
                                # correct_val += (pred == labels).sum().item()

                        # Loss promedio en el dataset de validacion
                        val_avg_loss = total_val_loss / len(val_loader)

                        # Calculate validation accuracy
                        # val_acc = 100 * correct_val / total_val

                        # Calcular validation fscore
                        val_fscore = (2 * val_tps) / ( 2* val_tps + val_fns + val_fps)

                    # Save loss to list
                    val_losses.append(val_avg_loss)
                    tr_losses.append(loss.cpu().detach().numpy())

                    # Append training and validation accuracies
                    # tr_accuracies.append(train_acc)
                    # val_accuracies.append(val_acc)
                    
                    # Append training and validation fscores
                    tr_fscores.append(train_fscore)
                    val_fscores.append(val_fscore)

                    # Update number of batches
                    total_batches += 1
                    batch_bar.update()

                    # # Check if performance increased
                    # if val_acc > current_best_acc:
                    #     # earlystop counter 0
                    #     early_counter = 0

                    #     # guardar checkpoint
                    #     best_model_params = net.state_dict()

                    #     # guardar mejor numero de batches
                    #     best_n_batches = total_batches

                    #     # actualizar current_best_acc
                    #     current_best_acc = val_acc

                    # else:
                    #     early_counter += 1

                    # Check if performance increased
                    if val_fscore > current_best_fscore:
                        # earlystop counter 0
                        early_counter = 0

                        # guardar checkpoint
                        best_model_params = net.state_dict()

                        # guardar mejor numero de batches
                        best_n_batches = total_batches

                        # actualizar current_best_fscore
                        current_best_fscore = val_fscore

                    else:
                        early_counter += 1

                    # Early stopping
                    if early_counter >= patience and epoch > 0 and earlystop:
                        break

                epoch_bar.update()

    if len(model_folder.split("/")) > 1:
        model_dirname = model_folder.split("/")[-1]

    else:
        model_dirname = ""

    # EVAL ITER VA A TENER QUE SER SIEMPRE 1
    # Save train and validation accuracies/losses to csv
    os.makedirs(f'LearningCurves/{dataset_name}/', exist_ok=True)

    # pd_train_acc = pd.DataFrame({'TrainAcc': tr_accuracies})
    pd_train_fscore = pd.DataFrame({'TrainAcc': tr_fscores})
    pd_train_loss = pd.DataFrame({'TrainLoss': tr_losses})

    # pd_val_acc = pd.DataFrame({'ValAcc': val_accuracies})
    pd_val_fscore = pd.DataFrame({'ValAcc': val_fscores})
    pd_val_loss = pd.DataFrame({'ValLoss': val_losses})

    # best_n_batches = [best_n_batches] * len(tr_accuracies)
    best_n_batches = [best_n_batches] * len(tr_fscores)
    pd_n_batches = pd.DataFrame({'Train_batches': best_n_batches})

    # pd_data = pd.concat([pd_train_acc,
    #                      pd_train_loss,
    #                      pd_val_acc,
    #                      pd_val_loss,
    #                      pd_n_batches], axis=1)

    pd_data = pd.concat([pd_train_fscore,
                         pd_train_loss,
                         pd_val_fscore,
                         pd_val_loss,
                         pd_n_batches], axis=1)

    pd_data.to_csv(f'LearningCurves/{dataset_name}/'
                   f'{model_name}.csv', index=False)

    # Plot train and validation accuracies
    # learning_curve_acc(tr_accuracies, val_accuracies,
    #                    f'Figures/Learning_curves/{dataset_name}/'
    #                    f'Accuracy/{model_dirname}',
    #                    model_name, best_n_batches[0])

    # Plot train and validation fscores
    learning_curve_fscore(tr_fscores, val_fscores,
                          f'Figures/Learning_curves/{dataset_name}/'
                          f'Fscore/{model_dirname}',
                          model_name, best_n_batches[0])

    # Plot train and validation losses
    learning_curve_loss(tr_losses, val_losses,
                        f'Figures/Learning_curves/{dataset_name}/'
                        f'Loss/{model_dirname}',
                        model_name, best_n_batches[0])

    if not os.path.exists(model_folder):
        os.makedirs(model_folder, exist_ok=True)

    torch.save(best_model_params,
               f'{model_folder.split("/")[-1]}/{model_name}.pth')

def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives


# def learning_curve_acc(tr_acc, val_acc, savepath,
#                        model_name, n_batch):

#     if not os.path.exists(savepath):
#         os.makedirs(savepath, exist_ok=True)

#     plt.figure(figsize=(20, 20))
#     line_tr, = plt.plot(tr_acc, label='Training accuracy')
#     line_val, = plt.plot(val_acc, label='Validation accuracy')
#     plt.axvline(n_batch, label='')
#     plt.grid(True)
#     plt.xlabel('Batches')
#     plt.ylabel('Accuracy')
#     plt.title(f'Accuracy learning curve model {model_name}')
#     plt.legend(handles=[line_tr, line_val], loc='best')
#     plt.savefig(f'{savepath}/{model_name}_accuracies.png')

def learning_curve_fscore(tr_fscore, val_fscore, savepath,
                          model_name, n_batch):

    if not os.path.exists(savepath):
        os.makedirs(savepath, exist_ok=True)

    plt.figure(figsize=(20, 20))
    line_tr, = plt.plot(tr_fscore, label='Training Fscore')
    line_val, = plt.plot(val_fscore, label='Validation Fscore')
    plt.axvline(n_batch, label='')
    plt.grid(True)
    plt.xlabel('Batches')
    plt.ylabel('Fscore')
    plt.title(f'Fscore learning curve model {model_name}')
    plt.legend(handles=[line_tr, line_val], loc='best')
    plt.savefig(f'{savepath}/{model_name}_fscores.png')


def learning_curve_loss(tr_loss, val_loss, savepath,
                        model_name, n_batch):

    if not os.path.exists(savepath):
        os.makedirs(savepath, exist_ok=True)

    plt.figure(figsize=(20, 20))
    line_tr, = plt.plot(tr_loss, label='Training Loss')
    line_val, = plt.plot(val_loss, label='Validation Loss')
    plt.axvline(n_batch, label='')
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
