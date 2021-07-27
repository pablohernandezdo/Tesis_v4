#!/bin/bash

# ANN_4K_2K
echo "Training model Ann_4k_2k, lr = 1e-4, epochs = 30, batch_size = 256"
python train_til_batch.py \
        --lr 1e-4 \
        --device 3 \
        --epochs 30 \
        --batch_size 256 \
        --final_batch 1260 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'modelos_batches'  \
        --classifier ANN \
        --model_name ANN_4k_2k_1e4_256 \
        --dataset_name "STEAD" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy"

# CNN2
echo "Training model Cnn2, lr = 1e-3, epochs = 30, batch_size = 256"
python train_til_batch.py \
        --lr 1e-3 \
        --device 3 \
        --epochs 30 \
        --batch_size 256 \
        --final_batch 10140 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'modelos_batches'  \
        --classifier CNN \
        --model_name CNN2_1e3_256 \
        --dataset_name "STEAD" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" 

# CRED
echo "Training model CRED, lr = 1e-4, epochs = 30, batch_size = 256"
python train_til_batch.py \
        --lr 1e-4 \
        --device 3 \
        --epochs 30 \
        --batch_size 256 \
        --final_batch 7740 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'modelos_batches'  \
        --classifier CRED \
        --model_name CRED_1e4_256 \
        --dataset_name "STEAD" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" 
