#!/bin/bash

# ANN_6K_3K
echo "Training model ANN_6k_3k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_til_batch.py \
        --lr 1e-5 \
        --device 3 \
        --epochs 30 \
        --batch_size 256 \
        --final_batch 3901 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'modelos_batches'  \
        --classifier ANN \
        --model_name ANN_6k_3k_1e5_256 \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy"

# CNN
echo "Training model CNN, lr = 1e-3, epochs = 30, batch_size = 256"
python train_til_batch.py \
        --lr 1e-3 \
        --device 3 \
        --epochs 30 \
        --batch_size 256 \
        --final_batch 7719 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'modelos_batches'  \
        --classifier CNN \
        --model_name CNN_1e3_256 \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" 

# CRED
echo "Training model CRED, lr = 1e-3, epochs = 30, batch_size = 256"
python train_til_batch.py \
        --lr 1e-3 \
        --device 3 \
        --epochs 30 \
        --batch_size 256 \
        --final_batch 8338 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'modelos_batches'  \
        --classifier CRED \
        --model_name CRED_1e3_256 \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" 
