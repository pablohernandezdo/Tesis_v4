#!/bin/bash

# ANN_4K_2K
echo "Training model Ann_4k_2k, lr = 1e-4, epochs = 30, batch_size = 256"
python train.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --final_batch 100\
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models/batches'  \
        --classifier ANN \
        --model_name ANN_4k_2k_1e4_256 \
        --dataset_name "STEAD" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy"