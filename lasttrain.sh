#!/bin/bash

# ANN_6K_3K
echo "Training model Ann_6k_3k, lr = 1e-5, epochs = 30, batch_size = 256"
python train.py \
        --lr 1e-5 \
        --device 3 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN \
        --model_name ANN_6k_3k_1e5_256 \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P1=$!

# CNN
echo "Training model Cnn2, lr = 1e-3, epochs = 30, batch_size = 256"
python train.py \
        --lr 1e-3 \
        --device 3 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier CNN \
        --model_name CNN_1e3_256 \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P2=$!

# CRED
echo "Training model CRED, lr = 1e-3, epochs = 30, batch_size = 256"
python train.py \
        --lr 1e-3 \
        --device 3 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier CRED \
        --model_name CRED_1e3_256 \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" 

P3=$!
wait $P1 $P2 $P3