#!/bin/bash

# ANN_6K_3K
echo "Training model Ann_6k_6k, lr = 1e-6, epochs = 200, batch_size = 256"
python train_fsc.py \
        --lr 1e-6 \
        --device 3 \
        --epochs 200 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN \
        --model_name ANN_6k_6k_1e6_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P1=$!

# CNN
echo "Training model Cnn1, lr = 1e-5, epochs = 200, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 3 \
        --epochs 200 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier CNN \
        --model_name CNN_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P2=$!

# CRED
echo "Training model CRED, lr = 1e-5 epochs = 200, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 3 \
        --epochs 200 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier CRED \
        --model_name CRED_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" 

P3=$!
wait $P1 $P2 $P3