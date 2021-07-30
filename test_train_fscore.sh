#!/bin/bash

# CNN
echo "Training model Cnn, lr = 1e-3, epochs = 50, batch_size = 256"
python train.py \
        --lr 1e-3 \
        --device 3 \
        --epochs 50 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier CNN \
        --model_name CNN_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" 