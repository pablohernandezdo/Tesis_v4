#!/bin/bash

# ANN_6K_3K
#echo "Training model ANN_6k_6k, lr = 1e-6, epochs = 100, batch_size = 256"
#python train_til_batch.py \
#        --lr 1e-6 \
#        --device 3 \
#        --epochs 100 \
#        --batch_size 256 \
#        --final_batch 22000 \
#        --earlystop 0 \
#        --eval_iter 30 \
#        --model_folder 'modelos_batches'  \
#        --classifier ANN \
#        --model_name ANN_6k_6k_1e6_256_batch_22000 \
#        --dataset_name "STEAD-ZEROS" \
#        --train_path "Data/TrainReady/train_zeros.npy" \
#        --val_path "Data/TrainReady/val_zeros.npy"

# CNN
# echo "Training model CNN, lr = 1e-3, epochs = 100, batch_size = 256"
# python train_til_batch.py \
#         --lr 1e-3 \
#         --device 3 \
#         --epochs 100 \
#         --batch_size 256 \
#         --final_batch 4472 \
#         --earlystop 0 \
#         --eval_iter 30 \
#         --model_folder 'modelos_batches'  \
#         --classifier CNN \
#         --model_name CNN_1e3_256_batch_4472 \
#         --dataset_name "STEAD-ZEROS" \
#         --train_path "Data/TrainReady/train_zeros.npy" \
#         --val_path "Data/TrainReady/val_zeros.npy" 

# CRED
echo "Training model CRED, lr = 1e-5, epochs = 100, batch_size = 256"
python train_til_batch.py \
        --lr 1e-5 \
        --device 3 \
        --epochs 100 \
        --batch_size 256 \
        --final_batch 30000 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'modelos_batches'  \
        --classifier CRED \
        --model_name CRED_1e3_256_batch_30000 \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" 
