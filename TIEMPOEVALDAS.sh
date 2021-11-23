#!/bin/bash

# ANN
python eval.py \
    --device 2 \
    --batch_size 256 \
    --model_folder 'modelos_batches'  \
    --classifier ANN \
    --model_name ANN_TIEMPO_PAPER \
    --dataset_name "DasTest" \
    --test_path "Data/TestReady/DasTest.npy"

# CNN
python eval.py \
    --device 2 \
    --batch_size 256 \
    --model_folder 'modelos_batches'  \
    --classifier CNN \
    --model_name CNN_TIEMPO_PAPER \
    --dataset_name "DasTest" \
    --test_path "Data/TestReady/DasTest.npy"

# CRED
python eval.py \
    --device 2 \
    --batch_size 256 \
    --model_folder 'modelos_batches'  \
    --classifier CRED \
    --model_name CRED_TIEMPO_PAPER \
    --dataset_name "DasTest" \
    --test_path "Data/TestReady/DasTest.npy"