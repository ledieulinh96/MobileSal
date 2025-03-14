#!/bin/bash
SAVE_PREFIX=./snapshots/train/

SAVE_PATH=$SAVE_PREFIX

CUDA_VISIBLE_DEVICES=4 python3 -m debugpy --listen 5678 --wait-for-client tools/train_remove.py --file_root ../data/ \
                                         --max_epochs 60 \
                                         --num_workers 1 \
                                         --batch_size 10 \
                                         --savedir $SAVE_PATH \
                                         --depth 1 \
                                         --lr_mode poly \
                                         --lr 1e-4 \
                                         --inWidth 320 \
                                         --inHeight 320 \
                                         --ms 1
                                         

