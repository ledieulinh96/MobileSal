#!/bin/bash
SAVE_PREFIX=./snapshots/train/

SAVE_PATH=$SAVE_PREFIX

# CUDA_VISIBLE_DEVICES=4 python3 tools/train.py --file_root ../data/ \
#                                          --max_epochs 60 \
#                                          --batch_size 10 \
#                                          --savedir $SAVE_PATH \
#                                          --depth 1 \
#                                          --lr_mode poly \
#                                          --lr 1e-4 \
#                                          --inWidth 320 \
#                                          --inHeight 320 \
#                                          --ms 1
#                                          --depth_weight 0.3

# Define the source file path
SOURCE_FILE="./snapshots/train/_ep60/model_60.pth"

# Define the destination directory path
DESTINATION_DIR="./pretrained"

# Check if the destination directory exists
if [ ! -d "$DESTINATION_DIR" ]; then
  mkdir -p "$DESTINATION_DIR"
fi

# Copy the file to the destination directory
cp "$SOURCE_FILE" "$DESTINATION_DIR"

# Check if the copy was successful
if [ $? -eq 0 ]; then
  echo "File has been successfully copied to $DESTINATION_DIR"
else
  echo "Failed to copy the file"
fi

PREFIX=./pretrained/
MODEL_NAME=model_60
MODEL_PATH=$PREFIX$MODEL_NAME.pth

python3 tools/test.py --pretrained $MODEL_PATH \
                                      --savedir ./maps/$MODEL_NAME/ \
                                      --depth 1 \
                                      --data_dir ../data

python3 speed_test.py