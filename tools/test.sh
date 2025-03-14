#!/bin/bash
PREFIX=./pretrained/
MODEL_NAME=model_60
MODEL_PATH=$PREFIX$MODEL_NAME.pth

python3 tools/test.py --pretrained $MODEL_PATH \
                                      --savedir ./maps/$MODEL_NAME/ \
                                      --depth 1 \
                                      --data_dir ../data \



