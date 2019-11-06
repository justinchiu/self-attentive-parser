#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python src/main.py train \
    --use-bert --model-path-base models/en_multibert_dot_zeroempty \
    --bert-model "bert-base-multilingual-cased" \
    --num-layers 2 --learning-rate 0.00005 \
    --batch-size 32 --eval-batch-size 16 \
    --subbatch-max-tokens 500 --d-label-hidden 256 --d-tag-hidden 256 \
    --zero-empty \
    | tee multilingual-dot-zeroempty.log

