#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python src/main.py train \
    --use-bert --model-path-base models/en_multibert_empty_l2 \
    --bert-model "bert-base-multilingual-cased" \
    --num-layers 2 --learning-rate 0.00005 \
    --batch-size 32 --eval-batch-size 16 \
    --subbatch-max-tokens 500 --d-label-hidden 256 --d-tag-hidden 256 \
    --metric l2 \
    | tee multilingual_empty_nl2.log
