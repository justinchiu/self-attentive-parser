#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 \
python src/main.py train \
    --use-bert --model-path-base models/en_bertbase_empty_nl2 \
    --bert-model "bert-base-cased" --num-layers 2 \
    --learning-rate 0.00005 --batch-size 32 --eval-batch-size 16 \
    --subbatch-max-tokens 500 \
    --d-label-hidden 256 --d-tag-hidden 256 --metric l2 \
    | tee en-base-256-nl2.log
