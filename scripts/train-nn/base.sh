#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python src/main.py train \
    --use-bert \
    --model-path-base models/en_bertbase_empty_nl2_dev\=95.32.pt \
    --bert-model "bert-base-cased" --num-layers 2 \
    --learning-rate 0.00005 \
    --batch-size 32 --eval-batch-size 16 --subbatch-max-tokens 500 \
    --d-label-hidden 256 --d-tag-hidden 256 \
    --metric l2 \
    --index-path index --nn-prefix nl2-base --k 32 \
    --library faiss --use-neighbours --train-through-nn \
    --index-devid 0 | tee base-nl2-nn.log
