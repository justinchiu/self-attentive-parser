#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python src/main.py train \
    --use-bert \
    --model-path-base models/en_multibert_empty_l2_dev\=94.94.pt \
    --bert-model "bert-base-multilingual-cased" \
    --num-layers 2 --learning-rate 0.00005 \
    --batch-size 32 --eval-batch-size 16 --subbatch-max-tokens 500 \
    --d-label-hidden 256 --d-tag-hidden 256 \
    --metric l2 \
    --index-path index \
    --nn-prefix nl2-multi --k 32 \
    --library faiss --use-neighbours --train-through-nn \
    --save-nn-prefix nl2-multi-nn \
    --index-devid 0 \
    | tee multi_nl2_nn.log
