#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python src/main.py test \
    --train-path ../../data/ctb_5.1/train.gold.stripped \
    --train-path ../../data/ctb_5.1/dev.gold.stripped \
    --model-path-base models/en_multibert_dot_zeroempty_dev=94.85.pt \
    --batch-size 128 --subbatch-max-tokens 500 \
    --nn-prefix dot-multi-zeroempty --library faiss \
    --redo-vocab \
    --k 8
