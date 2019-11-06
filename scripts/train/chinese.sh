CTB_DIR="/n/rush_lab/jc/data/ctb_5.1"

CUDA_VISIBLE_DEVICES=$1 python src/main.py train \
    --train-path ${CTB_DIR}/train.gold.stripped \
    --dev-path ${CTB_DIR}/dev.gold.stripped \
    --use-bert \
    --model-path-base models/zh_multibert_zeroempty_dot\
    --bert-model "bert-base-multilingual-cased" \
    --metric dot \
    --learning-rate 0.00005 --num-layers 2 --batch-size 32 --eval-batch-size 32 \
    --d-label-hidden 256 --d-tag-hidden 256 \
    | tee zh_multi_dot_zeroempty.log
