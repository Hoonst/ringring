#!/bin/bash

SEED=4
WINDOW_SIZE=1
VALID_PORTION=0.2
MODEL_TYPE=svr
KERNEL=rbf
GAMMA=1e1
C=1e4
DEGREE=1

TRAIN_FILE=resource/train.csv
TEST_FILE=resource/test.csv
EXPERIMENT_FILE=experiments/rbf_c1.csv
OUT_DIR=rbf_c1_result
GAMMA=1e-5
EPSILON=1e-5

python src/train.py \
    --model_type=svr \
    --window_size=1 \
    --valid_portion=0.2 \
    --kernel=rbf \
    --gamma=0.0000001 \
    --c=10000000000 \
    --epsilon=10000 \
    --degree=3 \
    --train_file=resource/train.csv \
    --test_file=resource/test.csv \
    --experiments_dir=experiments/sample.csv \
    --output_dir=result \
    --seed=4 \
    --overwrite_output_dir \
    --do_train \
    --do_summary \
    --with_csv




