#!/bin/bash

SEED=4
WINDOW_SIZE=1
VALID_PORTION=0.2
MODEL_TYPE=svr
KERNEL=rbf
DEGREE=1

TRAIN_FILE=resource/train.csv
TEST_FILE=resource/test.csv
EXPERIMENT_FILE=experiments/rbf_total.csv
OUT_DIR=result_total

for C in 1e4 1e5 1e6 1e7; do
        for GAMMA in 1e-8 1e-7 1e-6 1e-5; do
              for EPSILON in 1e4 1e5 1e6 1e7; do
                  python src/train.py \
                      --model_type=${MODEL_TYPE} \
                      --window_size=${WINDOW_SIZE} \
                      --valid_portion=${VALID_PORTION} \
                      --kernel=${KERNEL} \
                      --gamma=${GAMMA} \
                      --c=${C} \
                      --epsilon=${EPSILON} \
                      --degree=${DEGREE} \
                      --train_file=${TRAIN_FILE} \
                      --test_file=${TEST_FILE} \
                      --experiments_dir=${EXPERIMENT_FILE} \
                      --output_dir=${OUT_DIR} \
                      --seed=${SEED} \
                      --overwrite_output_dir \
                      --do_train
              done
        done
done

