#!/bin/bash

SEED=4
WINDOW_SIZE=1
VALID_PORTION=0.2
MODEL_TYPE=svr
KERNEL=sigmoid
C=1e4
DEGREE=1

TRAIN_FILE=resource/train.csv
TEST_FILE=resource/test.csv
EXPERIMENT_FILE=experiments/sigmoid_total.csv
OUT_DIR=result_sigmoid_total

for C in 1e-5 1e-4 1e-3 1e-2 1 1e1 1e2 1e3 1e4; do
        for GAMMA in 1e-5 1e-4 1e-3 1e-2 1 1e1 1e2 1e3 1e4; do
              for EPSILON in 1e-5 1e-4 1e-3 1e-2 1 1e1 1e2 1e3 1e4; do
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

