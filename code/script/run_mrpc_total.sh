#!/bin/bash

TASK_NAME="mrpc"
OUTPUT_DIR=${TASK_NAME}"_result"
MAX_SEQ_LENGTH=128
EXPERIMENT_DIR="experiments/"${TASK_NAME}"_ori_128.csv"
MODEL_PATH="pretrain_output/mlm_output/"${TASK_NAME}"/ori_50k"
SEED=4
for TRAIN_BATCH_SIZE in 8 16 32; do
      for NUM_EPOCH in 2 3; do
              for LEARNING_RATE in 1e-5 2e-5 3e-5 4e-5 5e-5; do
                    CUDA_VISIBLE_DEVICES=0,1 python run_glue.py \
                          --model_name_or_path ${MODEL_PATH} \
                          --task_name ${TASK_NAME} \
                          --seed ${SEED} \
                          --do_train \
                          --do_eval \
                          --max_seq_length ${MAX_SEQ_LENGTH} \
                          --per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
                          --per_device_eval_batch_size 64 \
                          --learning_rate ${LEARNING_RATE} \
                          --num_train_epochs ${NUM_EPOCH} \
                          --output_dir ${OUTPUT_DIR} \
                          --overwrite_output_dir \
                          --experiments_dir ${EXPERIMENT_DIR}
              done
      done
done


