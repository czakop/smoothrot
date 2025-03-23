#!/bin/bash

python main.py \
    --model meta-llama/Llama-3.2-1B \
    --seqlen 2048 \
    --eval_dataset c4_new \
    --eval_samples 1 \
    --lm_eval_tasks lambada \
    --batch_size 1 \
    --device cuda \
    --wandb \
    --wandb_project smoothrot \
    --w_bits 4 \
    --w_group_size -1 \
    --gptq_calib_samples 128 \
    --a_bits 8 \
    --k_bits 8 \
    --k_rotate \
    --v_bits 8 \
    --smooth_calib_samples 4 \
    --optimized_rotation_path "R.bin" \
    --seed 0
