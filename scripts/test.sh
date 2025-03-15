#!/bin/bash

python main.py \
    --model meta-llama/Llama-3.2-1B \
    --seqlen 2048 \
    --num_samples 8 \
    --batch_size 1 \
    --device cuda \
    --wandb \
    --wandb_project smoothrot \
    --quantize \
    --w_bits 8 \
    --w_group_size 128 \
    --w_clip_ratio 0.9 \
    --a_bits 8 \
    --k_bits 8 \
    --k_rotate \
    --v_bits 8 \
    --smooth \
    --smooth_calib_samples 4 \
    --rotate \
    --seed 0
