#!/bin/bash

python main.py \
    --model meta-llama/Llama-3.2-1B \
    --seqlen 2048 \
    --num_samples -1 \
    --batch_size 1 \
    --device cuda \
    --wandb \
    --wandb_project smoothrot \
    --quantize \
    --w_bits 4 \
    --a_bits 4 \
    --smooth \
    --smooth_calib_samples 4 \
    --rotate \
    --seed 0
