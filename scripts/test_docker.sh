#!/bin/bash

docker run --rm -it \
    --gpus all \
    --env-file .env \
    czakop/smoothrot:0.1.0 \
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
    --a_bits 8 \
    --smooth \
    --smooth_calib_samples 4 \
    --rotate \
    --seed 0
