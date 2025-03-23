#!/bin/bash

docker run --rm -it \
    --gpus all \
    czakop/smoothrot:0.1.0 \
    python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --seqlen 2048 \
    --num_samples 32 \
    --batch_size 4 \
    --device cuda \
    --quantize \
    --w_bits 4 \
    --a_bits 4 \
    --smooth \
    --smooth_calib_samples 512 \
    --rotate \
    --seed 0
