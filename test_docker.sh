#!/bin/bash

docker run --rm -it \
    --gpus all \
    --env-file .env \
    czakop/smoothrot:test \
    python main.py \
    --model meta-llama/Llama-3.2-1B \
    --seqlen 2048 \
    --num_samples 1 \
    --batch_size 1 \
    --device cuda \
    --seed 0 \
    --wandb \
    --wandb_project smoothrot
