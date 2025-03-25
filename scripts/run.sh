#!/bin/bash

docker run --rm -it \
    --gpus all \
    czakop/smoothrot:0.1.0 \
    python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --seqlen 2048 \
    --eval_samples -1 \
    --batch_size 8 \
    --device cuda \
    --wandb \
    --wandb_project smoothrot \
    --wandb_act_scale_artifact act_scales \
    --smooth \
    --rotate \
    --quantize \
    --w_bits 4 \
    --a_bits 4 \
    --a_clip_ratio 0.9 \
    --k_bits 4 \
    --k_clip_ratio 0.95 \
    --k_asym \
    --k_per_head \
    --k_rotate \
    --v_bits 4 \
    --v_clip_ratio 0.95 \
    --v_asym \
    --v_per_head \
    --seed 0
