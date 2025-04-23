#!/bin/bash

python main.py \
    --model meta-llama/Llama-3.2-1B \
    --seqlen 2048 \
    --eval_samples 8 \
    --lm_eval_tasks lambada \
    --batch_size 1 \
    --device cuda \
    --wandb \
    --wandb_project smoothrot \
    --wandb_act_scale_artifact act_scales \
    --quantize \
    --gptq \
    --gptq_calib_samples 2 \
    --w_bits 4 \
    --w_group_size -1 \
    --gptq_calib_samples 128 \
    --a_bits 4 \
    --k_bits 4 \
    --k_rotate \
    --v_bits 4 \
    --smooth_calib_samples 4 \
    --optimized_rotation_path "R.bin" \
    --seed 0
