#!/bin/bash

echo "Running smooth calib samples search"

start=1
end=1024

samples=$start
while (( $(echo "$samples <= $end" | bc -l) )); do
    echo "Running with samples=$samples"

    python main.py \
        --model meta-llama/Llama-2-7b-hf \
        --batch_size 8 \
        --device cuda \
        --wandb \
        --wandb_project smoothrot \
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
        --smooth \
        --smooth_alpha 0.5 \
        --smooth_calib_samples $samples \
        --smooth_calib_seqlen 512 \
        --rotate \
        --seed 0

    samples=$(echo "$samples * 2" | bc -l)
done

echo "Done"