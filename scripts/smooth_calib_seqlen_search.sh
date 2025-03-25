#!/bin/bash

echo "Running smooth calib seqlen search"

start=1
end=2048

seqlen=$start
while (( $(echo "$seqlen <= $end" | bc -l) )); do
    echo "Running with seqlen=$seqlen"

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
        --smooth_calib_samples 128 \
        --smooth_calib_seqlen $seqlen \
        --rotate \
        --seed 0

    seqlen=$(echo "$seqlen * 4" | bc -l)
done

echo "Done"