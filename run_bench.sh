#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
ep=2
batch_size=14
token_values=(bert0 xmlr0 xmlr1)

for token in "${token_values[@]}"; do
    echo Experiment token:${token} no lora start
    python -m torch.distributed.launch --nproc_per_node=7 ./src/lora.py \
        --model-name   ${token}  \
        --num-epochs   ${ep}  \
        --batch-size   ${batch_size}  \
        > ./log_cmd/test_${token}_no_lora.log 2>&1
    echo Experiment token:${token} no lora done
done
