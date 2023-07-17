#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
epoch=3
batch_size=8
token_values=(bert0 xmlr0 xmlr1)
lora_r_values=(8 16 32)      
lora_alpha_values=(16 32)    # twice of rank
lora_bias_values=(all lora_only)  
for token in "${token_values[@]}"; do
    for lora_r in "${lora_r_values[@]}"; do
        for lora_alpha in "${lora_alpha_values[@]}"; do
            for lora_bias in "${lora_bias_values[@]}"; do
                echo Experiment token:${token} with lora start
                python -m torch.distributed.launch --nproc_per_node=7 ./src/lora.py \
                --model-name   ${token}      \
                --with-lora                  \
                --lora-r       ${lora_r}     \
                --lora-alpha   ${lora_alpha} \
                --lora-bias    ${lora_bias}  \
                --num-epochs   ${epoch}         \
                --batch-size   ${batch_size} \
                > ./log_cmd/test_${token}_with_lora.log 2>&1
                echo Experiment token:${token} with lora done
            done
        done
    done
done



