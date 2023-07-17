#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
ep=2
batch_size=14
lora_r_values=(2 16)
lora_alpha_values=(16 64)
lora_bias_values=(none all)
for lora_r in "${lora_r_values[@]}"; do
    for lora_alpha in "${lora_alpha_values[@]}"; do
        for lora_bias in "${lora_bias_values[@]}"; do
            echo Experiment lora_r:${lora_r} lora_alpha:${lora_alpha} lora_bias:${lora_bias} start
            python -m torch.distributed.launch --nproc_per_node=7 ./src/lora.py \
             --model-name   xlmr1  \
             --with-lora             \
             --lora-r       ${lora_r}  \
             --lora-alpha   ${lora_alpha}  \
             --lora-bias    ${lora_bias}  \
             --num-epochs   ${ep}  \
             --batch-size   ${batch_size}  \
              > ./log_cmd/test_${lora_r}_${lora_alpha}_${lora_bias}.log 2>&1
            echo Experiment lora_r:${lora_r} lora_alpha:${lora_alpha} lora_bias:${lora_bias} done
        done
    done
done