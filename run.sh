python -m torch.distributed.launch --nproc_per_node=7 ./src/lora.py --model-name xlmr1 --lora-r 16 --lora-alpha 16 --lora-bias none --batch-size 2
# python -m torch.distributed.launch --nproc_per_node=7 ./src/lora.py --model-name xlmr1 --lora-r 16 --lora-alpha 16 --lora-bias none > ./log_cmd/test.log 2>&1
# python -m torch.distributed.launch --nproc_per_node=7 ./src/lora.py --model-name xlmr1 --lora-r 16 --lora-alpha 8 --lora-bias none > ./log_cmd/test.log 2>&1


