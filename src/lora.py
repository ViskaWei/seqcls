# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'
import numpy as np
import datetime
import argparse
import torch
# torch.cuda.device_count(), torch.cuda.is_available()
from transformers import AutoModelForSequenceClassification,  TrainingArguments, Trainer 
from peft import get_peft_model, LoraConfig, TaskType
import evaluate

import util

def run(args):

    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    token = args.model_name
    tokenizer = util.get_tokenizer(token)
    ds = util.get_ds(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(util.TOKEN[token],
                                                            #    device_map=device_map,
                                                            torch_dtype=torch.bfloat16,
                                                                num_labels=6,)
    
    #get time stamp
    now = datetime.datetime.now()
    timestamp = now.strftime("%m%d_%H%M")
    
    if args.with_lora:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, 
            inference_mode=False, 
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            target_modules=['query', 'key', 'value', 'dense'],
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        logname = f"r{args.lora_r}_a{args.lora_alpha}_b{args.lora_bias}_ep{args.num_epochs}_bs{args.batch_size}_lr{args.lr}"
    else:
        logname = f"ep{args.num_epochs}_bs{args.batch_size}_lr{args.lr}"


    training_args = TrainingArguments(
        output_dir=util.OUTDIR + f"{token}_{timestamp}_{logname}",
        logging_dir=util.LOGDIR + f"{token}_{timestamp}_{logname}",
        learning_rate=lr,
        weight_decay=0.01,

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,

        num_train_epochs=num_epochs,

        gradient_accumulation_steps=args.gradient_accumulation_step,
        
        evaluation_strategy="steps",
        eval_steps = args.eval_steps,

        logging_strategy="steps",
        logging_steps = args.logging_steps,

        save_strategy="steps",
        save_steps = 10,
        load_best_model_at_end=True,

        report_to="tensorboard",
    )


    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)


    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'], 
        tokenizer=tokenizer,
        # data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gradient-accumulation-step", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=5)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--model-name", type=str, default="xlmr0")


    ################ LoRA Args ################
    parser.add_argument("--with-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=2)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-bias", type=str, default="all", choices=["none", "lora_only", "all"])


    args = parser.parse_args()
    print(args)
    print(args.eval_steps)
    run(args)

    