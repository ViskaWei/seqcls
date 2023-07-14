import os
DATA_PATH = os.path.join(os.environ["DATA"], "emotion")
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'
import torch
# torch.cuda.device_count(), torch.cuda.is_available()

import datasets
ds = datasets.load_dataset(DATA_PATH)
ds = ds.rename_column('label', 'labels')

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

ds = ds.map(lambda x: tokenizer(x['text'], max_length=512, padding='max_length', return_tensors='pt', truncation=True), batched=True)
ds.set_format('torch')

from transformers import AutoModelForSequenceClassification,  TrainingArguments, Trainer 
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    inference_mode=False, 
    r=16, 
    lora_alpha=16, 
    lora_dropout=0.1,
    bias="lora_only"
)

model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", 
                                                        #    device_map=device_map,
                                                           torch_dtype=torch.float16,
                                                            num_labels=6,)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

lr = 1e-3
batch_size = 16
num_epochs = 2

training_args = TrainingArguments(
    output_dir="./ckpt",
    logging_dir="./logs",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=10,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps = 2,
    save_strategy="steps",
    save_steps = 100,
    load_best_model_at_end=True,
    report_to="tensorboard",
)


import evaluate
accuracy = evaluate.load("accuracy")

import numpy as np
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
