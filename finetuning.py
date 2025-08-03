
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
import sys
import unsloth
import torch
from torch.utils.data import DataLoader
import peft
from typing import Dict, Optional, List
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq,set_seed,AutoConfig

from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer,DataCollatorForCompletionOnlyLM
from utils.multi_way_tree_metric import MultiWayTreeMetric
import datasets
from tqdm import tqdm
import numpy as np
import evaluate
import json
import time
random_seed = 42
set_seed(random_seed)# 这个比torch.manual_seed更全面

import wandb
model_path = 'models/Qwen2.5-7B-Instruct-1M'
max_length = 20000
lora_adapter_path = ''



metric_rouge = evaluate.load('evaluate/hf-evaluate/rouge')
train_dataset_path = ''

dataset = datasets.load_dataset('json',data_files=train_dataset_path)
dataset = dataset['train'].train_test_split(test_size=0.2,seed=random_seed)

train_dataset = dataset['train']
eval_dataset = dataset['test']
#test_dataset = datasets.load_dataset('json',data_files={'test':test_dataset_path})['test']

tokenizer = AutoTokenizer.from_pretrained(model_path)
def dataset_function(examples) -> Dict:
    instruction =examples['question'] 
    instruction_tokenized = tokenizer(instruction,truncation=True, max_length=max_length, padding=False)
    instruction = tokenizer.decode(instruction_tokenized['input_ids'],skip_special_tokens=True)
    response = examples['cleaned_answer']
    result= {
        "instruction": instruction,
        "output": response,
        }
    return result
remove_columns = train_dataset.column_names
#["id","name","question","answer","cleaned_answer","reasoning",]
train_dataset = train_dataset.map(dataset_function,batched=False,remove_columns=remove_columns, load_from_cache_file=False)
eval_dataset = eval_dataset.map(dataset_function,batched=False,remove_columns=remove_columns, load_from_cache_file=False)
#test_dataset = test_dataset.map(dataset_function,batched=False,remove_columns=remove_columns, load_from_cache_file=False)

# load model and tokenizer
#config = AutoConfig.from_pretrained("models/Qwen2.5-7B-Instruct")
def formatting_prompts_func(example):
    output_texts = []
    
    if isinstance(example['instruction'],str):
        return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>\n"
    elif isinstance(example['instruction'],list):
        for i in range(len(example['instruction'])):
            text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{example['instruction'][i]}<|im_end|>\n<|im_start|>assistant\n{example['output'][i]}<|im_end|>\n"
            output_texts.append(text)
        return output_texts
    else:
        raise ValueError("Invalid type of input")
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

multi_way_tree_metric = MultiWayTreeMetric(
    tokenizer=tokenizer,
    sentence_transformer_model='models/GTE/gte-Qwen2-1.5B-instruct',
    bert_model='models/GTE/biobert-v1.1',
    device='cuda:0',
    )
def preprocess_logits_for_metrics(logits, labels):
    predictions = torch.argmax(logits, dim=2)
      
    return predictions


def compute_metrics(eval_pred):

    predictions, labels = eval_pred # np.array
    index = labels != -100
    new_labels = np.where(index, labels, tokenizer.pad_token_id)
    new_predictions_position_converted = np.zeros_like(predictions)
    for i, row in enumerate(predictions):
        idx = len(row) - 1
        while idx >= 0 and row[idx] == -100:
            idx -= 1
        if idx >= 0:
            target_element = row[idx]
            new_row = np.concatenate(([target_element], row[:idx], row[idx+1:]))
        else:
            new_row = row
        new_predictions_position_converted[i] = new_row
    new_predictions = np.where(index, new_predictions_position_converted, tokenizer.pad_token_id)
    p_decoded = tokenizer.batch_decode(new_predictions,skip_special_tokens=True)
    l_decoded = tokenizer.batch_decode(new_labels, skip_special_tokens=True)
    metric_rouge_result = metric_rouge.compute(predictions=p_decoded, references=l_decoded)
    
    multi_way_tree_metric.add_batch(predicts=p_decoded, references=l_decoded)
    multi_way_tree_metric_result = multi_way_tree_metric.compute()
    metric_rouge_result.update(multi_way_tree_metric_result)
    return metric_rouge_result

model, _ = FastLanguageModel.from_pretrained(
            model_name=model_path,
            dtype=None,
            load_in_4bit=False,
            max_seq_length= max_length+2048,
            device_map='cuda:0',
        )
model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=32,# 32
            lora_dropout=0.05,
            r=32, # 32
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"
                               ],
            use_gradient_checkpointing=False,
            random_state=random_seed,
        )
train_config = SFTConfig(
    output_dir=lora_adapter_path,
    do_eval=True,
    do_train=True,
    per_device_eval_batch_size=1,
    per_device_train_batch_size=1,
    #user_define_train_batch_size=1,
    #auto_find_batch_size=True,
    learning_rate=1e-4,
    num_train_epochs=3,
    save_strategy='epoch',
    #save_steps=1000,
    bf16=True,
    eval_strategy = 'epoch',
    
)

trainer = SFTTrainer(
        model=model,
        args=train_config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        formatting_func=formatting_prompts_func,
        dataset_num_proc=1,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        
  
)
#trainer.accelerator.print(f"{trainer.model}")
#trainer.model.print_trainable_parameters()
   # train
checkpoint = None

trainer.train(resume_from_checkpoint=checkpoint)



# saving final model
trainer.save_model()
