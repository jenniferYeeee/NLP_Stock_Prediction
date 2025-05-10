from datasets import load_dataset
from transformers import AutoTokenizer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dataset = load_dataset('csv', data_files={'train': '/home/serhii/NLP_Stock_Prediction/vader_finetuning/wsb_labeled_data.csv'})

from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")

def tokenize_function(examples):
    tokens = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=40)

    tokens['labels'] = examples['sentiment']
    return tokens

tokenized_datasets = dataset.map(tokenize_function, batched=True)
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone", num_labels=3)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='/home/serhii/NLP_Stock_Prediction/vader_finetuning/finbert_wsb',
    eval_strategy="epoch",   
    save_strategy="epoch",  
    eval_steps=500,
    logging_steps=500,
    save_steps=500,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['train'],
)

trainer.train()