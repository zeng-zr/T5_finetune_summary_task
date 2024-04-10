## chinese sum task
import torch
import os
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import nltk
import numpy as np
import preprocess 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # or "0,1" for multiple GPUs
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['WANDB_MODE'] = 'offline' # to be tested 
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
def map_function(examples):
    #prefix='summarize: '
    prefix=''
    max_input_length = 512
    max_target_length = 32
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding=True, return_tensors='pt')
    
    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True, padding=True, return_tensors='pt')

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

#  collate_fn accepts tokenized_dataset batch
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    TOKENIZE_CHINESE = lambda x: ' '.join(
        tokenizer.convert_ids_to_tokens(tokenizer(x).input_ids, skip_special_tokens=True)
    )
    
    decoded_preds = [TOKENIZE_CHINESE(pred.strip()) for pred in decoded_preds]
    decoded_labels = [TOKENIZE_CHINESE(label.strip()) for label in decoded_labels]

    # 计算ROUGE分数
    result = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
        use_aggregator=True,
    )
    
    result = {key: value * 100 for key, value in result.items()}

    prediction_lens = [len(pred.split()) for pred in decoded_preds]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

if __name__ == "__main__":
    train_raw, valid_raw, test_raw=preprocess.load_raw_dataset()
    metric = load("rouge")

    model_checkpoint = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False, leagcy=True)
    train_tokenized = train_raw.map(map_function, batched=True)
    valid_tokenized = valid_raw.map(map_function, batched=True)
#    test_tokenized = test_raw.map(map_function, batched=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, cache_dir="./model")

    batch_size = 64
    model_name = model_checkpoint.split("/")[-1]
    args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-lcsts",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        generation_max_length=25,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
    )

    #  not only pad the inputs to the maximum length in the batch, but also the labels:
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_tokenized,
#        eval_dataset=valid_tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    # Finetune
    trainer.train()
    trainer.save_model('./model_zh_saved')
    predictions=trainer.predict(valid_tokenized[:10])
    print(predictions)
    torch.cuda.empty_cache()
