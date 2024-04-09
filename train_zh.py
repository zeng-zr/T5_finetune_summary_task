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
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # or "0,1" for multiple GPUs
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['WANDB_MODE'] = 'offline' # to be tested 
os.environ['TOKENIZERS_PARALLELISM'] = true
def map_function(examples):
    prefix='summarize: '
    max_input_length = 512
    max_target_length = 64
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding=True, return_tensors='pt')

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True, padding=True, return_tensors='pt')

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    """
    #  define a function for compute the metrics from the predictions.
    #  and do a bit of pre-processing to decode the predictions into texts
    """
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]

    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
        use_aggregator=True,
    )
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}
## load dataset: 
## map dataset
if __name__ == "__main__":
    train_raw, valid_raw, test_raw=preprocess.load_raw_dataset()
    metric = load("rouge")

    model_checkpoint = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, leagcy=True)
    train_tokenized = train_raw.map(map_function, batched=True)
    valid_tokenized = valid_raw.map(map_function, batched=True)
    test_tokenized = test_raw.map(map_function, batched=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, cache_dir="./model")

    batch_size = 128
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
        eval_dataset=valid_tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    # Finetune
    trainer.train()
    trainer.save_model('./model_zh_saved')
    torch.cuda.empty_cache()
