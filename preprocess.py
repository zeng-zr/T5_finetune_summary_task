from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
def map_function(examples):
    prefix='summarize: '
    max_input_length = 512
    max_target_length = 32
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding=True, return_tensors='pt')

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True, padding=True, return_tensors='pt')

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

num_samples=200000
def load_train_dataset(doc_file, sum_file):
    with open(doc_file, encoding='utf-8') as f:
        documents = f.read().splitlines()[:num_samples]
    with open(sum_file, encoding='utf-8') as f:
        summaries = f.read().splitlines()[:num_samples]
    
    data = {'document': documents, 'summary': summaries}
    # 创建Dataset
    dataset = Dataset.from_dict(data)
    return dataset
def load_my_dataset(doc_file, sum_file):
    with open(doc_file, encoding='utf-8') as f:
        documents = f.read().splitlines()
    with open(sum_file, encoding='utf-8') as f:
        summaries = f.read().splitlines()
    
    data = {'document': documents, 'summary': summaries}
    # 创建Dataset
    dataset = Dataset.from_dict(data)
    return dataset

def load_raw_dataset():
        train_data=load_train_dataset('data/train.src.csv','data/train.tgt.csv')
        valid_data=load_my_dataset('data/valid.src.csv','data/valid.tgt.csv')
        test_data=load_my_dataset('data/test.src.csv','data/test.tgt.csv')
        data = {'train': train_data, 'valid': valid_data, 'test': test_data}
        #raw_dataset = Dataset.from_dict(data)
        #raw_dataset=load_dataset()
        return train_data,valid_data, test_data
    
if __name__=='__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # or "0,1" for multiple GPUs
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ['WANDB_MODE'] = 'offline' # to be tested 

    train,valid, test = load_raw_dataset()
    model_checkpoint = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, cache_dir="./model")
    train_tokenized = train.map(map_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    print(train_tokenized)
    print(data_collator)
    #print(raw_dataset)
