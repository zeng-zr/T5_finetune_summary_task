from datasets import load_dataset
from transformers import AutoTokenizer

max_input_length = 512
max_target_length = 64
prefix = 'summarize:'


tokenizer = AutoTokenizer.from_pretrained(
          "csebuetnlp/mT5_multilingual_XLSum", 
          legacy=True, 
          use_fast=False, 
          padding=True,
          truncation=True, 
          return_tensors="pt"
)

def preprocess_src(src):
    inputs = [prefix + doc for doc in src["text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length)
    return model_inputs

def preprocess_tgt(tgt):
    text_target=tgt['text']
    model_inputs = tokenizer(text_target , max_length=max_target_length)
    model_inputs["labels"] = model_inputs["input_ids"]
    return model_inputs


valid_dataset = load_dataset(
          "text", 
          data_files={
                "document": "data/valid.src.csv", 
                "summary": "data/test.tgt.csv"
          }
)


#tokenized_src=valid_dataset['document'].map(preprocess_src, batched=True)
#tokenized_tgt=valid_dataset['summary'].map(preprocess_tgt, batched=True)
#print(tokenized_tgt)


