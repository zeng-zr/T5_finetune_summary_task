from transformers import pipeline
from transformers import AutoTokenizer
import os
if __name__=='__main__': 
    model_checkpoint="t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    summarizer = pipeline("summarization", model='./model_saved',tokenizer=tokenizer,device=3)
    
    with open('pred.txt','r', encoding='utf-8') as pred_text:
        text=pred_text.read()
        pred=summarizer(text, max_length=35, min_length=25, length_penalty=2.0)
        summary_text=pred[0]['summary_text']
        print(f"predict result: {summary_text}")
        
        tokens = tokenizer.encode(summary_text, return_tensors="pt")
        num_tokens = tokens.size(1)
        print(f"Number of tokens in summary: {num_tokens}")
        