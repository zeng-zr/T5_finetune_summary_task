from transformers import pipeline
from transformers import AutoTokenizer
import os
if __name__=='__main__': 
    model_checkpoint="csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False, legacy = True)
    summarizer = pipeline("summarization", model='./model_zh_saved',tokenizer=tokenizer,device=2,cache_dir="./model")

    text="""
    受众在哪里，媒体就应该在哪里，媒体的体制、内容、技术就应该向哪里转变。
    媒体融合关键是以人为本，即满足大众的信息需求，为受众提供更优质的服务。
    这就要求媒体在融合发展的过程中，既注重技术创新，又注重用户体验。
    """
    pred=summarizer(text, max_length=25, min_length=5, length_penalty=2.0)
    summary_text=pred[0]['summary_text']
    print(pred)
    print(f"predict result: {summary_text}")

    '''    
        with open('pred.txt','r', encoding='utf-8') as pred_text:
            text=pred_text.read()   
            pred=summarizer(text, max_length=15, min_length=5, length_penalty=2.0)
            summary_text=pred[0]['summary_text']
            print(pred)
            print(f"predict result: {summary_text}")
            tokens = tokenizer.encode(summary_text, return_tensors="pt")
            num_tokens = tokens.size(1)
            print(f"Number of tokens in summary: {num_tokens}")
    '''