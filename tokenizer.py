from transformers import AutoTokenizer

model_checkpoint = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

if __name__=='__main__':
    inputs = tokenizer("我叫张三，在苏州大学学习计算机。")
    print(inputs)
    print(tokenizer.convert_ids_to_tokens(inputs.input_ids))