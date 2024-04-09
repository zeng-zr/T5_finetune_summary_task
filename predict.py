from transformers import pipeline

# 指定模型和分词器的保存路径
model_path = './model_saved'
# 如果使用了快速分词器 (Fast Tokenizer)，确保tokenizer也正确保存在这个目录下

# 加载训练好的模型和分词器，创建摘要pipeline
summarizer = pipeline("summarization", model=model_path, tokenizer=model_path)

# 你给定的text
text = ""

# 执行摘要任务
summary = summarizer(text, max_length=50, min_length=25, length_penalty=2.0)

# 打印生成的摘要
print(summary[0]['summary_text'])