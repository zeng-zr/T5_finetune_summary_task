# T5 Fine-Tuning Project for Text Summarization
This repository contains the work and results of my fine-tuning efforts on the T5 (Text-to-Text Transfer Transformer) model for the task of text summarization, specifically on the XSUM dataset.

## Project Overview
Text summarization is a natural language processing task that involves creating short, coherent summaries from longer pieces of text. This is particularly challenging due to the need for understanding the context, main ideas, and the ability to condense information without losing critical content.  
  
The T5 model is a versatile and powerful transformer model that converts all NLP problems into a text-to-text format. For the summarization task, T5 offers promising results due to its approach of treating every language problem as a text generation task, learning to map language input sequences to output sequences.
## Dataset
The project utilizes the XSUM dataset, which comprises news articles and their corresponding one-sentence summaries. It provides a good benchmark for assessing the model's capability to generate coherent and concise summaries.
  
## Fine-Tuning Process
Here's a quick summary of the fine-tuning steps involved in this project:  
1. Preprocessing: Tokenize the input text and summaries using T5's tokenizer.
2. Model Selection: Utilize the t5-small, t5-base, or t5-large models depending on resource availability.
3. Training: Use Hugging Face's Seq2SeqTrainer API for efficient training with the appropriate data collator, optimizer, and scheduler.
4. Evaluation: Employ metrics like ROUGE for evaluating summarization quality.
5. Optimization: Trial different hyperparameters and training approaches to optimize performance.

## Usage
- run `python train.py` to start fine-tuning
- run `python pipeline.py` to generate summary for `pred.txt`
