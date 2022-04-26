from transformers import pipeline
import os
import tensorflow as tf
## Setting to use the 0th GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Setting to use the bart-large-cnn model for summarization
summarizer = pipeline("summarization")

## To use the t5-base model for summarization:
## summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-ba

text = """The 1921–22 season was Cardiff City's first in the First Division of the Football League following election from the Southern Football League to the Second Division for the 1920–21 season."""

summary_text = summarizer(text, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
print(summary_text)
print(tf.__version__)