from transformers import pipeline
import os
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, Text
# from pdf import PdfFileReader
import textract





## Setting to use the 0th GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Setting to use the bart-large-cnn model for summarization
summarizer = pipeline("summarization")

## To use the t5-base model for summarization:
## summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-ba
apps = []

def summarizeTexts(): 
    text = """Prepared by experienced English teachers, the texts, articles and conversations are brief and appropriate to your level of proficiency. Take the multiple-choice quiz following each text, and you'll get the results immediately. You will feel both challenged and accomplished! You can even download (as PDF) and print the texts and exercises. It's enjoyable, fun and free. Good luck!
"""
    summary_text = summarizer(text, max_length=300, min_length=5, do_sample=False)[0]['summary_text']
    print(summary_text)
    print(tf.__version__)

root = tk.Tk()

canvas = tk.Canvas(root, height=700, width=700, bg="#263D42")
canvas.pack()

frame = tk.Frame(root, bg="white")
frame.place(relwidth=0.8, relheight=0.8, relx = 0.1, rely = 0.1)

summarizeText = tk.Button(root, text="Summarize Text", padx=10, pady=5, fg="white", bg="#263D42", command=summarizeTexts)
summarizeText.pack()

def summarizeFromPDF():
    filename = filedialog.askopenfilename(initialdir="/", title="Select File", 
    filetypes=(("pdf", "*.PDF"), ("all files", "*.PDF*")))
    print (filename)
    apps.append(filename)
    PDF_read = textract.process(filename, method='PDFminer')
    # temp = open(filename, 'rb')
    # PDF_read = PDFFileReader(temp)
    # first_page = PDF_read.getPage(0)
    # print(first_page.extractText())
    label = tk.Label(frame, text=PDF_read, bg="gray")
    label.pack()
summarizeTextFromPDF = tk.Button(root, text="Summarize Text From PDF", padx=10, pady=5, fg="white", bg="#263D42", command=summarizeFromPDF)
summarizeTextFromPDF.pack()

root.mainloop()
