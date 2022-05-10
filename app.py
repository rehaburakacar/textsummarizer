from transformers import pipeline
import os
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, Text
# from pdf import PdfFileReader
import textract
import googletrans
from googletrans import Translator
translator = Translator()

text1 = "subscribe my channel"
text2 = "L'allemand ne sert à rien"

print("Translated text:", translator.translate(text1, dest='tr'))
print("Translated text:", translator.translate(text2, dest='tr'))


## Setting to use the 0th GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Setting to use the bart-large-cnn model for summarization
summarizer = pipeline("summarization")

## To use the t5-base model for summarization:
## summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-ba
apps = []

root = tk.Tk()

canvas = tk.Canvas(root, height=700, width=700, bg="#263D42")
canvas.pack()

frame = tk.Frame(root, bg="white")
frame.place(relwidth=0.8, relheight=0.3, relx = 0.1, rely = 0.1)

frame2 = tk.Frame(root, bg="white")
frame2.place(relwidth=0.8, relheight=0.3, relx = 0.1, rely = 0.5)



def summarizeTexts(text): 
    print("typeof1", type(text))
    summary_text = summarizer(text, max_length=120000000000, min_length=5, do_sample=False)[0]['summary_text']
    print("summarized text", summary_text)
    print(tf.__version__)
    label = tk.Label(frame2, text=summary_text, wraplength=500)
    label.pack()

summarizeText = tk.Button(root, text="Summarize Text", padx=10, pady=5, fg="white", bg="#263D42", command=summarizeTexts)
summarizeText.pack()

app = [] 

def summarizeFromPDF():
    for widget in frame.winfo_children():
        widget.destroy()
    filename = filedialog.askopenfilename(initialdir="/", title="Select File", 
    filetypes=(("DOCX", "*.DOCX"), ("all files", "*.DOCX*")))
    print (filename)
    #apps.append(filename)
    print(filename)
    PDF_read = textract.process(filename, extension='docx', encoding='ascii')
    for app in apps: 
        label = tk.Label(frame, text=app)
        label.pack()
    # temp = open(filename, 'rb')
    # PDF_read = PDFFileReader(temp)
    # first_page = PDF_read.getPage(0)
    # print(first_page.extractText())
    label = tk.Label(frame, text=PDF_read, wraplength=500)
    label.pack()
    str1 = PDF_read.decode('UTF-8')  
    print("typeofobject", str1)
    summarizeTexts(str1)

summarizeTextFromPDF = tk.Button(root, text="Summarize Text From DOCX", padx=10, pady=5, fg="white", bg="#263D42", command=summarizeFromPDF)
summarizeTextFromPDF.pack()

root.mainloop()
