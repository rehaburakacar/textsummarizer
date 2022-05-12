from transformers import pipeline
import os
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, Text
# from pdf import PdfFileReader
import textract
import googletrans
from googletrans import Translator
import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence

# create a speech recognition object
r = sr.Recognizer()

# a function that splits the audio file into chunks
# and applies speech recognition
def get_large_audio_transcription(path):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """
    # open the audio file using pydub
    sound = AudioSegment.from_wav(path)  
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 500,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                print(chunk_filename, ":", text)
                whole_text += text
    # return the text for all chunks detected
    return whole_text
translator = Translator()


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
    for widget in frame2.winfo_children():
        widget.destroy()
    #print("typeof1", type(text))
    #print("Translated text:", translator.translate(text, dest='en'))
    #print("TYPEOFOBJECT", type(text))
    text = translator.translate(text, dest='en').text
    #print("INCOMING TEXT:", text)
    summary_text = summarizer(text, max_length=120000000000, min_length=5, do_sample=False)[0]['summary_text']
    #print("SUMMARIZED TEXT", summary_text)
    summary_text = translator.translate(summary_text, dest='tr').text
    #print("summarized text", summary_text)
    #print(tf.__version__)
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
    #print (filename)
    #apps.append(filename)
    #print(filename)
    PDF_read = textract.process(filename, extension='docx', encoding='utf-8')
    for app in apps: 
        label = tk.Label(frame, text=app)
        label.pack()
    # temp = open(filename, 'rb')
    # PDF_read = PDFFileReader(temp)
    # first_page = PDF_read.getPage(0)
    # print(first_page.extractText())
    str1 = PDF_read.decode('UTF-8')
    #print("TEXT:", str1)
    label = tk.Label(frame, text=str1, wraplength=500)
    label.pack()  
    #print("typeofobject", str1)
    summarizeTexts(str1)

def summarizeFromAudio():
        for widget in frame.winfo_children():
            widget.destroy()
        filename = filedialog.askopenfilename(initialdir="/", title="Select File", 
        filetypes=(("WAV", "*.wav"), ("all files", "*.wav*")))
        #print (filename)
        #apps.append(filename)
        #print(filename)
        print("FILENAME IS THAT", filename)
        Audio_read = get_large_audio_transcription(filename)
        print("AUDIO", Audio_read)
        print("TYPEOF", type(Audio_read))
        for app in apps: 
            label = tk.Label(frame, text=app)
            label.pack()
        # temp = open(filename, 'rb')
        # PDF_read = PDFFileReader(temp)
        # first_page = PDF_read.getPage(0)
        # print(first_page.extractText())
        #str1 = Audio_read.decode('UTF-8')
        #print("TEXT:", str1)

        label = tk.Label(frame, text=Audio_read, wraplength=500)
        label.pack()  

        #print("typeofobject", str1)

        summarizeTexts(Audio_read)

summarizeTextFromPDF = tk.Button(root, text="Summarize Text From DOCX", padx=10, pady=5, fg="white", bg="#263D42", command=summarizeFromPDF)
summarizeTextFromPDF.pack()

summarizeTextFromAudio = tk.Button(root, text="Summarize Text From Audio File", padx=10, pady=5, fg="white", bg="#263D42", command=summarizeFromAudio)
summarizeTextFromAudio.pack()

root.mainloop()
