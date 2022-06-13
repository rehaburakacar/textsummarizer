from turtle import color
from numpy import integer
from transformers import pipeline
import os
import tkinter as tk
import language_tool_python 
from tkinter import *
from tkinter import filedialog
import textract
from googletrans import Translator
import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence
translator = Translator()

root = Tk()
root.geometry('+%d+%d'%(350,10)) #place GUI at x=350, y=10

header = Frame(root, width=1000, height=450, bg="#52adc8")
header.grid(columnspan=3, rowspan=2, row=0)
#main content area - text and image extraction
main_content = Frame(root, width=1000, height=450, bg="#52adc8")
main_content.grid(columnspan=3, rowspan=2, row=4)

frame = tk.Frame(root, bg="#52adc8")
frame.place(relwidth=0.8, relheight=0.3, relx = 0.1, rely = 0.1)
label = tk.Label(frame, text="All content of document will be shown here.", wraplength=1000,  fg="white", bg= "#52adc8")
label.pack()

frame2 = tk.Frame(root, bg="#52adc8")
frame2.place(relwidth=0.8, relheight=0.3, relx = 0.1, rely = 0.65)
label = tk.Label(frame2, text="Summarized text will be shown here.", wraplength=1000, fg="white", bg= "#52adc8" )
label.pack()

save_img_menu = Frame(root, width=1000, height=60, bg="#387ca3")
save_img_menu.grid(columnspan=3, rowspan=1, row=3)

def makeChunksAndExecute(text):
    ## To get better results from long texts, we divided our data into chunks in this function.
    summarized = ""
    i = 0 
    end = 90
    slice = text.split()
    lenofwords = len(slice)
    count = int(lenofwords/90)
    mod = lenofwords % 90
    for x in range(0, count+1):
        temp = slice[i:end]
        tempText = ""
        control = ""
        for x in range(0, len(temp)):
            control += temp[x] + " "
            tempText += temp[x] + " "
            x+=1
        summarized = summarized + summarizeTexts(tempText)
        i += 90
        end += 90
        x += 1
    return summarized

def makeMeaningful(my_text):
    my_tool = language_tool_python.LanguageTool('en-US')
    # Given text      
    # Getting the matches  
    my_matches = my_tool.check(my_text)  
    # Defining some variables  
    myMistakes = []  
    myCorrections = []  
    startPositions = []  
    endPositions = []  
    # Using the for-loop  
    for rules in my_matches:  
        if len(rules.replacements) > 0:  
            startPositions.append(rules.offset)  
            endPositions.append(rules.errorLength + rules.offset)  
            myMistakes.append(my_text[rules.offset : rules.errorLength + rules.offset])  
            myCorrections.append(rules.replacements[0])  
    # Creating new object  
    my_NewText = list(my_text)   
    # Rewriting the correct passage  
    for n in range(len(startPositions)):  
        for i in range(len(my_text)):  
            my_NewText[startPositions[n]] = myCorrections[n]  
            if (i > startPositions[n] and i < endPositions[n]):  
                my_NewText[i] = ""  
    my_NewText = "".join(my_NewText)  
    return my_NewText 
# Create a speech recognition object
r = sr.Recognizer()
# A function that splits the audio file into chunks
# And applies speech recognition
def get_large_audio_transcription(path):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """
    # Open the audio file using pydub
    sound = AudioSegment.from_wav(path)  
    # Split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # Experiment with this value for your target audio file
        min_silence_len = 500,
        # Adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # Keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    # Create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # Process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # Export audio chunk and save it in
        # The `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # Recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            # Try converting it to text
            try:
                text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                print(chunk_filename, ":", text)
                whole_text += text
    # Return the text for all chunks detected

    return whole_text


## Setting to use the 0th GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Setting to use the bart-large-cnn model for summarization
summarizer = pipeline("summarization")

## To use the t5-base model for summarization:
## summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-ba
apps = []





def summarizeTexts(text): 
    ## To get rid of previous run outputs
    for widget in frame2.winfo_children():
        widget.destroy()
    text = translator.translate(text, dest='en').text
    summary_text = summarizer(text, max_length=120000000000, min_length=5, do_sample=False)[0]['summary_text']
    ##In this codeblock, summarized text can be translated to any language if wanted.
    summary_text = translator.translate(summary_text, dest='tr').text
    return summary_text

app = [] 

def summarizeFromPDF():
    # To get rid of previous run outputs.
    for widget in frame.winfo_children():
        widget.destroy()
    # Ask which file will be opened as a popup window
    filename = filedialog.askopenfilename(initialdir="/", title="Select File", 
    filetypes=(("DOCX", "*.DOCX"), ("all files", "*.DOCX*")))
    PDF_read = textract.process(filename, extension='docx', encoding='utf-8')
    for app in apps: 
        label = tk.Label(frame, text=app)
        label.pack()
    # Encoding the content of DOCX file to process
    str1 = PDF_read.decode('UTF-8')
    if (translator.detect(str1).lang == "en"): 
            str1 = makeMeaningful(str1)
    # Process the content of DOCX file as chunks to get better outputs from long texts
    summarized = makeChunksAndExecute(str1)
    label = tk.Label(frame, text=str1, wraplength=800,  fg="white", bg= "#52adc8")
    label.pack()
    label = tk.Label(frame2, text=summarized, wraplength=800,  fg="white", bg= "#52adc8")
    label.pack()

def summarizeFromAudio():
    #To get rid of previous run outputs
        for widget in frame.winfo_children():
            widget.destroy()
        # Popup window to determine which file to be opened
        filename = filedialog.askopenfilename(initialdir="/", title="Select File", 
        filetypes=(("WAV", "*.wav"), ("all files", "*.wav*")))
        Audio_read = get_large_audio_transcription(filename)
        for app in apps: 
            label = tk.Label(frame, text=app)
            label.pack()
        label = tk.Label(frame, text=Audio_read,  wraplength=800,  fg="white", bg= "#52adc8")
        label.pack()  
        summarized = makeMeaningful(Audio_read)
        summarized = makeChunksAndExecute(Audio_read)
        label = tk.Label(frame2, text=summarized, wraplength=800,  fg="white", bg= "#52adc8")
        label.pack()


# Placing buttons
copyText_btn = Button(root, text="Summarize From DOCX", font=("shanti", 10), height=1, width=20, command=summarizeFromPDF, bg="#55c39e", fg="white")
saveAll_btn = Button(root, text="Summarize From Audio", font=("shanti", 10), height=1, width=20, command = summarizeFromAudio, bg="#55c39e", fg="white")
copyText_btn.grid(row=3,column=0)
saveAll_btn.grid(row=3,column=2)


root.mainloop()
