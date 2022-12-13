from transformers import pipeline
import gradio as gr
from pytube import YouTube
import os

pipe = pipeline(model="CsanadT/whisper_small_sv")

def transcribe_live(audio):
    text = pipe(audio)["text"]
    return text

def transcribe_url(url):
    youtube = YouTube(str(url))
    audio = youtube.streams.filter(only_audio=True).first().download('yt_video')
    text = pipe(audio)["text"]
    return text

url_demo = gr.Interface(
    fn = transcribe_url, 
    inputs = "text", 
    outputs = "text",
    title = "Swedish Whisper",
    description = "Transciption of a swedish YouTube video via a fine-tuned Whisper model",
)

voice_demo = gr.Interface(
    fn=transcribe_live, 
    inputs=gr.Audio(source="microphone", type="filepath"), 
    outputs="text",
    title="Swedish Whisper",
    description="Live transcription of swedish speech via a fine-tuned Whisper model",
)

demo = gr.TabbedInterface([url_demo, voice_demo], ["YouTube video transciption", "Live audio to Text"])

demo.launch()
