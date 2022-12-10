## Lab 2

In this lab I fine-tuned the [Whisper](https://openai.com/blog/whisper/) model, which is a speech recognition model developed by OpenAI, to transcribe swedish audio.

Thanks to the community over at Huggingface, the components of the model are readily available and usable.

The project a broken down into 3 steps:

- Feature pipeline: get the dataset from Huggingface, create features as input to the network, and store it in Google Drive.
- Training pipeline: tune the base model with our features on Colab
- Creating an interface to interact with the model: for this I used Huggingface spaces + Gradio

There are 3 ways to interact ith the model:
- Transcribe a YouTube video
- Transcribe live audio
- Transcribe a local file

You can play with th model [here](https://huggingface.co/spaces/CsanadT/Swedish_ASmR)
