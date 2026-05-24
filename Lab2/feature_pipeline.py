import asyncio
import os

from dotenv import load_dotenv
from modal import App, Image, Secret, Volume
from transformers import WhisperFeatureExtractor, WhisperTokenizer

load_dotenv()
hf_token = os.environ['HF_TOKEN']

MODEL_FOLDER = '/model'
MODEL_NAME = "openai/whisper-small"
DATASET_PATH = '/dataset'
MINUTES = 60
CPU_REQUEST = 2.0
CPU_LIMIT = 4.0
MEMORY_REQUEST=8192
MEMORY_LIMIT=16384

image = Image\
    .from_registry('python:3.12-slim-bookworm') \
    .apt_install(["ffmpeg"])\
    .uv_sync() \
    .env({'HF_XET_HIGH_PERFORMANCE':'1', 'HF_HUB_CACHE':MODEL_FOLDER}) \

model_volume = Volume.from_name('model_volume', create_if_missing=True)
data_volume = Volume.from_name('processed_data', create_if_missing=True)
secret = Secret.from_dotenv(__file__)

app = App("Hungarian AS(M)R", image=image, volumes = {MODEL_FOLDER:model_volume, DATASET_PATH:data_volume}, secrets=[secret])
"""
@app.function(timeout=20*MINUTES)
def create_dataset():
    from datasets import Audio, load_dataset
    for sp in ['train', 'dev', 'test']:
       ds = load_dataset('audiofolder', data_dir=f'/dataset/clips/{sp}/', split='train').cast_column('audio', Audio(sampling_rate =16_000))
       ds.save_to_disk(f'/dataset/{sp}')
"""

@app.function(timeout=10*MINUTES)
def load_model():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_NAME, ignore_patterns = ['*.h5', '*.bin'])

def process(batch:dict, extractor, tokenizer):
    feat =extractor([el.get_all_samples().data.squeeze().numpy() for el in batch['audio']], sampling_rate=16_000, padding="max_length", return_attention_mask=True)
    batch['input_features'] = list(feat.input_features)
    batch['attention_mask'] = list(feat.attention_mask)
    batch['labels'] = tokenizer(batch['sentence']).input_ids
    return batch

@app.function(
        timeout=60*MINUTES,
        cpu = (CPU_REQUEST, CPU_LIMIT),
        memory=(MEMORY_REQUEST, MEMORY_LIMIT),
        max_containers=3,
)
async def transcribe_dataset(split:str):
    from datasets import load_from_disk
    extractor =  WhisperFeatureExtractor.from_pretrained(MODEL_NAME, hf_token=hf_token,  cache_dir=MODEL_FOLDER)
    tokenizer =  WhisperTokenizer.from_pretrained(MODEL_NAME, hf_token=hf_token,  cache_dir=MODEL_FOLDER)
    ds = load_from_disk(f"/dataset/{split}")
    ds = ds.map(process, fn_kwargs = dict(extractor=extractor, tokenizer=tokenizer), batch_size=200, batched=True, remove_columns = ds.column_names)
    ds.save_to_disk(f'/dataset/processed_{split}', max_shard_size='1GB', num_proc = os.cpu_count() - 1)
    await data_volume.commit.aio()

@app.local_entrypoint()
async def main() -> None:
    split =['train', 'dev', 'test']
    await asyncio.gather(*[transcribe_dataset.remote.aio(sp) for sp in split])
