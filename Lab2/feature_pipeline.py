import asyncio
import os

import modal
from dotenv import load_dotenv
from modal import App, Image, Secret, Volume

load_dotenv()
hf_token = os.environ['HF_TOKEN']

MODEL_FOLDER = '/model'
MODEL_NAME = "openai/whisper-small"
DATASET_PATH = '/dataset'
MINUTES = 60
CPU_REQUEST = 2.0
CPU_LIMIT = 4.0
MEMORY_REQUEST=8192
MEMORY_LIMIT=8192

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
with data_volume.batch_upload() as batch:
    batch.put_directory('./dataset', '/') # with using data_volume, / is already /dataset in the volume

@app.function(timeout=20*MINUTES)
def create_dataset():
    from datasets import Audio, load_dataset
    for sp in ['train', 'dev', 'test']:
       ds = load_dataset('audiofolder', data_dir=f'/dataset/clips/{sp}/', split='train').cast_column('audio', Audio(sampling_rate =16_000))
       ds.save_to_disk(f'/dataset/{sp}')
"""

@app.function(timeout=10*MINUTES)
def fre_load_model():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_NAME, ignore_patterns = ['*.h5', '*.bin'])

@app.cls(timeout = 5*MINUTES,max_containers=3,)
class Extractor:

    @modal.enter()
    def load(self):
        from transformers import WhisperFeatureExtractor, WhisperTokenizer
        self.extractor =  WhisperFeatureExtractor.from_pretrained(MODEL_NAME, hf_token=hf_token,  cache_dir=MODEL_FOLDER)
        self.tokenizer =  WhisperTokenizer.from_pretrained(MODEL_NAME, hf_token=hf_token,  cache_dir=MODEL_FOLDER)
        
    @modal.batched(wait_ms = 60000, max_batch_size=200)
    def process(self, batch):
        feat = self.extractor([el['audio'].get_all_samples().data.squeeze().numpy for el in batch], sampling_rate=16_000, padding=True)
        feat = list(feat.input_features)
        labels = self.tokenizer(batch['text']).input_ids
        return [dict(input_features=feats, labels=token) for feats, token in zip(feat, labels)]

@app.function(
        timeout=60*MINUTES,
)
async def transcribe_dataset(split:str):
    from datasets import load_dataset
    ext = Extractor()
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    result = []
    async for res in ext.process.map.aio(ds):
        result.extend(res)
    print(len(result))
    # new_ds = Dataset.from_list(result)
    # new_ds.save_to_disk(f'/dataset/processed_clips/{split}')
    # data_volume.commit()

@app.local_entrypoint()
async def main() -> None:
    # split =['train', 'dev', 'test']
    split = ['we']
    await asyncio.gather(*[transcribe_dataset.remote.aio(sp) for sp in split])
