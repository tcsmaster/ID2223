from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import torch
import transformers
from datasets import Dataset, DatasetDict
from modal import App, Image, Secret, Volume
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

MODEL_FOLDER = '/model'
MODEL_NAME = "openai/whisper-small"
DATASET_PATH = '/dataset'

image = Image\
    .from_registry('python:3.12-slim-bookworm') \
    .apt_install(["ffmpeg"])\
    .uv_sync() \
    .env({'HF_XET_HIGH_PERFORMANCE':'1', 'HF_HUB_CACHE':MODEL_FOLDER}) \

volume = Volume.from_name('processed_data', create_if_missing=True)
secret = Secret.from_dotenv(__file__)
DATASET_PATH = Path("./datasets")
app =App("Hungarian AS(M)R", image=image, volumes = {DATASET_PATH:volume}, secrets=[secret])

def compute_metrics(pred:Dataset,metric:evaluate.Metric, tokenizer:transformers.AutoTokenizer):
    pred_ids = pred.predicitons
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens = True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens = True)
    wer = 100*metric.compute(predictions = pred_str, references = label_str)
    return wer

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods,
        # first treat the audio inputs by simply returning torch tensors,
        #TODO: define the processor from feature_pipeline.py here
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors= "pt")
        # get the tokenized label sequences,
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length,
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly,
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # if bos token is appended in previous tokenization step,,
        # cut bos token here as it's append later anyways,
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

@app.function()
def main():
    common_voice= DatasetDict.load_from_disk("common_voice")
    processor = WhisperProcessor('openai/whisper-small')
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("wer")
    model = WhisperForConditionalGeneration("openai/whisper-small")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    training_args = Seq2SeqTrainingArguments(
        num_train_epochs=1,
        output_dir="./output_dir",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=4000, # use less steps if needed
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        generation_max_length=225,
        eval_steps=500,
        logging_steps=25,
        report_to=None,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
    )
    trainer = Seq2SeqTrainer(
        args = training_args,
        model = model,
        train_dataset=common_voice['train'],
        eval_dataset=common_voice['test'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()
if __name__ == "__main__":
    main()
