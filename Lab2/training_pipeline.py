import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import datasets
import evaluate
import torch
import transformers
from datasets import Dataset
from modal import App, Image, Retries, Secret, Volume

MINUTES = 60
HOURS = 60*MINUTES
MODEL_FOLDER = '/model'
DATASET_PATH = '/dataset'
OUTPUT_DIR = Path( '/output' )
EXPERIMENT_PATH = OUTPUT_DIR / 'experiments'
retries = Retries(initial_delay=0.0, max_retries=10)

image = Image\
    .debian_slim(python_version='3.12') \
    .apt_install(["ffmpeg"])\
    .uv_sync() \

data_volume = Volume.from_name('processed_data', create_if_missing=True)
model_volume = Volume.from_name('model_volume', create_if_missing=True)
output_volume = Volume.from_name('output_volume', create_if_missing=True)
secret = Secret.from_dotenv(__file__)
app =App("Hungarian AS(M)R", image=image, volumes = {MODEL_FOLDER:model_volume,DATASET_PATH:data_volume, OUTPUT_DIR:output_volume}, secrets=[secret])
#TODO: wandb integration, hyperparameter tuning on dev dataset

@dataclass
class Config:
    """Training configuration."""

    model_output_name: str = "whisper-fine-tune"  # Name used for saving and loading
    wandb_project_name:str = "Whisper"

    # Model config
    model_name: str = "openai/whisper-small"
    dataset_path = Path(DATASET_PATH)
    # Dataset config
    dataset_split: str = "train"  # The test and val splits don't have category labels
    max_duration_in_seconds: float = 20.0
    min_duration_in_seconds: float = 0.0

    # Training config
    num_train_epochs: int = 5
    warmup_steps: int = 400
    max_steps: int = -1
    batch_size: int = 64
    learning_rate: float = 1e-5
    eval_strategy: str = "epoch"

def compute_metrics(pred:Dataset,metric:evaluate.Metric, tokenizer:transformers.AutoTokenizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens = True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens = True)
    wer = 100*metric.compute(predictions = pred_str, references = label_str)
    return {'wer':wer}

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    decoder_start_token_id: int
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods,
        # first treat the audio inputs by simply returning torch tensors,
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors= "pt") #TODO: is this necessary? the preprocess pads to max_length
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

@app.function(
    image=image,
    gpu="a10g",
    timeout=3 * HOURS,
    max_containers=1,
    retries = retries,
)
def train(
        experiment:str
):
    """Loads data and trains the model."""
    config = Config
    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir= EXPERIMENT_PATH / experiment,
        num_train_epochs=config.num_train_epochs,
        save_strategy="epoch",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        logging_steps=5,
        learning_rate=config.learning_rate,
        lr_scheduler_type='cosine',
        warmup_steps=config.warmup_steps,
        torch_compile=True,
        max_steps=config.max_steps,
        eval_strategy=config.eval_strategy,
        bf16=True,
        report_to = "wandb",
        run_name = config.wandb_project_name,
        predict_with_generate=True,
        generation_max_length=40,
        generation_num_beams=1,
    )

    os.environ["WANDB_PROJECT"] = config.wandb_project_name
    os.environ["WANDB_WATCH"] = "false"
    os.environ["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    print(f"Loading model: {config.model_name}")
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(
        config.model_name, cache_dir = MODEL_FOLDER
    )
    tokenizer = transformers.WhisperTokenizer.from_pretrained(
        config.model_name, cache_dir = MODEL_FOLDER
    )
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        config.model_name, cache_dir = MODEL_FOLDER
    )
    ds1 = datasets.load_from_disk(config.dataset_path / "processed_train")
    ds2 = datasets.load_from_disk(config.dataset_path / "processed_dev")
    ds3 = datasets.load_from_disk(config.dataset_path / "processed_test")
    dataset = datasets.concatenate_datasets([ds1, ds2])

    # Create a processor that combines the feature extractor and tokenizer
    processor = transformers.WhisperProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )

    # Custom data collator handles batching of variable-length audio sequences
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Set up the Hugging Face trainer with all of our components
    metric = evaluate.load("wer")

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset, 
        eval_dataset=ds3,
        processing_class=feature_extractor,
        data_collator=data_collator,
        compute_metrics=functools.partial(
            compute_metrics,
            tokenizer=tokenizer,
            metric=metric,
        ),
    )

    print("Running evals before training to establish a baseline")
    metrics = trainer.evaluate(
        metric_key_prefix="baseline",
        max_length=training_args.generation_max_length,
        num_beams=training_args.generation_num_beams,
    )
    trainer.log_metrics("baseline", metrics)
    trainer.save_metrics("baseline", metrics)

    print(f"Starting training! Weights will be saved to '{training_args.output_dir}'")
    train_result = trainer.train()

    # Save the model weights, tokenizer, and feature extractor
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    feature_extractor.save_pretrained(training_args.output_dir)

    # Log training metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Final evaluation to see how much we improved
    print("Running final evals")
    metrics = trainer.evaluate(
        metric_key_prefix="test",
        max_length=training_args.generation_max_length,
        num_beams=training_args.generation_num_beams,
    )
    metrics["eval_samples"] = len(dataset["test"])

    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)
    output_volume.commit()  # Ensure everything is saved to the Volume

    print(f"\nTraining complete! Model saved to '{training_args.output_dir}'")

@app.local_entrypoint()
def main(experiment:Optional[str] = None):
    """Run Whisper fine-tuning on Modal."""
    if experiment is None:  
        import uuid
        experiment = uuid.uuid4().hex[:8]
    print(f"Staritng experiment {experiment}")

    train.spawn(experiment).get()

