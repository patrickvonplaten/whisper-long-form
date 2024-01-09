# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Evaluating a Whisper model on one or more evaluation datasets.
"""
# You can also adapt this script for your own speech recognition validation. Pointers for this are left as comments.

from dataclasses import dataclass
import logging
import os
from typing import Optional
import numpy as np
import sys
from dataclasses import field
from functools import partial

import datasets
import evaluate
import torch
from datasets import DatasetDict, IterableDatasetDict, load_dataset
from tqdm import tqdm
import transformers
import whisper
from transformers import HfArgumentParser, is_wandb_available
from whisper.normalizers import EnglishTextNormalizer

SAMPLING_RATE = 16_000

logger = logging.getLogger(__name__)
metric = evaluate.load("wer")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: str = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library). Load and combine "
            "multiple datasets by separating dataset hours by a '+' symbol."
        },
    )
    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The name of the model to use (via the transformers library). "
        },
    )
    condition_on_prev_tokens: bool = field(
        default=False,
        metadata={"help": "Whether to condition on previous tokens or not"},
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "The number of beams used for evluation."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    dataset_split_name: Optional[str] = field(
        default=None,
        metadata={"help": "The split name of the dataset to use (via the datasets library)."},
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to cache directory for saving and loading datasets"},
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset column containing the text data. Defaults to `text`."},
    )
    wandb_project: str = field(
        default="distil-whisper-speed-benchmark",
        metadata={"help": "The name of the wandb project."},
    )
    wandb_name: str = field(
        default=None,
        metadata={"help": "The name of the wandb run."},
    )
    wandb_job_type: str = field(
        default="distil-whisper",
        metadata={"help": "The name of the wandb job type."},
    )
    wandb_dir: str = field(
        default=None,
        metadata={"help": "The absolute path to save the wandb logs."},
    )
    save_code_to_wandb: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to save main script to wandb. This is valuable for improving"
                " experiment reproducibility and to diff code across experiments in"
                " the UI."
            )
        },
    )
    streaming: bool = field(
        default=True,
        metadata={"help": "Whether to use Datasets' streaming mode to load and the data."},
    )


def write_metric(summary_writer, eval_metrics, step, prefix="eval"):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"{prefix}/{metric_name}", value, step)


def write_wandb_metric(wandb_logger, metrics, train_time, prefix):
    log_metrics = {}
    for k, v in metrics.items():
        log_metrics[f"{prefix}/{k}"] = v
    log_metrics[f"{prefix}/time"] = train_time
    wandb_logger.log(log_metrics)  # TODO(SG): bug with wandb means we can't log the step count

def compute_metrics(pred_str, label_str, normalizer):
    # normalize everything and re-compute the WER
    norm_pred_str = [normalizer(pred) for pred in pred_str]
    norm_label_str = [normalizer(label) for label in label_str]

    wer = 100 * metric.compute(predictions=norm_pred_str, references=norm_label_str)

    return wer



def convert_dataset_str_to_list(
    dataset_names, dataset_config_names, splits=None, text_column_names=None, dataset_hours=None, default_split="train"
):
    if isinstance(dataset_names, str):
        dataset_names = dataset_names.split("+")

        # we assume that all the datasets we're using derive from the distil-whisper org on the Hub - prepend the org name if necessary
        for i in range(len(dataset_names)):
            ds_name = dataset_names[i]
            dataset_names[i] = f"distil-whisper/{ds_name}" if "/" not in ds_name else ds_name

        dataset_config_names = dataset_config_names.split("+")
        splits = splits.split("+") if splits is not None else None
        text_column_names = text_column_names.split("+") if text_column_names is not None else None
        dataset_hours = dataset_hours.split("+") if dataset_hours is not None else None

    # basic checks to ensure we've got the right number of datasets/configs/splits/columns/probs
    if len(dataset_names) != len(dataset_config_names):
        raise ValueError(
            f"Ensure one config is passed for each dataset, got {len(dataset_names)} datasets and"
            f" {len(dataset_config_names)} configs."
        )

    if splits is not None and len(splits) != len(dataset_names):
        raise ValueError(
            f"Ensure one split is passed for each dataset, got {len(dataset_names)} datasets and {len(splits)} splits."
        )

    if text_column_names is not None and len(text_column_names) != len(dataset_names):
        raise ValueError(
            f"Ensure one text column name is passed for each dataset, got {len(dataset_names)} datasets and"
            f" {len(text_column_names)} text column names."
        )

    if dataset_hours is not None:
        if len(dataset_hours) != len(dataset_names):
            raise ValueError(
                f"Ensure one probability is passed for each dataset, got {len(dataset_names)} datasets and "
                f"{len(dataset_hours)} hours."
            )
        dataset_hours = [float(ds_hours) for ds_hours in dataset_hours]
    else:
        dataset_hours = [None] * len(dataset_names)

    text_column_names = (
        text_column_names if text_column_names is not None else ["text" for _ in range(len(dataset_names))]
    )
    splits = splits if splits is not None else [default_split for _ in range(len(dataset_names))]

    dataset_names_dict = []
    for i, ds_name in enumerate(dataset_names):
        dataset_names_dict.append(
            {
                "name": ds_name,
                "config": dataset_config_names[i],
                "split": splits[i],
                "text_column_name": text_column_names[i],
                "hours": dataset_hours[i],
            }
        )
    return dataset_names_dict


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser([DataTrainingArguments])

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        data_args = parser.parse_args_into_dataclasses()[0]

    # 2. Setup logging
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    has_wandb = is_wandb_available()
    if has_wandb:
        import wandb as wandb_logger
        import wandb

        # Set up wandb run
        wandb_logger.init(
            project=data_args.wandb_project,
            name=data_args.wandb_name,
            job_type=data_args.wandb_job_type,
            dir=data_args.wandb_dir,
            save_code=data_args.save_code_to_wandb,
        )
        wandb_logger.log({"torch_version": str(torch.__version__)})
        wandb_logger.log({"transformers_version": str(transformers.__version__)})
        wandb_logger.log({"batch_size": 1})
    else:
        raise ValueError("Wandb logging requires wandb to be installed. Run `pip install wandb` to enable.")

    # 3. Load dataset
    raw_datasets = IterableDatasetDict() if data_args.streaming else DatasetDict()

    # Convert lists of dataset names/configs/splits to a dict
    # names: "librispeech_asr+gigaspeech", configs: "all+l", splits: "validation.clean+validation"
    # -> [{"name: "librispeech_asr": "config": "all", "split": "validation.clean"}, {"name: "gigaspeech": "config": "l", "split": "validation"}
    dataset_names_dict = convert_dataset_str_to_list(
        data_args.dataset_name,
        data_args.dataset_config_name,
        splits=data_args.dataset_split_name,
        text_column_names=data_args.text_column_name,
    )

    if len(dataset_names_dict) == 1:
        # load a single eval set
        dataset_dict = dataset_names_dict[0]
        raw_datasets["eval"] = load_dataset(
            dataset_dict["name"],
            dataset_dict["config"],
            split=dataset_dict["split"],
            cache_dir=data_args.dataset_cache_dir,
            streaming=data_args.streaming,
        )
        if dataset_dict["text_column_name"] not in list(raw_datasets["eval"].features.keys()):
            raise ValueError(
                f"--text column name {dataset_dict['text_column_name']} not found in the evaluation "
                f"dataset {dataset_dict['name']}. Ensure `text_column_name` is set to the correct column "
                f"for the target text. Should be one of {' '.join(list(raw_datasets['eval'].features.keys()))}"
            )
        if dataset_dict["text_column_name"] != "text":
            raw_datasets["eval"] = raw_datasets["eval"].rename_column(dataset_dict["text_column_name"], "text")
    else:
        # load multiple eval sets
        for dataset_dict in tqdm(dataset_names_dict, desc="Loading datasets..."):
            # Clean-up the dataset name for pretty logging
            # ("distil-whisper/librispeech_asr", "validation.clean") -> "librispeech_asr/validation-clean"
            pretty_name = f"{dataset_dict['name'].split('/')[-1]}/{dataset_dict['split'].replace('.', '-')}"
            raw_datasets[pretty_name] = load_dataset(
                dataset_dict["name"],
                dataset_dict["config"],
                split=dataset_dict["split"],
                cache_dir=data_args.dataset_cache_dir,
                streaming=data_args.streaming,
            )
            if dataset_dict["text_column_name"] not in list(raw_datasets[pretty_name].features.keys()):
                raise ValueError(
                    f"`--text_column_name` {dataset_dict['text_column_name']} not found in the evaluation "
                    f"dataset {dataset_dict['name']}. Ensure `text_column_name` is set to the correct column "
                    f"for the target text. Should be one of {' '.join(list(raw_datasets[pretty_name].features.keys()))}"
                )
            if dataset_dict["text_column_name"] != "text":
                raw_datasets[pretty_name] = raw_datasets[pretty_name].rename_column(
                    dataset_dict["text_column_name"], "text"
                )

    # 4. Resample speech dataset: `datasets` takes care of automatically loading and resampling the audio,
    # so we just need to set the correct target sampling rate.
    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name,
        datasets.features.Audio(SAMPLING_RATE),
    )

    # 5. Load model & normalizer
    model_name = data_args.model_name_or_path.split("/")[-1].split("whisper-")[-1]

    model = whisper.load_model(model_name)
    model.cuda()

    normalizer = EnglishTextNormalizer()

    # 6. Run evaluation
    def evaluate(batch):
        # batch_size has to be 1 for openai/whisper
        raw_audio = batch[data_args.audio_column_name][0]["array"]
        raw_audio = raw_audio.astype(np.float32)

        # generate
        out_dict = model.transcribe(raw_audio, condition_on_previous_text=data_args.condition_on_prev_tokens, language="en")

        batch["transcription"] = [out_dict["text"]]
        batch["reference"] = batch["text"]

        return batch

    result_datasets = DatasetDict()
    for split in raw_datasets:
        map_fn = partial(
            raw_datasets[split].map,
            function=evaluate,
            remove_columns=raw_datasets[split].features.keys(),
            batch_size=1,
            batched=True,
        )

        result_datasets[split] = (
            map_fn(num_proc=1, desc="benchmark eval dataset")
            if not data_args.streaming
            else map_fn()
        )

    # 7. Compute WER and upload
    count = 0
    for split in result_datasets:
        transcriptions = []
        references = []
        all_wers = []

        if data_args.streaming:
            result_iter = iter(result_datasets[split])

        for result in result_iter:
            transcriptions.append(result["transcription"])
            references.append(result["reference"])
            try:
                all_wers.append(compute_metrics(transcriptions[-1:], references[-1:], normalizer))
            except:
                all_wers.append(None)

            count += 1
            print(f"Processed {count} samples...")

        log_stats = {
            f"{split}_wer": compute_metrics(transcriptions, references, normalizer),
            f"{split}_all_wer": all_wers,
        }
        wandb_logger.log(log_stats)

    print("Done!")


if __name__ == "__main__":
    main()
