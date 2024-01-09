#!/usr/bin/env bash
# names=("openai/whisper-small.en" "openai/whisper-base.en" "openai/whisper-tiny.en" "openai/whisper-large" "openai/whisper-medium")
names=("openai/whisper-small" "openai/whisper-base" "openai/whisper-tiny" "openai/whisper-large-v2" "openai/whisper-medium.en")
# names=("openai/whisper-small.en")
names=("openai/whisper-medium" "openai/whisper-medium.en")
names=("openai/whisper-large-v2")

# chunk_lengths=("15.0" "30.0")
# --assistant_model_name_or_path "patrickvonplaten/whisper-large-v2-32-2" \
# --attn_type "flash" \
#
#    --dataset_name "distil-whisper/meanwhile" \
#    --dataset_config_name "default" \
#    --dataset_split_name "test" \
#    --text_column_name "text" \
#
#    --dataset_name "distil-whisper/earnings21" \
#    --dataset_config_name "full" \
#    --dataset_split_name "test" \
#    --text_column_name "transcription" \


#    --dataset_name "distil-whisper/meanwhile+distil-whisper/earnings21+distil-whisper/earnings22+distil-whisper/rev16" \
#    --dataset_config_name "default+full+full+whisper_subset" \
#    --dataset_split_name "test+test+test+test" \
#    --text_column_name "text+transcription+transcription+transcription" \


#    --dataset_name "distil-whisper/meanwhile+distil-whisper/earnings21+distil-whisper/earnings22+distil-whisper/rev16" \
#    --dataset_config_name "default+full+full+whisper_subset" \
#    --dataset_split_name "test+test+test+test" \
#    --text_column_name "text+transcription+transcription+transcription" \

# --use_pipeline \
# Double loop
# --use_orig_whisper \

for name in "${names[@]}"; do
  python ./run_whisper_transformers.py \
    --dataset_name "distil-whisper/meanwhile+distil-whisper/earnings21+distil-whisper/earnings22+distil-whisper/rev16" \
    --dataset_config_name "default+full+full+whisper_subset" \
    --dataset_split_name "test+test+test+test" \
    --text_column_name "text+transcription+transcription+transcription" \
    --wandb_name "RTX4090-${name}-cond-false" \
    --model_name_or_path ${name} \
    --wandb_project "whisper-long-form-transformers" \
    --condition_on_prev_tokens "False" \
    --use_fp16 "True"
done
