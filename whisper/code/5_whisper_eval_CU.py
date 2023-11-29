#----------------------------------------------------------
# Purpose: Uses whisper to fine tune for kids speech
#          with children's speech corpus.
# Based on source:
# https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_tuning_Wav2Vec2_for_English_ASR.ipynb
# https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/fine_tune_whisper.ipynb#scrollTo=34d4360d-5721-426e-b6ac-178f833fedeb
# https://huggingface.co/openai/whisper-large-v2
# Author: Renee Lu, 2021
# Moddified: Jordan Chan, 2023
#----------------------------------------------------------

# ------------------------------------------
#      Install packages if needed
# ------------------------------------------
#pip install datasets==1.8.0
#pip install transformers
#pip install soundfile
#pip install jiwer

# ------------------------------------------
#       Import required packages
# ------------------------------------------
# For printing filepath
import os
# ------------------------------------------
print('Running: ', os.path.abspath(__file__))
# ------------------------------------------
# For accessing date and time
from datetime import date
from datetime import datetime
now = datetime.now()
# Print out dd/mm/YY H:M:S
# ------------------------------------------
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("Started:", dt_string)
# ------------------------------------------ 
print("\n------> IMPORTING PACKAGES.... ---------------------------------------\n")
print("-->Importing datasets...")
# Import datasets and evaluation metric
from datasets import load_dataset, load_metric, ClassLabel
# Convert pandas dataframe to DatasetDict
from datasets import Dataset
# Generate alignment for OOV check
print("-->Importing jiwer...")
import jiwer
# Generate random numbers
print("-->Importing random...")
import random
# Manipulate dataframes and numbers
print("-->Importing pandas & numpy...")
import pandas as pd
import numpy as np
# Use regex
print("-->Importing re...")
import re
# Read, Write, Open json files
print("-->Importing json...")
import json
# Convert numbers to words
print("-->Importing num2words...")
from num2words import num2words
print("-->Importing string...")
import string
print('Importing partial')
from functools import partial
# Use models and tokenizers
print("-->Importing Whisper Packages...")
from transformers import WhisperTokenizer
from transformers import WhisperForConditionalGeneration
from transformers import WhisperFeatureExtractor
from transformers import WhisperProcessor
# Loading audio files
print("-->Importing soundfile...")
import soundfile as sf
print("-->Importing librosa...")
import librosa
# For training
print("-->Importing torch, dataclasses & typing...")
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
print("-->Importing from transformers for training...")
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
print("-->Importing pyarrow for loading dataset...")
import pyarrow as pa
import pyarrow.csv as csv
print("-->SUCCESS! All packages imported.")

# ------------------------------------------
#      Setting experiment arguments
# ------------------------------------------
print("\n------> EXPERIMENT ARGUMENTS ----------------------------------------- \n")

base_fp = '/srv/scratch/z5313567/thesis/'
print('base_fp:', base_fp)

model = 'whisper'
print('model:', model)

dataset_name = 'CU'
print('dataset_name:', dataset_name)

experiment_id = 'whisper_eval_CU_lowercase_20231101_5'
print('experiment_id:', experiment_id)

cache_name = 'CU-eval'
print('cache_name:', cache_name)


# Perform Training (True/False)
# If false, this will go straight to model evaluation 
training = False
print("training:", training)

# Resume training from/ use checkpoint (True/False)
# Set to True for:
# 1) resuming from a saved checkpoint if training stopped midway through
# 2) for using an existing finetuned model for evaluation 
# If 2), then must also set eval_pretrained = True
use_checkpoint = True
print("use_checkpoint:", use_checkpoint)

# Set checkpoint if resuming from/using checkpoint
checkpoint = "/srv/scratch/z5313567/thesis/whisper/model/AusKidTalk/whisper_medium_finetune_AusKidTalk_scripted_spontaneous_combined_freeze_encoder_lowercase_20231030"
if use_checkpoint:
    print("checkpoint:", checkpoint)

# Use a pretrained tokenizer (True/False)
#     True: Use existing tokenizer (if custom dataset has same vocab)
#     False: Use custom tokenizer (if custom dataset has different vocab)
use_pretrained_tokenizer = True
print("use_pretrained_tokenizer:", use_pretrained_tokenizer)

# Set tokenizer
pretrained_tokenizer = "openai/whisper-small"
if use_pretrained_tokenizer:
    print("pretrained_tokenizer:", pretrained_tokenizer)

# Evaluate existing model instead of newly trained model (True/False)
#     True: use the model in the filepath set by 'eval_model' for eval
#     False: use the model trained from this script for eval
eval_pretrained = True
print("eval_pretrained:", eval_pretrained)

# Set existing model to evaluate, if evaluating on existing model
eval_model = checkpoint
if eval_pretrained:
    print("eval_model:", eval_model)

# Baseline model for evaluating baseline metric
# This model will be evaluated at the end for the baseline WER
baseline_model = "openai/whisper-small"
print("baseline_model:", baseline_model)

# Evalulate the baseline model or not (True/False)
#   True: evaluate baseline model on test set
#   False: do not evaluate baseline model on test set
eval_baseline = True
print("eval_baseline:", eval_baseline)

print("\n------> TRAINING ARGUMENTS... ----------------------------------------\n")
# For setting training_args = TrainingArguments()

set_learning_rate = 1e-05                 # Default = "5e-5"
print("learning_rate:", set_learning_rate)
set_per_device_train_batch_size = 8     # Default = 8
print("per_device_train_batch_size:", set_per_device_train_batch_size)
set_per_device_eval_batch_size = 32       # Default = 8
print("per_device_eval_batch_size:", set_per_device_eval_batch_size)
set_seed = 42                             # Default = 42
print("seed:", set_seed)
set_adam_beta1 = 0.9                      # Default = 0.9
print("adam_beta1:", set_adam_beta1)
set_adam_beta2 = 0.999                    # Default = 0.999
print("adam_beta2:", set_adam_beta2)
set_adam_epsilon = 1e-08                  # Default = 1e-08
print("adam_epsilon:", set_adam_epsilon)
set_lr_scheduler_type = "linear"          # Default = "linear"
print("lr_scheduler_type:", set_lr_scheduler_type)
set_warmup_steps = 500                    # Default = 0
print("warmup_steps:", set_warmup_steps)
set_max_steps = 5500                      # Default = 1
print("max_steps:", set_max_steps)
set_gradient_accumulation_steps = 1       # Default = 1
print("gradient_accumulation_steps:", set_gradient_accumulation_steps)
set_gradient_checkpointing = True         # Default = False
print("gradient_checkpointing:", set_gradient_checkpointing)
set_fp16 = True                           # Default = False
print("fp16:", set_fp16)
set_evaluation_strategy = "steps"         # Default = "no"
print("evaluation_strategy:", set_evaluation_strategy)
set_predict_with_generate = True          # Default = False
print("predict_with_generate:", set_predict_with_generate)
set_generation_max_length = 225           # Optional
print("generation_max_length:", set_generation_max_length)
set_save_steps = 1000                     # Default = 500
print("save_steps:", set_save_steps)
set_eval_steps = 1000                     # Optional
print("eval_steps:", set_eval_steps)
set_logging_steps = 500                  # Default = 500
print("logging_steps:", set_logging_steps)
set_load_best_model_at_end = True         # Default = False
print("load_best_model_at_end:", set_load_best_model_at_end)
set_metric_for_best_model = "wer"         # Optional
print("metric_for_best_model:", set_metric_for_best_model)
set_greater_is_better = False              # Optional
print("greater_is_better:", set_greater_is_better)
set_group_by_length = True                # Default = False
print("group_by_length:", set_group_by_length)


# ------------------------------------------
#        Generating file paths
# ------------------------------------------
print("\n------> GENERATING FILEPATHS... --------------------------------------\n")
# Path to dataframe csv for train dataset
data_train_fp = '/srv/scratch/z5313567/thesis/CU_local/CU_test.csv'
print("--> data_train_fp:", data_train_fp)

# Path to dataframe csv for test dataset
data_test_fp = '/srv/scratch/z5313567/thesis/CU_local/4h/CU_4h_test_dataframe_15sec_only_transcription_filepath.csv'
print("--> data_test_fp:", data_test_fp)

# Dataframe file 
# |-----------|---------------------|----------|---------|
# | file path | transcription_clean | duration | spkr_id |
# |-----------|---------------------|----------|---------|
# |   ...     |      ...            |  ..secs  | ......  |
# |-----------|---------------------|----------|---------|
# NOTE: The spkr_id column may need to be removed beforehand if
#       there appears to be a mixture between numerical and string ID's
#       due to this issue: https://github.com/apache/arrow/issues/4168
#       when calling load_dataset()

# Path to datasets cache
data_cache_fp = '/srv/scratch/chacmod/.cache/huggingface/datasets/' + cache_name
print("--> data_cache_fp:", data_cache_fp)

# Path to pretrained model cache
model_cache_fp = '/srv/scratch/z5313567/thesis/cache'
print("--> model_cache_fp:", model_cache_fp)

# Path to save vocab.json
vocab_fp =  base_fp + model + '/vocab/' + dataset_name + '/' + experiment_id + '_vocab.json'
print("--> vocab_fp:", vocab_fp)

# Path to save model
model_fp = base_fp + model + '/model/' + dataset_name + '/' + experiment_id
print("--> model_fp:", model_fp)

# Path to save baseline results output
baseline_results_fp = base_fp + model + '/baseline_result/' + dataset_name + '/'  + experiment_id + '_baseline_result.csv'
print("--> baseline_results_fp:", baseline_results_fp)

# Path to save baseline alignments between model predictions and references
baseline_alignment_results_fp = base_fp + model + '/baseline_result/' + dataset_name + '/'  + experiment_id + '_baseline_result.txt'
print("--> baseline_alignment_results_fp:", baseline_alignment_results_fp)

# Path to save finetuned results output
finetuned_results_fp = base_fp + model + '/finetuned_result/' + dataset_name + '/'  + experiment_id + '_finetuned_result.csv'
print("--> finetuned_results_fp:", finetuned_results_fp)

# Path to save finetuned alignments between model predictions and references
finetuned_alignment_results_fp = base_fp + model + '/finetuned_result/' + dataset_name + '/'  + experiment_id + '_finetuned_result.txt'
print("--> finetuned_alignment_results_fp:", finetuned_alignment_results_fp)

# Pre-trained checkpoint model
# For 1) Fine-tuning or
#     2) resuming training from pre-trained model
# If 1) must set use_checkpoint = False
# If 2)must set use_checkpoint = True
# Default model to fine-tune is facebook's model
pretrained_mod = "openai/whisper-small"
if use_checkpoint:
    pretrained_mod = checkpoint
print("--> pretrained_mod:", pretrained_mod)
# Path to pre-trained tokenizer
# If use_pretrained_tokenizer = True
if use_pretrained_tokenizer:
    print("--> pretrained_tokenizer:", pretrained_tokenizer)

# ------------------------------------------
#         Preparing dataset
# ------------------------------------------
# Run the following scripts to prepare data
# 1) Prepare data from kaldi file: 
# /srv/scratch/z5160268/2020_TasteofResearch/kaldi/egs/renee_thesis/s5/wav2vec_exp/data_prep.py
# 3) [Optional] Limit the files to certain duration:
# /srv/scratch/z5160268/2020_TasteofResearch/kaldi/egs/renee_thesis/s5/wav2vec_projects/data_getShortWavs.py
# 2) Split data into train and test:
# /srv/scratch/z5160268/2020_TasteofResearch/kaldi/egs/renee_thesis/s5/wav2vec_projects/data_split.py

print("\n------> PREPARING DATASET... ------------------------------------\n")
# Read the existing csv saved dataframes and
# load as a DatasetDict 
data = load_dataset('csv', 
                    data_files={'train': data_train_fp,
                                'test': data_test_fp},
                    cache_dir=data_cache_fp)
print("--> dataset...")
print(data)

# Display some random samples of the dataset
print("--> Printing some random samples...")
def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Picking more elements than in dataset"
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    df = pd.DataFrame(dataset[picks])
    print(df)
show_random_elements(data["train"], num_examples=5)
print("SUCCESS: Prepared dataset.")
# ------------------------------------------
#       Processing transcription
# ------------------------------------------
# Create vocab.json
# Extracting all distinct letters of train and test set
# and building vocab from this set of letters
print("\n------> PROCESSING TRANSCRIPTION... ---------------------------------------\n")
# Mapping function that concatenates all transcriptions
# into one long transcription and then transforms the
# string into a set of chars. Set batched=True to the 
# map(...) function so that the mapping function has access
# to all transcriptions at once.

#chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def process_transcription(batch):
    #batch["transcription_clean"] = re.sub(chars_to_ignore_regex, '', batch["transcription_clean"]).upper()
    batch["transcription_clean"] = batch["transcription_clean"].lower()
    batch["transcription_clean"] = batch["transcription_clean"].replace("<UNK>", "<unk>")
    return batch

data = data.map(process_transcription)

def extract_all_chars(batch):
    all_text = " ".join(batch["transcription_clean"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}
    

print("\n------> Defining feature extractor... ---------------------------------------\n")
feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_mod, cache_dir=model_cache_fp)
print("SUCCESS: Feature extractor defined.")

print("\n------> Defining tokenizer... ---------------------------------------\n")
tokenizer = WhisperTokenizer.from_pretrained(pretrained_mod, language="english", task="transcribe", cache_dir=model_cache_fp)
print("SUCCESS: Tokenizer defined.")

print("\n------> Preparaing processor... ---------------------------------------\n")
processor = WhisperProcessor.from_pretrained(pretrained_mod, language="english", task="transcribe", cache_dir=model_cache_fp)
processor.save_pretrained(model_fp)
print("SUCCESS: Processor defined.")


# ------------------------------------------
#             Pre-process Data
# ------------------------------------------
print("\n------> PRE-PROCESSING DATA... ----------------------------------------- \n")
# Audio files are stored as .wav format
# We want to store both audio values and sampling rate
# in the dataset. 
# We write a map(...) function accordingly.

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = librosa.load(batch['filepath'], sr=feature_extractor.sampling_rate)
    batch["speech"] = speech_array
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["transcription_clean"]
    return batch

data = data.map(speech_file_to_array_fn, remove_columns=data.column_names["train"], num_proc=4)
# Check a few rows of data to verify data properly loaded
print("--> Verifying data with a random sample...")
rand_int = random.randint(0, len(data["train"])-1)
print("Target text:", data["train"][rand_int]["target_text"])
print("Input array shape:", np.asarray(data["train"][rand_int]["speech"]).shape)
print("Sampling rate:", data["train"][rand_int]["sampling_rate"])
# Process dataset to the format expected by model for training
# Using map(...)
# 1) Check all data samples have same sampling rate (16kHz)
# 2) Extract input_values from loaded audio file.
#    This only involves normalisation but could also correspond
#    to extracting log-mel features
# 3) Encode the transcriptions to label ids

def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
    
    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_features
    
    # encode target text to label ids
    batch["labels"] = tokenizer(batch["target_text"]).input_ids
    return batch
data_prepared = data.map(prepare_dataset, remove_columns=data.column_names["train"], batch_size=8, num_proc=4, batched=True)

print("SUCCESS: Data ready for training and evaluation.")

# ------------------------------------------
#         Training & Evaluation
# ------------------------------------------
# Set up the training pipeline using HuggingFace's Trainer:
# 1) Define a data collator: Wav2Vec has much larger input
#    length than output length. Therefore, it is more
#    efficient to pad the training batches dynamically meaning
#    that all training samples should only be padded to the longest
#    sample in their batch and not the overall longest sample.
#    Therefore, fine-tuning Wav2Vec2 required a special 
#    padding data collator, defined below.
# 2) Evaluation metric: we evaluate the model using word error rate (WER)
#    We define a compute_metrics function accordingly.
# 3) Load a pre-trained checkpoint
# 4) Define the training configuration
print("\n------> PREPARING FOR TRAINING & EVALUATION... ----------------------- \n")
# 1) Defining data collator
print("--> Defining data collator...")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
print("SUCCESS: Data collator defined.")

# 2) Evaluation metric
#    Using word error rate (WER)
print("--> Defining evaluation metric...")
# The model will return a sequence of logit vectors y.
# A logit vector yi contains the log-odds for each word in the
# vocabulary defined earlier, thus len(yi) = config.vocab_size
# We are interested in the most likely prediction of the mode and 
# thus take argmax(...) of the logits. We also transform the
# encoded labels back to the original string by replacing -100
# with the pad_token_id and decoding the ids while making sure
# that consecutive tokens are not grouped to the same token in
# CTC style.
wer_metric = load_metric("wer")
cer_metric = load_metric("cer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
    
print("SUCCESS: Defined WER evaluation metric.")

# 3) Load pre-trained checkpoint
# Load pre-trained Wav2Vec2 checkpoint. The tokenizer's pad_token_id
# must be to define the model's pad_token_id or in the case of Wav2Vec2ForCTC
# also CTC's blank token. To save GPU memory, we enable PyTorch's gradient
# checkpointing and also set the loss reduction to "mean".
print("--> Loading pre-trained checkpoint...")
model = WhisperForConditionalGeneration.from_pretrained(pretrained_mod, cache_dir=model_cache_fp)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


# 4) Configure training parameters
#    - group_by_length: makes training more efficient by grouping
#      training samples of similar input length into one batch.
#      Reduces useless padding tokens passed through model.
#    - learning_rate and weight_decay: heuristically tuned until
#      fine-tuning has become stable. These paramteres strongly
#      depend on Timit dataset and might be suboptimal for this
#      dataset.
# For more info: https://huggingface.co/transformers/master/main_classes/trainer.html?highlight=trainer#trainingarguments

training_args = Seq2SeqTrainingArguments(
  output_dir=model_fp,
  learning_rate=set_learning_rate,
  per_device_train_batch_size=set_per_device_train_batch_size,
  per_device_eval_batch_size=set_per_device_eval_batch_size,
  seed=set_seed,
  adam_beta1=set_adam_beta1,
  adam_beta2=set_adam_beta2,
  adam_epsilon=set_adam_epsilon,
  lr_scheduler_type=set_lr_scheduler_type,
  warmup_steps=set_warmup_steps,
  max_steps=set_max_steps,
  gradient_accumulation_steps=set_gradient_accumulation_steps,
  gradient_checkpointing=set_gradient_checkpointing,
  fp16=set_fp16,
  evaluation_strategy=set_evaluation_strategy,
  predict_with_generate=set_predict_with_generate,
  generation_max_length=set_generation_max_length,
  save_steps=set_save_steps,
  eval_steps=set_eval_steps,
  logging_steps=set_logging_steps,
  load_best_model_at_end=set_load_best_model_at_end,
  metric_for_best_model=set_metric_for_best_model,
  greater_is_better=set_greater_is_better,
  group_by_length = set_group_by_length,
)
# All instances can be passed to Trainer and 
# we are ready to start training!
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=data_prepared["train"],
    eval_dataset=data_prepared["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# ------------------------------------------
#               Training
# ------------------------------------------
# While the trained model yields a satisfying result on Timit's
# test data, it is by no means an optimally fine-tuned model
# for children's data.

if training:
    print("\n------> STARTING TRAINING... ----------------------------------------- \n")
    torch.cuda.empty_cache()
    # Train
    if use_checkpoint:
        trainer.train(pretrained_mod)
    else:
        trainer.train()
    # Save the model
    model.save_pretrained(model_fp)
print("\n------> Training finished... ------------------------------------------ \n")

# ------------------------------------------
#            Evaluation
# ------------------------------------------
# Evaluate fine-tuned model on test set.
print("\n------> EVALUATING MODEL... ------------------------------------------ \n")
torch.cuda.empty_cache()

if eval_pretrained:
    processor = WhisperProcessor.from_pretrained(eval_model, cache_dir=model_cache_fp)
    model = WhisperForConditionalGeneration.from_pretrained(eval_model, cache_dir=model_cache_fp)
else:
    processor = WhisperProcessor.from_pretrained(model_fp, cache_dir=model_cache_fp)
    model = WhisperForConditionalGeneration.from_pretrained(model_fp, cache_dir=model_cache_fp)
model.config.forced_decoder_ids = None
# Now, we will make use of the map(...) function to predict 
# the transcription of every test sample and to save the prediction 
# in the dataset itself. We will call the resulting dictionary "results".
# Note: we evaluate the test data set with batch_size=1 on purpose due 
# to this issue (https://github.com/pytorch/fairseq/issues/3227). Since 
# padded inputs don't yield the exact same output as non-padded inputs, 
# a better WER can be achieved by not padding the input at all.
def map_to_result(batch):
    model.to("cuda")
    input_features = processor(
        batch["speech"], 
        sampling_rate=batch["sampling_rate"], 
        return_tensors="pt"
    ).input_features.to("cuda")

    with torch.no_grad():
        # generate token ids
        predicted_ids = model.generate(input_features.to("cuda"))
    batch["pred_str"] = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return batch

def post_process(results):
    pred_str = results['pred_str']
    
    # make all the characters lowercase
    pred_str = pred_str.lower()
    
    # remoce symbols and punctuation
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    pred_str = re.sub(chars_to_ignore_regex, "", pred_str)
    
    # add a whitespace before each number
    pred_str = re.sub(r'(\d)', r' \1', pred_str)
    
    #convert numerical numbers to English characters
    pred_str = ' '.join([num2words(word) if word.isdigit() else word for word in pred_str.split()])
    
    # remove extra whitespace between words
    pred_str = re.sub(r'\s+', ' ', pred_str)
    
    # remove leading and trailing whitespaces
    pred_str = pred_str.strip()
    
    results['pred_str'] = pred_str
    return results
    
results = data["test"].map(map_to_result)
results = results.map(post_process)

# Save results to csv
results_df = results.to_pandas()
results_df = results_df.drop(columns=['speech', 'sampling_rate'])
results_df.to_csv(finetuned_results_fp)
print("Saved results to:", finetuned_results_fp)

# Getting the WER and CER
print("--> Getting fine-tuned test results...")
print("Fine-tuned Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], 
      references=results["target_text"])))
print("Fine-tuned Test CER: {:.3f}".format(cer_metric.compute(predictions=results["pred_str"], 
      references=results["target_text"])))
print('\n')

print("--> Getting finetuned alignment output...")
word_output = jiwer.process_words(results["target_text"], results["pred_str"])
alignment = jiwer.visualize_alignment(word_output)
output_text = "--> Getting the finetuned alignment result...\n\n" + alignment + '\n\n\n'

with open(finetuned_alignment_results_fp, 'w') as output_file:
    output_file.write(output_text)
print("Saved Alignment output to:", finetuned_alignment_results_fp)
print('\n')

# Showing prediction errors
print("--> Showing some fine-tuned prediction errors...")
show_random_elements(results.remove_columns(["speech", "sampling_rate"]))
# Deeper look into model: running the first test sample through the model, 
# take the predicted ids and convert them to their corresponding tokens.
print("--> Taking a deeper look...")
model.to("cuda")
input_features = processor(data["test"][0]["speech"], sampling_rate=data["test"][0]["sampling_rate"], return_tensors="pt").input_features.to("cuda")
# generate token ids
predicted_ids = model.generate(input_features)
# convert ids to tokens
print(" ".join(processor.tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())))

# Evaluate baseline model on test set if eval_baseline = True
if eval_baseline:
    print("\n------> EVALUATING BASELINE MODEL... ------------------------------------------ \n")
    torch.cuda.empty_cache()
    processor = WhisperProcessor.from_pretrained(baseline_model, cache_dir=model_cache_fp)
    model = WhisperForConditionalGeneration.from_pretrained(baseline_model, cache_dir=model_cache_fp)
    tokenizer = WhisperTokenizer.from_pretrained(baseline_model, cache_dir=model_cache_fp)

    # Now, we will make use of the map(...) function to predict 
    # the transcription of every test sample and to save the prediction 
    # in the dataset itself. We will call the resulting dictionary "results".
    # Note: we evaluate the test data set with batch_size=1 on purpose due 
    # to this issue (https://github.com/pytorch/fairseq/issues/3227). Since 
    # padded inputs don't yield the exact same output as non-padded inputs, 
    # a better WER can be achieved by not padding the input at all.
    results = data["test"].map(map_to_result)
    results = results.map(post_process)
    
    # Saving results to csv
    results_df = results.to_pandas()
    results_df = results_df.drop(columns=['speech', 'sampling_rate'])
    results_df.to_csv(baseline_results_fp)
    print("Saved results to:", baseline_results_fp)
    # Getting the WER
    print("--> Getting baseline test results...")
    print("Baseline Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], 
          references=results["target_text"])))
    print("Baseline Test CER: {:.3f}".format(cer_metric.compute(predictions=results["pred_str"], 
          references=results["target_text"])))
    print('\n')
    
    print("--> Getting baseline alignment output...")
    word_output = jiwer.process_words(results["target_text"], results["pred_str"])
    alignment = jiwer.visualize_alignment(word_output)
    output_text = "--> Getting the baseline alignment result...\n\n" + alignment + '\n\n\n'

    with open(baseline_alignment_results_fp, 'w') as output_file:
        output_file.write(output_text)
    print("Saved Alignment output to:", baseline_alignment_results_fp)
    print('\n')
    
    # Showing prediction errors
    print("--> Showing some baseline prediction errors...")
    show_random_elements(results.remove_columns(["speech", "sampling_rate"]))
    # Deeper look into model: running the first test sample through the model, 
    # take the predicted ids and convert them to their corresponding tokens.
    print("--> Taking a deeper look...")
    model.to("cuda")
    input_features = processor(data["test"][0]["speech"], sampling_rate=data["test"][0]["sampling_rate"], return_tensors="pt").input_features.to("cuda")
    # generate token ids
    predicted_ids = model.generate(input_features)
    # convert ids to tokens
    print(" ".join(processor.tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())))

print("\n------> SUCCESSFULLY FINISHED ---------------------------------------- \n")
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("Finished:", dt_string)