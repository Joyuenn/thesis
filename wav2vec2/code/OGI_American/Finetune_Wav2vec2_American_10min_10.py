#----------------------------------------------------------
# Purpose: Uses wav2vec2 to fine tune for kids speech
#          with children's speech corpus.
# Based on source:
# https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_tuning_Wav2Vec2_for_English_ASR.ipynb
# Author: Renee Lu, 2021
# Moddified: Jordan Chan, 2023
#----------------------------------------------------------

# pip install datasets
# pip install transformers==4.29.2
# pip install jiwer
# pip install torch

from datetime import date
from datetime import datetime

now = datetime.now()
# Print out dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("Started:", dt_string)

import os 
current_file = os.path.basename(__file__)
print("Currently executing file:", current_file)

print("\n------------ Importing libraries... ------------\n")
import pandas as pd
import random
import json
import librosa
import torch
import numpy as np

import transformers
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer

from datasets import load_dataset, load_metric, ClassLabel, Audio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

print("pandas version:", pd.__version__)
print("json version:", json.__version__)
print("librosa version:", librosa.__version__)
print("Numpy version:", np.__version__)
print("Transformers version:", transformers.__version__)
print("Torch version:", torch.__version__)
print("Test cuda_device_count", torch.cuda.device_count())
print("Test cuda_is_available", torch.cuda.is_available())
print("Test get_device_name", torch.cuda.get_device_name(0))

'''
print("\n------> MODEL ARGUMENTS... -------------------------------------------\n")
# For setting model = Wav2Vec2ForCTC.from_pretrained()

set_hidden_dropout = 0.1                    # Default = 0.1
print("hidden_dropout:", set_hidden_dropout)
set_activation_dropout = 0.1                # Default = 0.1
print("activation_dropout:", set_activation_dropout)
set_attention_dropout = 0.1                 # Default = 0.1
print("attention_dropoutput:", set_attention_dropout)
set_feat_proj_dropout = 0.0                 # Default = 0.1
print("feat_proj_dropout:", set_feat_proj_dropout)
set_layerdrop = 0.1                       # Default = 0.1
print("layerdrop:", set_layerdrop)
set_mask_time_prob = 0.075                  # Default = 0.05
print("mask_time_prob:", set_mask_time_prob)
set_mask_time_length = 10                   # Default = 10
print("mask_time_length:", set_mask_time_length)
set_ctc_loss_reduction = "mean"             # Default = "sum"
print("ctc_loss_reduction:", set_ctc_loss_reduction)
set_ctc_zero_infinity = False               # Default = False
print("ctc_zero_infinity:", set_ctc_zero_infinity)
set_gradient_checkpointing = True           # Default = False
print("gradient_checkpointing:", set_gradient_checkpointing)

print("\n------> TRAINING ARGUMENTS... ----------------------------------------\n")
# For setting training_args = TrainingArguments()

set_evaluation_strategy = "steps"           # Default = "no"
print("evaluation strategy:", set_evaluation_strategy)
set_per_device_train_batch_size = 8         # Default = 8
print("per_device_train_batch_size:", set_per_device_train_batch_size)
set_gradient_accumulation_steps = 1         # Default = 1
print("gradient_accumulation_steps:", set_gradient_accumulation_steps)
set_learning_rate = 0.00005                 # Default = 0.00005
print("learning_rate:", set_learning_rate)
set_weight_decay = 0.01                     # Default = 0
print("weight_decay:", set_weight_decay)
set_adam_beta1 = 0.9                        # Default = 0.9
print("adam_beta1:", set_adam_beta1)
set_adam_beta2 = 0.98                       # Default = 0.999
print("adam_beta2:", set_adam_beta2)
set_adam_epsilon = 0.00000001               # Default = 0.00000001
print("adam_epsilon:", set_adam_epsilon)
set_num_train_epochs = 2000                   # Default = 3.0
print("num_train_epochs:", set_num_train_epochs)
set_max_steps = 12000                       # Default = -1, overrides epochs
print("max_steps:", set_max_steps)
set_lr_scheduler_type = "linear"            # Default = "linear"
print("lr_scheduler_type:", set_lr_scheduler_type )
set_warmup_ratio = 0.1                      # Default = 0.0
print("warmup_ratio:", set_warmup_ratio)
set_logging_strategy = "steps"              # Default = "steps"
print("logging_strategy:", set_logging_strategy)
set_logging_steps = 500                      # Default = 500
print("logging_steps:", set_logging_steps)
set_save_strategy = "steps"                 # Default = "steps"
print("save_strategy:", set_save_strategy)
set_save_steps = 500                         # Default = 500
print("save_steps:", set_save_steps)
set_save_total_limit = 2                   # Optional                 
print("save_total_limit:", set_save_total_limit)
set_fp16 = True                             # Default = False
print("fp16:", set_fp16)
set_eval_steps = 500                         # Optional
print("eval_steps:", set_eval_steps)
#set_load_best_model_at_end = True           # Default = False
#print("load_best_model_at_end:", set_load_best_model_at_end)
#set_metric_for_best_model = "wer"           # Optional
#print("metric_for_best_model:", set_metric_for_best_model)
#set_greater_is_better = False               # Optional
#print("greater_is_better:", set_greater_is_better)
set_group_by_length = True                  # Default = False
print("group_by_length:", set_group_by_length)
'''

'''
print("\n------------------ Model arguments... ------------------\n")
# For setting model = Wav2Vec2ForCTC.from_pretrained()

set_hidden_dropout = 0.1                    # Default = 0.1
print("hidden_dropout:", set_hidden_dropout)
set_activation_dropout = 0.1                # Default = 0.1
print("activation_dropout:", set_activation_dropout)
set_attention_dropout = 0.1                 # Default = 0.1
print("attention_dropout:", set_attention_dropout)
set_feat_proj_dropout = 0.0                 # Default = 0.1
print("feat_proj_dropout:", set_feat_proj_dropout)
set_layerdrop = 0.01                        # Default = 0.1
print("layerdrop:", set_layerdrop)
set_mask_time_prob = 0.075                  # Default = 0.05
print("mask_time_prob:", set_mask_time_prob)
set_mask_time_length = 10                   # Default = 10
print("mask_time_length:", set_mask_time_length)
set_ctc_loss_reduction = "mean"             # Default = "sum"
print("ctc_loss_reduction:", set_ctc_loss_reduction)
set_ctc_zero_infinity = True               # Default = False
print("ctc_zero_infinity:", set_ctc_zero_infinity)


print("\n------------------ Training arguments... ------------------\n")
# For setting training_args = TrainingArguments()

set_per_device_train_batch_size = 4         # Default = 8
print("per_device_train_batch_size:", set_per_device_train_batch_size)
set_group_by_length = True                  # Default = False
print("group_by_length:", set_group_by_length)
set_gradient_accumulation_steps = 1         # Default = 1
print("gradient_accumulation_steps:", set_gradient_accumulation_steps)
set_gradient_checkpointing = True           # Default = False
print("gradient_checkpointing:", set_gradient_checkpointing)
set_weight_decay = 0.01                     # Default = 0
print("weight_decay:", set_weight_decay)
set_fp16 = True                             # Default = False
print("fp16:", set_fp16)

set_learning_rate = 0.00001                 # Default = 0.00005
print("learning_rate:", set_learning_rate)
set_lr_scheduler_type = "linear"            # Default = "linear"
print("lr_scheduler_type:", set_lr_scheduler_type )
set_adam_beta1 = 0.9                        # Default = 0.9
print("adam_beta1:", set_adam_beta1)
set_adam_beta2 = 0.98                       # Default = 0.999
print("adam_beta2:", set_adam_beta2)
set_adam_epsilon = 0.00000001               # Default = 0.00000001
print("adam_epsilon:", set_adam_epsilon)
set_warmup_ratio = 0.2                     # Default = 0.0
print("warmup_ratio:", set_warmup_ratio)

set_num_train_epochs = 2000                 # Default = 3.0
print("num_train_epochs:", set_num_train_epochs)
set_max_steps = 12000                       # Default = -1, overrides epochs
print("max_steps:", set_max_steps)

set_logging_strategy = "steps"              # Default = "steps"
print("logging_strategy:", set_logging_strategy)
set_logging_steps = 1000                      # Default = 500
print("logging_steps:", set_logging_steps)
set_save_strategy = "steps"                 # Default = "steps"
print("save_strategy:", set_save_strategy)
set_save_steps = 1000                         # Default = 500
print("save_steps:", set_save_steps)
set_evaluation_strategy = "steps"           # Default = "no"
print("evaluation_strategy:", set_evaluation_strategy)
set_eval_steps = 1000                         # Optional
print("eval_steps:", set_eval_steps)
set_save_total_limit = 30                   # Optional                 
print("save_total_limit:", set_save_total_limit)

set_load_best_model_at_end = True           # Default = False
print("load_best_model_at_end:", set_load_best_model_at_end)
set_metric_for_best_model = "wer"           # Optional
print("metric_for_best_model:", set_metric_for_best_model)
set_greater_is_better = False               # Optional
print("greater_is_better:", set_greater_is_better)
'''


print("\n------------------ Model arguments... ------------------\n")
# For setting model = Wav2Vec2ForCTC.from_pretrained()
set_ctc_loss_reduction = "mean"
print('ctc_loss_reduction:', set_ctc_loss_reduction)

print("\n------------------ Training arguments... ------------------\n")
# For setting training_args = TrainingArguments()
set_group_by_length = True
print('group_by_length:', set_group_by_length)
set_per_device_train_batch_size = 32
print('per_device_train_batch_size:', set_per_device_train_batch_size)
set_warmup_ratio = 0.1
print('warmup_ratio:', set_warmup_ratio)
set_learning_rate = 0.0001
print('learning_rate:', set_learning_rate)
set_max_steps = 12000
print('max_steps:', set_max_steps)
set_weight_decay = 0.005
print('weight_decay:', set_weight_decay)
set_evaluation_strategy = 'steps'
print('evaluation_strategy:', set_evaluation_strategy)
#set_num_train_epochs = 30
#print('num_train_epochs:', set_num_train_epochs)
set_fp16 = True
print('fp16:', set_fp16)
set_gradient_checkpointing = True
print('gradient_checkpointing:', set_gradient_checkpointing)
set_save_steps = 500
print('save_steps:', set_save_steps)
set_eval_steps = 500
print('eval_steps:', set_eval_steps)
set_logging_steps = 500
print('logging_steps:', set_logging_steps)
set_save_total_limit = 2
print('save_total_limit:', set_save_total_limit)


print("\n------------------ Experiment arguments... ------------------\n")
use_checkpoint = False
print('use_checkpoint:', use_checkpoint)

training = True
print('training:', training)


print("\n------------------ Loading files... ------------------\n")
train_df_fp = '/srv/scratch/z5313567/thesis/OGI_local/new_10min_datasets/10min_OGI_scripted_train_dataframe.csv'
print(f'Training dataset is stored at {train_df_fp}\n')
#dev_df_fp = '/srv/scratch/z5313567/thesis/OGI_local/new_10min_datasets/10min_OGI_scripted_dev_dataframe.csv'
#print(f'Development dataset is stored at {dev_df_fp}\n')
test_df_fp = '/srv/scratch/z5313567/thesis/OGI_local/OGI_scripted_test_dataframe.csv'
print(f'Testing dataset is stored at {test_df_fp}\n')

cache_fp = '/srv/scratch/chacmod/.cache/huggingface/datasets/Jordan-OGI-finetune'
print(f'Cache filepath is {cache_fp}\n')
model_fp = '/srv/scratch/z5313567/thesis/wav2vec2/model/OGI_American/10min/10min_model_OGI_American_20230621'
print(f'Model filepath is {model_fp}\n')
vocab_fp = '/srv/scratch/z5313567/thesis/wav2vec2/vocab/OGI_American/10min/10min_vocab_OGI_American_20230621.json'
print(f'Vocab filepath is {vocab_fp}\n')
finetuned_result_fp = '/srv/scratch/z5313567/thesis/wav2vec2/finetuned_result/OGI_American/10min/10min_result_OGI_American_20230621.csv'
print(f'Fine-tuned result filepath is {finetuned_result_fp}\n')

# use_checkpoint = False -----> pretrained_mod = 'facebook/wav2vec2-base'
# use_checkpoint = True  -----> pretrained_mod = checkpoint_dir
pretrained_mod = 'facebook/wav2vec2-base'
if use_checkpoint:
    checkpoint_dir = '/srv/scratch/z5313567/thesis/wav2vec2/model/OGI_American/10min/10min_model_OGI_American_20230620/checkpoint-11000'
    pretrained_mod = checkpoint_dir
    print(f'Checkpoint directory is {checkpoint_dir}\n')
print(f'Pretrained model is {pretrained_mod}\n')



print("\n------------------ Loading datasets... ------------------\n")

dataset = load_dataset('csv', 
                    data_files={'train': train_df_fp,
                                #'dev': dev_df_fp,
                                'test': test_df_fp},
                    cache_dir=cache_fp)
print('dataset is:', dataset)


print("\n------------------ Showing some random elements... ------------------\n")

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), f"Can't pick more elements than there are in the dataset {len(dataset)}." 
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    print(df)
show_random_elements(dataset["train"].remove_columns(["filepath", "duration", "speaker_id"]), num_examples=10)


print("\n------------------ Extracting individual characters... ------------------\n")
def extract_all_chars(batch):
    all_text = " ".join(batch["transcription"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}
vocabs = dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset.column_names["train"])
print('vocabs is:', vocabs)
vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
print("vocab_dict is:", vocab_dict)
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
print(f'The lendth of vocab_list is {len(vocab_dict)}')

with open(vocab_fp, 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)
print('Successfully created vocab.json file at:', vocab_fp)


print("\n------------------ Creating tokenizer... ------------------\n")
tokenizer = Wav2Vec2CTCTokenizer(vocab_fp, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")


print("\n------------------ Creating feature extractor... ------------------\n")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)


print("\n------------------ Creating processor... ------------------\n")
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
processor.save_pretrained(model_fp)


def process_dataset(batch):
    batch['speech'], batch['sampling_rate'] = librosa.load(batch['filepath'], sr=feature_extractor.sampling_rate)
    batch["target_text"] = batch["transcription"]
    return batch
print("\n------------------ Obtaining speech arrays, sampling rates, and target texts... ------------------\n")
dataset = dataset.map(process_dataset, remove_columns=dataset.column_names["train"], num_proc=4)
print('dataset is:', dataset)


# the sampling rate of the data that was used to pretrain the model should match the sampling rate of the dataset used to fine-tune the model.
def check_sampling_rate(dataset): # say, dataset = dataset['train']
    target_sr = feature_extractor.sampling_rate
    if len(set(dataset['sampling_rate'])) == 1:
        actual_sr = list(set(dataset['sampling_rate']))[0] # sampling rate = 22050 Hz for OGI corpus
        if actual_sr != target_sr:
            print('MISMATCH!: the sampling rate used for fine-tuning Wav2vec2.0 does not match the sampling rate used for pretraining')
            
            now = datetime.now()
            # Print out dd/mm/YY H:M:S
            # ------------------------------------------
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            print('Resampling starts...', dt_string)
            for i in range(len(dataset)):
                dataset[i]['speech'] = librosa.resample(np.asarray(dataset[i]['speech']), orig_sr=actual_sr, target_sr=target_sr)
            now = datetime.now()
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            print('Resampling completes', dt_string)
        else:
            print('MATCH!: The sampling rate used for fine-tuning Wav2vec2.0 matches the sampling rate used for pretraining')    

print('\n------------------Ckech sampling rates of training datasets... ------------------\n')
check_sampling_rate(dataset['train'])

print('\n------------------Ckeck sampling rates of development datasets... ------------------\n')
#check_sampling_rate(dataset['dev'])

print('\n------------------Ckech sampling rates of testing datasets... ------------------\n')
check_sampling_rate(dataset['test'])


print("\n------------------ Verifying some ramdom samples... ------------------\n")
# verify if the data is a 1-dimensional array, the sampling rate corresponds to 16kHz, and the target text is clean.
for i in range(6):
    rand_int = random.randint(0, len(dataset["train"])-1)
    #rand_int = i
    print('Target text:', dataset['train'][rand_int]['target_text'])
    print('Input array shape:', np.asarray(dataset['train'][rand_int]['speech']).shape)
    print('Sampling rate:', dataset['train'][rand_int]["sampling_rate"])
    print('\n')
    
    
def prepare_dataset(batch):
    batch["input_values"] = processor(batch['speech'], sampling_rate=batch["sampling_rate"]).input_values[0]
    # batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch
print("\n------------------ Obtaining input values, input length, and labels... ------------------\n")
#prep_dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=4)
# dataset = dataset.map(prepare_dataset, num_proc=4)
data_prepared = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=4)
print('data_prepared is:', data_prepared)


#---------------Training + Evaluation--------------------
# ** Define a padding data collator used to pad the training samples to longest sample in their batch. In contrast to most NLP models,
#    Wav2Vec2 has a much larger input length than output length. E.g., a sample of input length 50000 has an output length of no more than 100. 
#    Given the large input sizes, it is much more efficient to pad the training batches dynamically meaning that all training samples should only 
#    be padded to the longest sample in their batch and not the overall longest sample. Therefore, fine-tuning Wav2Vec2 requires a special padding data collator, 
#    which we will define below
# ** Evaluation metric. During training, the model should be evaluated on the word error rate. We should define a compute_metrics function accordingly
# ** Load a pretrained checkpoint. We need to load a pretrained checkpoint and configure it correctly for training.
# ** Define the training configuration.

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    '''
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    
    '''
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        '''
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )
        '''
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )
        
        # replace label ids of padding tokens with -100 so that those tokens are not taken into account when computing the loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
    

print('\n------------------ Setting up the padding data collator... ------------------\n')
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


print('\n------------------ Setting up WER metric... ------------------\n')
wer_metric = load_metric("wer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

'''
print('\n------------------ Loading a pretrained checkpount... ------------------\n')
model = Wav2Vec2ForCTC.from_pretrained(
    pretrained_mod, 
    vocab_size=len(processor.tokenizer),
    pad_token_id=processor.tokenizer.pad_token_id,
    hidden_dropout=set_hidden_dropout,
    activation_dropout=set_activation_dropout,
    attention_dropout=set_attention_dropout,
    feat_proj_dropout=set_feat_proj_dropout,
    layerdrop=set_layerdrop,
    mask_time_prob=set_mask_time_prob,
    mask_time_length=set_mask_time_length,
    ctc_loss_reduction=set_ctc_loss_reduction,
    ctc_zero_infinity=set_ctc_zero_infinity    
)
'''


print('\n------------------ Loading a pretrained checkpount... ------------------\n')
model = Wav2Vec2ForCTC.from_pretrained(
    pretrained_mod, 
    ctc_loss_reduction=set_ctc_loss_reduction,
    pad_token_id=processor.tokenizer.pad_token_id    
)


# CNN layers of Wav2vec2.0 model is sufficiently trained, hence they do not need to be finetuned anymore
model.freeze_feature_encoder()

'''
print('\n------------------ Setting TrainingArguments... ------------------\n')
training_args = TrainingArguments(
    output_dir=model_fp,
    per_device_train_batch_size=set_per_device_train_batch_size,
    group_by_length=set_group_by_length,
    gradient_accumulation_steps=set_gradient_accumulation_steps,
    gradient_checkpointing=set_gradient_checkpointing,
    weight_decay=set_weight_decay,
    fp16=set_fp16,
    learning_rate=set_learning_rate,
    lr_scheduler_type=set_lr_scheduler_type,
    adam_beta1=set_adam_beta1,
    adam_beta2=set_adam_beta2,
    adam_epsilon=set_adam_epsilon,
    warmup_ratio=set_warmup_ratio,
    num_train_epochs=set_num_train_epochs,
    max_steps=set_max_steps,
    logging_strategy=set_logging_strategy,
    logging_steps=set_logging_steps,
    save_strategy=set_save_strategy,
    save_steps=set_save_steps,
    evaluation_strategy=set_evaluation_strategy,
    eval_steps=set_eval_steps,
    save_total_limit=set_save_total_limit,
    #load_best_model_at_end=set_load_best_model_at_end,
    #metric_for_best_model=set_metric_for_best_model,
    #greater_is_better=set_greater_is_better
)
'''


print('\n------------------ Setting TrainingArguments... ------------------\n')
training_args = TrainingArguments(
    output_dir=model_fp,
    group_by_length = set_group_by_length,
    per_device_train_batch_size = set_per_device_train_batch_size,
    evaluation_strategy = set_evaluation_strategy,
    max_steps = set_max_steps,
    fp16 = set_fp16,
    gradient_checkpointing = set_gradient_checkpointing,
    save_steps = set_save_steps,
    eval_steps = set_eval_steps,
    logging_steps = set_logging_steps,
    learning_rate = set_learning_rate,
    weight_decay = set_weight_decay,
    warmup_ratio = set_warmup_ratio,
    save_total_limit = set_save_total_limit,
)


print('\n------------------ Setting Trainer... ------------------\n')
'''
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    tokenizer=processor.feature_extractor,
)
'''
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=data_prepared["train"],
    eval_dataset=data_prepared["test"],
    tokenizer=processor.feature_extractor,
)


if training:
    print('\n------------------ Starting training... ------------------\n')
    torch.cuda.empty_cache()
    if use_checkpoint:
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        trainer.train()
    model.save_pretrained(model_fp)
    
       
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Training Finished:", dt_string)
    print('\n------------------ Training finished... ------------------\n')
    

print('\n------------------ Evaluation starts... ------------------\n')
processor = Wav2Vec2Processor.from_pretrained(model_fp)
model = Wav2Vec2ForCTC.from_pretrained(model_fp)


# Now, we will make use of the map(...) function to predict the transcription of every test sample and to save the prediction 
# in the dataset itself. We will call the resulting dictionary "results".

# Note: we evaluate the test data set with batch_size=1 on purpose due to this issue. Since padded inputs don't yield the 
# exact same output as non-padded inputs, a better WER can be achieved by not padding the input at all.

print('\n------------------ Generating fine-tuned results... ------------------\n')

def map_to_result(batch):
    model.to("cuda")
    input_values = processor(
      batch["speech"], 
      sampling_rate=batch["sampling_rate"], 
      return_tensors="pt"
    ).input_values.to("cuda")

    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
  
    return batch

'''
def map_to_result(batch):
    with torch.no_grad():
        #input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        input_values = torch.tensor(batch["input_values"], device="cuda")
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
  
    return batch
'''

results = dataset["test"].map(map_to_result)
print('results are:', results)

results_df = results.to_pandas()
# results_df = results_df.drop(columns=["speech", "sampling_rate", "input_values", "input_length", "labels"])
results_df = results_df.drop(columns=["speech", "sampling_rate"])
results_df.to_csv(finetuned_result_fp)
print('Fine-tuned results are saved to:', finetuned_result_fp)


print('\n------------------ Generating WER result on test dataset... ------------------\n')
print("Fine-tuned Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["target_text"])))


print('\n------------------ Showing some random prediction errors... ------------------\n')
# show_random_elements(results.remove_columns([["speech", "sampling_rate", "input_values", "input_length", "labels"]), num_examples=10)
show_random_elements(results.remove_columns(["speech", "sampling_rate"]), num_examples=10)


print('\n------------------ Generating the exact output of the model... ------------------\n')


model.to("cuda")
input_values = processor(
      dataset["test"][0]["speech"], 
      sampling_rate=dataset["test"][0]["sampling_rate"], 
      return_tensors="pt"
  ).input_values.to("cuda")

with torch.no_grad():
    logits = model(input_values).logits

pred_ids = torch.argmax(logits, dim=-1)

# convert ids to tokens
print(" ".join(processor.tokenizer.convert_ids_to_tokens(pred_ids[0].tolist())))


'''
model.to("cuda")
with torch.no_grad():
    logits = model(torch.tensor(dataset["test"][:1]["input_values"], device="cuda")).logits

pred_ids = torch.argmax(logits, dim=-1)

# convert ids to tokens
print(" ".join(processor.tokenizer.convert_ids_to_tokens(pred_ids[0].tolist())))
'''

now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("Evaluation Finished:", dt_string)
print('\n------------------ Evaluation finished... ------------------\n')
