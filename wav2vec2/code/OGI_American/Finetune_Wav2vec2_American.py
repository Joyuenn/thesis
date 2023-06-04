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
set_mask_time_prob = 0.065                  # Default = 0.05
print("mask_time_prob:", set_mask_time_prob)
set_mask_time_length = 10                   # Default = 10
print("mask_time_length:", set_mask_time_length)
set_ctc_loss_reduction = "mean"             # Default = "sum"
print("ctc_loss_reduction:", set_ctc_loss_reduction)
set_ctc_zero_infinity = True               # Default = False
print("ctc_zero_infinity:", set_ctc_zero_infinity)


print("\n------------------ Training arguments... ------------------\n")
# For setting training_args = TrainingArguments()

set_per_device_train_batch_size = 8         # Default = 8
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

set_learning_rate = 0.00004                 # Default = 0.00005
print("learning_rate:", set_learning_rate)
set_lr_scheduler_type = "linear"            # Default = "linear"
print("lr_scheduler_type:", set_lr_scheduler_type )
set_adam_beta1 = 0.9                        # Default = 0.9
print("adam_beta1:", set_adam_beta1)
set_adam_beta2 = 0.98                       # Default = 0.999
print("adam_beta2:", set_adam_beta2)
set_adam_epsilon = 0.00000001               # Default = 0.00000001
print("adam_epsilon:", set_adam_epsilon)
set_warmup_ratio = 0.1                      # Default = 0.0
print("warmup_ratio:", set_warmup_ratio)

set_num_train_epochs = 22                 # Default = 3.0
print("num_train_epochs:", set_num_train_epochs)
set_max_steps = 35000                       # Default = -1, overrides epochs
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
set_save_total_limit = 40                   # Optional                 
print("save_total_limit:", set_save_total_limit)

set_load_best_model_at_end = True           # Default = False
print("load_best_model_at_end:", set_load_best_model_at_end)
set_metric_for_best_model = "wer"           # Optional
print("metric_for_best_model:", set_metric_for_best_model)
set_greater_is_better = False               # Optional
print("greater_is_better:", set_greater_is_better)


print("\n------------------ Loading files... ------------------\n")

train_df_fp = '/srv/scratch/z5313567/thesis/OGI_local/OGI_scripted_train_dataframe.csv'
dev_df_fp = '/srv/scratch/z5313567/thesis/OGI_local/OGI_scripted_dev_dataframe.csv'
test_df_fp = '/srv/scratch/z5313567/thesis/OGI_local/OGI_scripted_test_dataframe.csv'
cache_fp = '/srv/scratch/chacmod/.cache/huggingfacse/datasets/Jordan-OGI-finetune'
model_fp = '/srv/scratch/z5313567/thesis/wav2vec2/model/OGI_American/model_OGI_American_20230528'
vocab_fp = '/srv/scratch/z5313567/thesis/wav2vec2/vocab/OGI_American/vocab_OGI_American_20230528.json'

print(f'Training dataset is stored at {train_df_fp}\n')
print(f'Development dataset is stored at {dev_df_fp}\n')
print(f'Testing dataset is stored at {test_df_fp}\n')
print(f'Cache filepath is {cache_fp}\n')
print(f'Model filepath is {model_fp}\n')
print(f'Vocab filepath is {vocab_fp}\n')


print("\n------------------ Loading datasets... ------------------\n")

dataset = load_dataset('csv', 
                    data_files={'train': train_df_fp,
                                'dev': dev_df_fp,
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
check_sampling_rate(dataset['dev'])

print('\n------------------Ckech sampling rates of testing datasets... ------------------\n')
check_sampling_rate(dataset['test'])


print("\n------------------ Verifying some ramdom samples... ------------------\n")
# verify if the data is a 1-dimensional array, the sampling rate corresponds to 16kHz, and the target text is clean.
for i in range(6):
    rand_int = random.randint(0, len(dataset["train"]))
    #rand_int = i
    print('Target text:', dataset['train'][rand_int]['target_text'])
    print('Input array shape:', np.asarray(dataset['train'][rand_int]['speech']).shape)
    print('Sampling rate:', dataset['train'][rand_int]["sampling_rate"])
    print('\n')
    
    
def prepare_dataset(batch):
    batch["input_values"] = processor(batch['speech'], sampling_rate=batch["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch
print("\n------------------ Obtaining input values, input length, and labels... ------------------\n")
#prep_dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=4)
dataset = dataset.map(prepare_dataset, num_proc=4)
print('dataset is:', dataset)


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

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

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


print('\n------------------ Loading a pretrained checkpount... ------------------\n')
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base", 
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


# CNN layers of Wav2vec2.0 model is sufficiently trained, hence they do not need to be finetuned anymore
model.freeze_feature_encoder()


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
    load_best_model_at_end=set_load_best_model_at_end,
    metric_for_best_model=set_metric_for_best_model,
    greater_is_better=set_greater_is_better
)


print('\n------------------ Setting Trainer... ------------------\n')
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    tokenizer=processor.feature_extractor,
)

print('\n------------------ Starting training... ------------------\n')
torch.cuda.empty_cache()
trainer.train()
model.save_pretrained(model_fp)
print('\n------------------ Training finished... ------------------\n')

now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("Finished:", dt_string)