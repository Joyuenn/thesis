print("\n------------ Importing libraries... ------------\n")

print('Importing partial')
from functools import partial


print('Importing string')
import string
print('Importing sys')
import sys
print(sys.version)
print('\n')
print('Importing datetime')
from datetime import datetime
print('Importing math')
import math
print('Importing platfom')
import platform




print('Importing numpy')
import numpy as np
print('Importing panda')
import pandas as pd
print('Importing random')
import random
print('Importing json')
import json
print('Importing librosa')
import librosa
print('Importing torch')
import torch
print('Importing praatio')
from praatio import tgio

 
print('Importing transformers')
import transformers
print('Importing Wav2Vec2CTCTokenizer')
from transformers import Wav2Vec2CTCTokenizer
print('Importing Wav2Vec2FeatureExtractor')
from transformers import Wav2Vec2FeatureExtractor
print('Importing Wav2Vec2Processor')
from transformers import Wav2Vec2Processor
print('Importing Wav2Vec2ForCTC')
from transformers import Wav2Vec2ForCTC
print('Importing TrainingArguments')
from transformers import TrainingArguments
print('Importing Trainer')
from transformers import Trainer
print('Importing transformers')

print('Importing datasets')
from datasets import load_dataset, load_metric, ClassLabel, Audio
print('Importing dataclasses')
from dataclasses import dataclass, field
print('Importing typing')
from typing import Any, Dict, List, Optional, Union

print("\n------------ Successfully imported libraries... ------------\n")

print ("\n")

print ("Test script executed on", datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))
print ("\n")

print ("The square root of 42 is", math.sqrt(42))
print ("\n")

print (datetime.now())
print ("\n")

print ("Python 2 is deprecated and you should avoid if possible. This is python version", platform.python_version())
print ("\n")
print("Transformers Version:", transformers.__version__)
print("Numpy Version:", np.__version__)
print("Test cuda_device_count", torch.cuda.device_count())
print("Test cuda_is_available", torch.cuda.is_available())

try:
    torch.cuda.get_device_name(0)
    print("Get device successful", torch.cuda.get_device_name(0))
except RuntimeError:
    print("Runtime error: Found no NVIDIA driver on your system")
