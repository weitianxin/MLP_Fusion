# Importing stock libraries
import sys
sys.path.append("./")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.utils.prune as prune
import torch.nn as nn
import scipy.sparse as sp

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from transformers import AutoTokenizer, AutoConfig

from transformers import (
    SwitchTransformersConfig,
    SwitchTransformersForConditionalGeneration,
    SwitchTransformersSparseMLP,
    SwitchTransformersTop1Router,
    PretrainedConfig
)
from torch import cuda
device = 'cuda:3' if cuda.is_available() else 'cpu'
torch.cuda.set_device(device)

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch

from wb import (
  is_diagonal,
  get_approx_loss,
  permute_weights,
  get_optimal_permutation
)


import ot
from data.utils import (
  TASK_MAPPING_DATASET_ARGUMENTS,
  my_dataset_map
)

from utils import(
  validate,
)
from data.utils import(
  eval_map,
  mySet
)



model = SwitchTransformersForConditionalGeneration.from_pretrained("google/switch-base-16")#.to(device)
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-16")


# print(model.encoder,model.decoder)



for idx in [5,7,9,11]:
  # print(model.encoder.block[idx].layer[1].mlp.experts)
  
  MLP = model.encoder.block[idx].layer[1].mlp
  expert_indices = MLP.experts.keys()
  
  w1 = torch.stack([(MLP.experts[idx].wi.weight) for idx in expert_indices])
  w2 = torch.stack([(MLP.experts[idx].wo.weight) for idx in expert_indices])    
  wd = torch.cat((w1,w2.transpose(1, 2)),dim=2)
  
  print(wd.shape)
  
  torch.save(wd,f'./switch-base-16-wd/encoder-layer{idx}.pt')
  
  
  
  
# w1 = torch.stack([(MLP.experts[idx].wi.weight) for idx in expert_indices])
#     w2 = torch.stack([(MLP.experts[idx].wo.weight) for idx in expert_indices])    
#     wd = torch.cat((w1,w2.transpose(1, 2)),dim=2)


for idx in [5,7,9,11]:
  # print(model.decoder.block[idx].layer[2].mlp.experts)
  MLP = model.decoder.block[idx].layer[2].mlp
  expert_indices = MLP.experts.keys()
  
  w1 = torch.stack([(MLP.experts[idx].wi.weight) for idx in expert_indices])
  w2 = torch.stack([(MLP.experts[idx].wo.weight) for idx in expert_indices])    
  wd = torch.cat((w1,w2.transpose(1, 2)),dim=2)
  
  torch.save(wd,f'./switch-base-16-wd/decoder-layer{idx}.pt')