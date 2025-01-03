# Importing stock libraries
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, get_scheduler
from switch_transformers import SwitchForSequenceClassification, SwitchTransformersConfig


from data.utils import (
  my_dataset_map,
  mySet,
  Config,
)


from utils import(
  train_val,
  freeze_switch_routers_and_experts_for_finetuning,
  freeze_switch_routers_for_finetuning,
  validate_head,
  get_nb_trainable_parameters
)


from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--dataset", default="mrpc", type=str, help="dataset_name")
parser.add_argument("--device", default="3", type=str, help="device")
parser.add_argument("--mode", default="", type=str, help="baseline method")
parser.add_argument("--batch", default="64", type=int, help="batch size")
parser.add_argument("--lr", default="1e-4", type=float, help="learning rate")
parser.add_argument("--epoch", default="10", type=int, help="traning epoch")
parser.add_argument("--seed", default="42", type=int, help="seed")
parser.add_argument("--distill", default="0", type=int, help="distillation")


args = parser.parse_args()
dataset_name = args.dataset
device = 'cuda:' + args.device
distill = args.distill



def train_val(model, device, train_loader, optimizer,val_loader,scheduler):
     
  
    model.train()
    for _,data in enumerate(train_loader, 0):
        
        labels = data['target_ids_y'].to(device,dtype = torch.long)
        ids = data['source_ids'].to(device,dtype = torch.long)
        mask = data['source_mask'].to(device,dtype = torch.long)
        
        
        outputs = model(input_ids = ids, attention_mask = mask, labels=labels,return_dict=True)
        train_loss = outputs[0]      
        wandb.log({"train_loss": train_loss.item()})    
        
        train_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        
        
    model.eval()
    with torch.no_grad():  # No need to track the gradients
        for _, data in enumerate(val_loader, 0):
            
            labels = data['target_ids_y'].to(device,dtype=torch.long)
            ids = data['source_ids'].to(device,dtype=torch.long)
            mask = data['source_mask'].to(device,dtype=torch.long)
            
            
            outputs = model(input_ids=ids, attention_mask=mask, labels=labels, return_dict=True)
            loss = outputs.loss
            wandb.log({"val_loss": loss.item()})    

    return train_loss, loss
  
  
config = Config()       # Initialize config
config.TRAIN_BATCH_SIZE = args.batch   # input batch size for training (default: 64)
config.TRAIN_EPOCHS = args.epoch  # number of epochs to train (default: 10)
config.LEARNING_RATE = args.lr    # learning rate (default: 0.01)
config.SEED = args.seed               # random seed (default: 42)
config.INPUT_MAX_LEN = 100
config.OUT_MAX_LEN = 10
config.VALID_BATCH_SIZE = 10   # input batch size for testing (default: 1000)
config.VAL_EPOCHS = 1 
    

# Set random seeds and deterministic pytorch for reproducibility
torch.manual_seed(config.SEED) # pytorch random seed
np.random.seed(config.SEED) # numpy random seed
torch.backends.cudnn.deterministic = True

# tokenzier for encoding the text
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-16")       
encoded_dataset = my_dataset_map(dataset_name)
    
train_dataset=encoded_dataset["train"]
val_dataset=encoded_dataset["validation_matched"] if dataset_name == "mnli" else encoded_dataset["validation"]
test_dataset=encoded_dataset["test_matched"] if dataset_name == "mnli" else encoded_dataset["test"]
  
  
# Creating the Training and Validation dataset for further creation of Dataloader
train_set = mySet(train_dataset, tokenizer, config.INPUT_MAX_LEN, config.OUT_MAX_LEN,dataset_name)
val_set = mySet(val_dataset, tokenizer, config.INPUT_MAX_LEN, config.OUT_MAX_LEN,dataset_name)
test_set = mySet(test_dataset, tokenizer, config.INPUT_MAX_LEN, config.OUT_MAX_LEN,dataset_name)

# Defining the parameters for creation of dataloaders
train_params = {
    'batch_size': config.TRAIN_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 2
    }

val_params = {
    'batch_size': 10,
    'shuffle': False,
    'num_workers': 2
}

# Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
train_loader = DataLoader(train_set, **train_params)
val_loader = DataLoader(val_set, **val_params)
test_loader = DataLoader(test_set, **val_params)


switch_config = SwitchTransformersConfig.from_pretrained(
        "google/switch-base-16",
        num_labels=3 if dataset_name == "mnli" else 2, 
        finetuning_task=dataset_name,
        ffn_mode=args.mode
)


model = SwitchForSequenceClassification.from_pretrained(
    "google/switch-base-16",
    config=switch_config
).to(device)


if switch_config.ffn_mode == 'sketch':
    with torch.no_grad():
        model.update_sketch([5,7,9,11])
if switch_config.ffn_mode == 'mmd':
    with torch.no_grad():
        model.update_mmd([5,7,9,11])
if switch_config.ffn_mode == 'cluster':
    with torch.no_grad():
        model.update_clusters([5,7,9,11])
if switch_config.ffn_mode == 'ntk_cluster':
    with torch.no_grad():        
        model.update_ntk_clusters([5,7,9,11])
if switch_config.ffn_mode == 'svd':
    with torch.no_grad():    
        model.update_svd([5,7,9,11])
if switch_config.ffn_mode == 'structure_prune':
    with torch.no_grad():    
        model.update_structure_pruning([5,7,9,11]) 
        
        
if distill:
    ori_model = copy.deepcopy(model).to(device)
    params_to_update = []
    encoder_layers = model.switch_transformers.encoder.block
    decoder_layers = model.switch_transformers.decoder.block
    for ii in [5,7,9,11]:
        encoder_experts = encoder_layers[ii].layer[1].mlp.experts
        decoder_experts = decoder_layers[ii].layer[2].mlp.experts
        
        expert_indices = encoder_experts.keys()
        for idx in expert_indices:
          # print(encoder_experts[idx].wi.parameters())
          params_to_update.append({"params": encoder_experts[idx].wi.parameters()})
          params_to_update.append({"params": encoder_experts[idx].wo.parameters()})
          
        expert_indices = decoder_experts.keys()
        for idx in expert_indices:
          # print(encoder_experts[idx].wi.parameters())
          params_to_update.append({"params": decoder_experts[idx].wi.parameters()})
          params_to_update.append({"params": decoder_experts[idx].wo.parameters()})
          
        
    opt_distill = torch.optim.Adam(params_to_update, lr=Config.LEARNING_RATE)
    
    # NOTE: number of epoch for distilation
    for epoch in range(1):
      for data in tqdm(train_loader):
        
        ids = data['source_ids'].to(device,dtype = torch.long)
        mask = data['source_mask'].to(device,dtype = torch.long)
        
        with torch.no_grad():
            ori_hidden = ori_model(input_ids = ids, attention_mask = mask, return_dict=True)['decoder_hidden_states']
        # print(ori_hidden.keys())
        opt_distill.zero_grad()
        loss = model.forward_distill(input_ids = ids, attention_mask = mask, return_dict=True, output_hidden_states=True, ori_hidden=ori_hidden)
        loss.backward()
        opt_distill.step()
        
            
    del(ori_model)
    torch.cuda.empty_cache()
            
            
 
        
model = freeze_switch_routers_for_finetuning(model)



get_nb_trainable_parameters(model)



no_decay = ["bias", "layer_norm.weight", "LayerNorm", "layernorm", "layer_norm", "ln"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = torch.optim.AdamW(params = optimizer_grouped_parameters, lr=config.LEARNING_RATE, betas=(0.9, 0.98), eps=1e-08, weight_decay=0.01)



num_steps_per_epoch = len(train_loader)
num_training_steps = num_steps_per_epoch * config.TRAIN_EPOCHS


scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps= 8 * num_steps_per_epoch,
    num_training_steps=num_training_steps
)


wandb.init(
    # set the wandb project where this run will be logged
    project=f"switch-base-16-{dataset_name}-head",
        # track hyperparameters and run metadata
    config={
    "learning_rate": config.LEARNING_RATE,
    "batch_size": config.TRAIN_BATCH_SIZE,
    "epochs": config.TRAIN_EPOCHS,
    }
)
  

metric = evaluate.load("accuracy")
print('Initiating Fine-Tuning for the model','\n',f"BEGINING THE FINETUNING FOR switch-base-16-{dataset_name}-{switch_config.ffn_mode}-epoch{str(config.TRAIN_EPOCHS)}-batch{str(config.TRAIN_BATCH_SIZE)}-lr{str(config.LEARNING_RATE)}")


import time
t = time.time()


for epoch in range(config.TRAIN_EPOCHS):
    train_val(model, device, train_loader, optimizer,val_loader,scheduler)
    
    
    predictions, actuals = validate_head(model, device, val_loader)
    
    result = metric.compute(predictions=predictions, references=actuals)
    
    print(result)
  
print("Take ",time.time()-t," senconds.")
