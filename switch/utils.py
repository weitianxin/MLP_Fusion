# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import sys
import wandb
sys.path.append("./")

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, SwitchTransformersForConditionalGeneration

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda:3' if cuda.is_available() else 'cpu'

import logging
# warm-up
from transformers import get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s: %(levelname)s| %(message)s',
                    filename='training.log',  # Log to this file
                    filemode='a')  # Append to the file, 'w' to overwrite
  
def train(epoch, model, device, loader, optimizer, scheduler, wandb):
    model.train()
    for _,data in enumerate(loader, 0):
        # data = {k: v.cuda() for k, v in data.items()}        
        # with accelerator.accumulate(model):
        
          # labels = data['target_ids']        
          # labels = labels.masked_fill_(labels == 0, -100)
          # ids = data['source_ids']
          # mask = data['source_mask']
          
          # print(labels.device,ids.device,mask.device)          
          
        labels = data['target_ids'].to(device, dtype = torch.long)
        labels = model._shift_right(labels)
        
        labels = labels.masked_fill_(labels == 0, -100)
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)
        
        # print(labels.shape,ids.shape,mask.shape)
        # decoder_input_ids = torch.zeros_like(labels).long()
        
        # with torch.autocast("cuda"):
            # outputs = model(**eval_batch, output_router_logits=True, return_dict=True)    
        # outputs = model(input_ids = ids, attention_mask = mask, labels=labels)    
        outputs = model(input_ids = ids, attention_mask = mask, labels=labels, output_router_logits=True, return_dict=True)
        train_loss = outputs[0]      
        wandb.log({"train_loss": train_loss.item()})    
        
          
        train_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
      
    # logging.info(f"Epoch: {epoch}, Train loss: {train_loss}")
  
  
def validate_seq2seq(tokenizer, model, device, loader, OUT_MAX_LEN):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids_y'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                early_stopping = True,
                max_length=OUT_MAX_LEN,
                num_beams = 4
                # "early_stopping": true,
                # "max_length": 300,
                # "num_beams": 4,
                # "prefix": "translate English to French: "

                )
                      
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [t.item() for t in y]
            
            
            # print(preds,target)
            if _!=0 and _%100==0:
                print(f'Completed {_}')
                # break

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals
  
  
  
def validate(tokenizer, model, device, loader, resultClass):
    model.eval()
    predictions = []
    actuals = []
    resultClassEncoded = [tokenizer.encode(x) for x in resultClass]
    # print(resultClass)
    print(resultClassEncoded)

    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids_y'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            
            
            decoder_input_ids = torch.full((ids.shape[0], 1), model.config.decoder_start_token_id, dtype=torch.long).to(device)
            # dec_input_ids = tokenizer("<extra_id_0>", return_tensors="pt").input_ids.cuda()[:, :1].to('cuda:1')
            
            # Forward pass
            outputs = model(input_ids=ids,decoder_input_ids=decoder_input_ids, attention_mask=mask)
            
            # print(outputs.logits.shape)

            # Convert logits to probabilities
            probabilities = F.softmax(outputs.logits, dim=-1)
            
            
            # print(probabilities.shape)
            
            for g in probabilities:
              # print(g[0].shape)
              # print([i[0] for i in resultClassEncoded])
              pred_logits = [g[0][i[0]] for i in resultClassEncoded]
              
              prediction = resultClass[pred_logits.index(max(pred_logits))]
              predictions.append(prediction)
              # print(g[0][resultClassEncoded[0]],g[0][resultClassEncoded[2]])
              
              # predictions.append(resultClass[0] if g[0][resultClassEncoded[0]]>g[0][resultClassEncoded[2]] else resultClass[1])
              
            # preds = torch.argmax(probabilities, dim=-1)
            # print(preds)
            target = [t.item() for t in y]
            
            
            # print(preds,target)
            if _!=0 and _%100==0:
                print(f'Completed {_}')
                
            actuals.extend(target)
    return predictions, actuals
  
def validate_head(model, device, loader):
  model.eval()
  predictions = []
  actuals = []

  with torch.no_grad():
      for _, data in enumerate(loader, 0):
          y = data['target_ids_y'].to(device, dtype = torch.long)
          ids = data['source_ids'].to(device, dtype = torch.long)
          mask = data['source_mask'].to(device, dtype = torch.long)
          
          
          # decoder_input_ids = torch.full((ids.shape[0], 1), model.config.decoder_start_token_id, dtype=torch.long).to(device)
          # decoder_input_ids = model._shift_right(ids).to(device)
          
          # Forward pass
          outputs = model(input_ids=ids,attention_mask=mask)#,decoder_input_ids=decoder_input_ids, )
          
                      
          probabilities = F.softmax(outputs.logits, dim=-1)
          
          # print()
          
          predictions+=(list(probabilities.argmax(1)))
          
          # for g in probabilities:
          #   predictions.append(g.argmax(0).item())
            
          target = [t.item() for t in y]
          
          
          # print(preds,target)
          # if _!=0 and _%100==0:
          #     print(f'Completed {_}')
              
          actuals.extend(target)
  return predictions, actuals

  
def train_val(epoch, model, device, train_loader, optimizer,val_loader,scheduler):
     
  
    model.train()
    for _,data in enumerate(train_loader, 0):
        labels = data['target_ids'].to(device, dtype = torch.long)
        
        labels = labels.masked_fill_(labels == 0, -100)
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)
        # decoder_input_ids = torch.zeros_like(labels).long()
        outputs = model(input_ids = ids, attention_mask = mask, labels=labels, output_router_logits=True, return_dict=True)
        train_loss = outputs[0]      
        wandb.log({"train_loss": train_loss.item()})    
        
          
        train_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        
        
    model.eval()
    val_loss = 0
    with torch.no_grad():  # No need to track the gradients
        for _, data in enumerate(val_loader, 0):
            labels = data['target_ids'].to(device, dtype=torch.long)
            labels = labels.masked_fill_(labels == 0, -100)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)
            outputs = model(input_ids=ids, attention_mask=mask, labels=labels, return_dict=True)
            loss = outputs.loss
            wandb.log({"val_loss": loss.item()})    
            # val_loss += loss.item()
    
    # Calculate the average losses
    # train_loss /= len(train_loader)
    # val_loss /= len(val_loader)

    logging.info(f"Epoch: {epoch}, Train loss: {train_loss}, Val loss: {loss}")
    return train_loss, loss



def freeze_switch_routers_for_finetuning(
        model: SwitchTransformersForConditionalGeneration
) -> SwitchTransformersForConditionalGeneration:
    model.router_z_loss_coef = 0
    model.router_aux_loss_coef = 0
    for name, param in model.named_parameters():
        if "router" in name:
            param.requires_grad = False
        # else:
        #     param.requires_grad = True
    return model

def freeze_switch_routers_and_experts_for_finetuning(
        model: SwitchTransformersForConditionalGeneration
) -> SwitchTransformersForConditionalGeneration:
    # model.router_z_loss_coef = 0
    # model.router_aux_loss_coef = 0
    for name, param in model.named_parameters():
        if "router" in name or "expert" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    return model
  
def freeze_switch_routers_and_center_experts_for_finetuning(
        model: SwitchTransformersForConditionalGeneration
) -> SwitchTransformersForConditionalGeneration:
    model.router_z_loss_coef = 0
    model.router_aux_loss_coef = 0
    for name, param in model.named_parameters():
        if "router" in name or "center_expert" in name or 'C' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    return model


def get_nb_trainable_parameters(model):
        r"""
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_bytes = param.quant_storage.itemsize if hasattr(param, "quant_storage") else 1
                num_params = num_params * 2 * num_bytes

            all_param += num_params
            if param.requires_grad:
                # print(_)
                trainable_params += num_params

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )