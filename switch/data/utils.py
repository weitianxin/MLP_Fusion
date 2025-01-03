from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch




TASK_MAPPING_DATASET_ARGUMENTS = {
    "cola": ["glue", "cola"],
    "stsb": ["glue", "stsb"],
    "rte": ["glue", "rte"],
    "mnli": ["glue", "mnli"],
    "sst2": ["glue", "sst2"],
    "qqp": ["glue", "qqp"],
    "qnli": ["glue", "qnli"],
    "mrpc": ["glue", "mrpc"],
    "winogrande": ["winogrande", "winogrande_xl"],
    "hellaswag": ["hellaswag"],
    "lambada": ["lambada"],
}

def my_dataset_map(dataset_name):
  dataset = load_dataset(*TASK_MAPPING_DATASET_ARGUMENTS[dataset_name]) 

  if dataset_name == "sst2":
    def preprend(example):
      prompts = [
        # f"Sentiment analysis (negative or positive): {x}" for x in example['sentence']
        f"sentence: {x}" for x in example['sentence']
      ]
      answer = [
        "positive" if x==1 else "negative" for x in example['label']
      ]
      return {"sentence":prompts,"answer":answer}
    
  elif dataset_name == "mnli":
    def preprend(example):
      prompts = [
        # f"Sentiment analysis (negative or positive): {x}" for x in example['sentence']
        f"premise: {x}\nhypothesis: {y}" for x,y in zip(example['premise'],example['hypothesis'])
      ]
      answer = [
        ["entailment","neutral","contradiction"][x] for x in example['label']
      ]
      return {"sentence":prompts,"answer":answer}
    
  elif dataset_name == "cola":
    def preprend(example):
      prompts = [
        # f"Sentiment analysis (negative or positive): {x}" for x in example['sentence']
        f"cola sentence: {x}" for x in example['sentence']
      ]
      answer = [
        "acceptable" if x==1 else "unacceptable" for x in example['label']
      ]
      return {"sentence":prompts,"answer":answer}
    
    
  elif dataset_name == "copa":
    def preprend(examples):
      prompts = [
        f"copa choice1: {choice1}\nchoice2: {choice2}\npremise: {premise}\nquestion: {question}"
      # f"What is the {'cause' if question == 'cause' else 'effect'} of the premise: {premise}\n\nChoose the best option:\n- {choice1}\n- {choice2}"
      for premise, question, choice1, choice2 in zip(examples['premise'], examples['question'], examples['choice1'], examples['choice2'])
      ]    
      answer = ["1" if label==0 else "2"
        for choice1, choice2, label in zip(examples['choice1'], examples['choice2'],examples['label'])
      ]
      return {"sentence":prompts,"answer":answer}
    
  elif dataset_name == "mrpc":
    def preprend(examples):
      prompts = [
      # f"Do the following two sentences mean the same thing?\n\n- {sentence1}\n- {sentence2}"
      f"mrpc sentence1: {sentence1}\nsentence2: {sentence2}"
      for sentence1, sentence2 in zip(examples['sentence1'], examples['sentence2'])
      ]    
      answer = ["equivalent" if label==1 else "inequivalent"
        for label in examples['label']
      ]
      return {"sentence":prompts,"answer":answer}
  
    
    
    
  encoded_dataset = dataset.map(preprend, batched=True)    
  return encoded_dataset


class mySet(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len, dataset):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        if dataset in ["XSum","opus"]:        
          self.context = self.data["document"]
          self.summaries = self.data["summary"]
          
        elif dataset in ["sst2","copa","mrpc","winogrande","cola","mnli"]:        
          self.context = self.data["sentence"]
          self.answers = self.data["answer"]
          self.labels = self.data["label"]
          
        # print(self.context)


    def __len__(self):
        return len(self.context)

    def __getitem__(self, index):
        data = self.context[index]
        answer = self.answers[index]
        label = self.labels[index]
        
        # print(label)

        source = self.tokenizer.batch_encode_plus([data], max_length= self.source_len, padding='max_length',return_tensors='pt',truncation = True)
        target = self.tokenizer.batch_encode_plus([answer], max_length= self.summ_len, padding='max_length',return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_ids_y = torch.tensor(label).reshape(-1,)
        # target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids_y.to(dtype=torch.long)
        }
        
 
     
def eval_map(val_dataset,dataset_name):
  if dataset_name == "copa":
    id2choices = {
        item['idx']: ["1","2"] for item in val_dataset
    }
    id2references = {
        item['idx']: item['label'] for item in val_dataset
    }
  elif dataset_name == "mrpc":
    id2choices = {
        idx: ["inequivalent","equivalent"] for idx,item in enumerate(val_dataset)
    }
    id2references = {
        idx: item['label'] for idx,item in enumerate(val_dataset)
    }
  elif dataset_name == "sst2":
    id2choices = {
        item['idx']: ["negative","positive"] for idx,item in enumerate(val_dataset)
    }
    id2references = {
        item['idx']: item['label'] for item in val_dataset
    }
    
  elif dataset_name == "cola":
    id2choices = {
        item['idx']: ["unacceptable","acceptable"] for idx,item in enumerate(val_dataset)
    }
    id2references = {
        item['idx']: item['label'] for item in val_dataset
    }  
    
  elif dataset_name == "mnli":
    id2choices = {
        item['idx']: ["entailment","neutral","contradiction"] for idx,item in enumerate(val_dataset)
    }
    id2references = {
        item['idx']: item['label'] for item in val_dataset
    }  
  elif dataset_name == "winogrande":
    id2choices = {
        idx: [item['option1'], item['option2']] for idx,item in enumerate(val_dataset)
    }
    id2references = {
        idx: item['label'] for idx,item in enumerate(val_dataset)
    }
    
  return id2choices,id2references
     
        
class Config():
  def __init__(self, TRAIN_BATCH_SIZE=16, VALID_BATCH_SIZE=16, TRAIN_EPOCHS=1, VAL_EPOCHS=1,LEARNING_RATE=1e-4,SEED=42,INPUT_MAX_LEN=256,OUT_MAX_LEN=80):
    Config.TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE    # input batch size for training (default: 64)
    Config.VALID_BATCH_SIZE = VALID_BATCH_SIZE    # input batch size for testing (default: 1000)
    Config.TRAIN_EPOCHS = TRAIN_EPOCHS        # number of epochs to train (default: 10)
    Config.VAL_EPOCHS = VAL_EPOCHS 
    Config.LEARNING_RATE = LEARNING_RATE    # learning rate (default: 0.01)
    Config.SEED = SEED               # random seed (default: 42)
    Config.INPUT_MAX_LEN = INPUT_MAX_LEN
    Config.OUT_MAX_LEN = OUT_MAX_LEN