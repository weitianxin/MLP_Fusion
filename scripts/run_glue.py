#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
import numpy as np
from datasets import load_dataset, load_metric
path = os.getcwd()
sys.path.append(path)
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import torch.nn.utils.prune as prune

sys.path.append("..")
from petl.options import (
    GenerationArguments,
    TuneArguments,
)
from petl.petl_enc_model import PETLEncModel
from models.roberta.modeling_roberta import RobertaForSequenceClassification
import torch
import copy
from petl.utils import convert_str_to_list, create_optimizer, create_scheduler


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    max_tokens_per_batch: Optional[int] = field(
        default=0,
        metadata={
            "help": "dynamic batching. Override batch size when larger than 0"
        },
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

def rewind(pre_weight):

    recover_dict = {}
    name_list = []
    for ii in range(12):
        name_list.append('roberta.encoder.layer.'+str(ii)+'.intermediate.dense.weight')
        name_list.append('roberta.encoder.layer.'+str(ii)+'.output.dense.weight')

    for key in pre_weight.keys():

        if 'roberta' in key:
            if key in name_list:
                new_key = key+'_orig'
            else:
                new_key = key

            recover_dict[new_key] = pre_weight[key]

    return recover_dict


def pruning_model(model,px):

    parameters_to_prune =[]
    for ii in range(12):
        parameters_to_prune.append((model.roberta.encoder.layer[ii].intermediate.dense, 'weight'))
        parameters_to_prune.append((model.roberta.encoder.layer[ii].output.dense, 'weight'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )


def pruning_model_custom(model, mask_dict, sketch_layers_list):

    parameters_to_prune =[]
    mask_list = []
    for ii in sketch_layers_list:
        parameters_to_prune.append(model.roberta.encoder.layer[ii].intermediate.dense)
        mask_list.append(mask_dict['roberta.encoder.layer.'+str(ii)+'.intermediate.dense.weight_mask'])
        parameters_to_prune.append(model.roberta.encoder.layer[ii].output.dense)
        mask_list.append(mask_dict['roberta.encoder.layer.'+str(ii)+'.output.dense.weight_mask'])


    for ii in range(len(parameters_to_prune)):
        prune.CustomFromMask.apply(parameters_to_prune[ii], 'weight', mask=mask_list[ii])


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments,
                               TuneArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, tune_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, tune_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # put useful args into config: these arguments will be used in models, thus adding them to config
    # interested_args = ['use_prefix', 'mid_dim', 'preseqlen', 'prefix_dropout', 'unfreeze_params']
    for k, v in vars(tune_args).items():
        if not hasattr(config, k):
            setattr(config, k, v)
    
    setattr(config, "pretrained_model_name_or_path", model_args.model_name_or_path)
    setattr(config, "seed", training_args.seed)

    setattr(training_args, 'max_tokens_per_batch', data_args.max_tokens_per_batch)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )


    if model_args.model_type == "roberta_custom": # load the customized model
        # config.num_hidden_layers = 8
        model = RobertaForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    
    trainable_params = sum(p.numel() for p in 
                    model.parameters() if p.requires_grad)
    print("***** Model Trainable Parameters {} *****"
            .format(trainable_params))
    
    # config.ffn_mode = ""
    # model.roberta.encoder.layer[0] = RobertaLayer(config)
    #########################################################
    ori_model = copy.deepcopy(model).to(training_args.device)
    # approx check before
    if tune_args.approx_check:
        if "validation" not in datasets and "validation_matched" not in datasets:
                raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=training_args.per_device_eval_batch_size)
        model = model.to(training_args.device)
        emb_before = []
        for batch in eval_dataloader:
            inputs = tokenizer(batch['sentence'], return_tensors='pt', padding=True, truncation=True, max_length=data_args.max_seq_length)
            inputs = {k: v.to(training_args.device) for k, v in inputs.items()}
            hidden = model(**inputs, output_hidden_states=True)["hidden_states"]
            before_sketch = hidden[1]
            emb_before.append(before_sketch)
        model = model.cpu()
    
    # approx check ntk before
    if tune_args.approx_check_ntk:
        if "validation" not in datasets and "validation_matched" not in datasets:
                raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1)
        # try:
        #     ntk_before = torch.load("ntk_roberta_ori.pt")
        # except:
        model = model.to(training_args.device)
        parameters = []
        parameters.extend(model.roberta.encoder.layer[0].intermediate.dense.parameters())
        parameters.extend(model.roberta.encoder.layer[0].output.dense.parameters())
        grad_list = []
        for i, batch in enumerate(eval_dataloader):
                # if i==10:
                #     break
                inputs = tokenizer(batch['sentence'], return_tensors='pt', padding=True, truncation=True, max_length=data_args.max_seq_length)
                inputs = {k: v.to(training_args.device) for k, v in inputs.items()}
                outputs = model(**inputs)
                loss = torch.nn.functional.softmax(outputs.logits, dim=-1)[:, 0]
                model.zero_grad()
                grad = torch.autograd.grad(loss, parameters)
                grad = [sth.detach().cpu() for sth in grad]
                grad_vector = torch.cat([torch.flatten(sth) for sth in grad])
                grad_list.append(grad_vector)
        grad = torch.stack(grad_list)
        sign_grad = torch.sign(grad)
        ntk_before = torch.matmul(grad, sign_grad.T)
        # torch.save(ntk_before, "ntk_roberta_ori.pt")
        model = model.cpu()
    # mmd
    if config.ffn_mode == 'mmd':
        # with torch.no_grad():
        model.update_mmd(convert_str_to_list(tune_args.sketch_layers))
    # cluster
    if config.ffn_mode == 'cluster':
        with torch.no_grad():
            print("input cluster number: ", tune_args.ffn_bn)
            model.update_clusters(convert_str_to_list(tune_args.sketch_layers))
    # mlp fusion
    if config.ffn_mode == 'ntk_cluster':
        with torch.no_grad():
            print("input cluster number: ", tune_args.ffn_bn)
            model.update_ntk_clusters(convert_str_to_list(tune_args.sketch_layers))
            
    # Apply sketchining to change the weight matrix and bias of the FFNs
    if config.ffn_mode == 'sketch' and tune_args.sketch_layers:
        with torch.no_grad():
            model.update_sketch(convert_str_to_list(tune_args.sketch_layers))
    
    # Singular value decomposition
    if config.ffn_mode == 'svd' and tune_args.sketch_layers:
        model.update_svd(convert_str_to_list(tune_args.sketch_layers))
    
    # prunning
    if config.ffn_mode == 'prune':
        mask = torch.load("pretrain_prun/mask.pt")
        pruning_model_custom(model, mask, convert_str_to_list(tune_args.sketch_layers))

    # lottery ticket hypothesis
    if config.ffn_mode == 'lth':
        model = model.to(training_args.device)
        mask = torch.load("pretrain_prun/mask_lth_{}.pt".format(data_args.task_name))
        pruning_model_custom(model, mask, convert_str_to_list(tune_args.sketch_layers))

    if config.ffn_mode == 'lth_pre':
        origin_model_dict = rewind(model.state_dict())

    if config.ffn_mode in ["sketch", "cluster", "mmd"] and tune_args.re_init_layers:
        print("reinit layers: {}".format(tune_args.re_init_layers))
        with torch.no_grad():
            model.reinit(convert_str_to_list(tune_args.re_init_layers))
    
    # distillation
    if tune_args.distill:
        model = model.to(training_args.device)
        train_dataset = datasets["train"]
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size)
        params_to_update = []
        for ii in convert_str_to_list(tune_args.sketch_layers):
            params_to_update.append({"params": model.roberta.encoder.layer[ii].intermediate.dense.parameters()})
            params_to_update.append({"params": model.roberta.encoder.layer[ii].output.dense.parameters()})
        opt_distill = torch.optim.Adam(params_to_update, lr=training_args.learning_rate)
        
        # NOTE: number of epoch for distilation
        for epoch in range(1):
            for batch in tqdm(train_dataloader):
                inputs = tokenizer(batch['sentence'], return_tensors='pt', padding=True, truncation=True, max_length=data_args.max_seq_length)
                inputs = {k: v.to(training_args.device) for k, v in inputs.items()}
                with torch.no_grad():
                    ori_hidden = ori_model(**inputs, output_hidden_states=True)["hidden_states"]
                opt_distill.zero_grad()
                loss = model.forward_distill(**inputs, output_hidden_states=True, ori_hidden=ori_hidden)
                loss.backward()
                opt_distill.step()
    trainable_params = sum(p.numel() for p in 
                    model.parameters() if p.requires_grad)
    print("***** Model Trainable Parameters {} *****"
            .format(trainable_params))
    # approx check
    if tune_args.approx_check:
        emb_after = []
        model = model.to(training_args.device)
        for batch in eval_dataloader:
            inputs = tokenizer(batch['sentence'], return_tensors='pt', padding=True, truncation=True, max_length=data_args.max_seq_length)
            inputs = {k: v.to(training_args.device) for k, v in inputs.items()}
            
            hidden = model(**inputs, output_hidden_states=True)["hidden_states"]
            after_sketch = hidden[1]
            emb_after.append(after_sketch)
        norm_list = []
        for i in range(len(emb_after)):
            matrix = emb_after[i]-emb_before[i]
            norm = torch.mean(torch.sqrt(torch.sum(torch.square(matrix),dim=-1)),dim=1).detach().cpu().numpy()
            norm_list.extend(norm)
        print("{}: {}".format(config.ffn_mode, np.mean(norm)))
        exit()
    # approx check ntk
    if tune_args.approx_check_ntk:
        model = model.to(training_args.device)
        parameters = []
        parameters.extend(model.roberta.encoder.layer[0].intermediate.dense.parameters())
        parameters.extend(model.roberta.encoder.layer[0].output.dense.parameters())
        grad_list = []
        for i, batch in enumerate(eval_dataloader):
            # if i==10:
            #     break
            inputs = tokenizer(batch['sentence'], return_tensors='pt', padding=True, truncation=True, max_length=data_args.max_seq_length)
            inputs = {k: v.to(training_args.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            loss = torch.nn.functional.softmax(outputs.logits, dim=-1)[:, 0]
            model.zero_grad()
            grad = torch.autograd.grad(loss, parameters)
            grad = [sth.detach().cpu() for sth in grad]
            grad_vector = torch.cat([torch.flatten(sth) for sth in grad])
            grad_list.append(grad_vector)
        grad = torch.stack(grad_list)
        sign_grad = torch.sign(grad)
        ntk_after = torch.matmul(grad, sign_grad.T)
        norm = torch.sqrt(torch.sum(torch.square(ntk_after-ntk_before)))
        print("{} NTK: {}".format(config.ffn_mode, norm.item()))
        exit()
   

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # Dynamically padding at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if tune_args.attn_mode != "none" or tune_args.ffn_mode != "none":
        if tune_args.load_path == "":
            model = PETLEncModel(config, tune_args, model)
        else:
            model = PETLEncModel.from_pretrained(
                tune_args.load_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                args=tune_args,
                pretrained_model=model,
            )


    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datasets = datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        metric = load_metric("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    if config.ffn_mode == "none" or tune_args.lr_sketch_factor < 0:
        optimizers=(None, None)
    else:
        optimizer = create_optimizer(training_args, 
            model, tune_args.lr_sketch_factor, tune_args.sketch_layers)
        scheduler = create_scheduler(training_args, 
            len(train_dataset), optimizer)
        optimizers = (optimizer, scheduler)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=optimizers,
    )
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        
        if tune_args.ffn_mode=="lth_pre":
            print('starting pruning')
            pruning_model(model.pretrained_model, 0.75)
            print('rewinding')
            model_dict = model.pretrained_model.state_dict()
            model_dict.update(origin_model_dict)
            mask_dict = {}
            for key in model_dict.keys():
                if 'mask' in key:
                    mask_dict[key] = model_dict[key]   
            torch.save(mask_dict, os.path.join("pretrain_prun", "mask_lth_{}.pt".format(data_args.task_name))) 
            print("save finished")
            exit()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    #########################################################    
    # print(list(trainer.model.parameters())[0].grad)

    # for n, p in trainer.model.named_parameters():
    #     if "dense.weight" in n and "0" in n: 
    #         print(n, p.shape)
    #         print(p)
    #########################################################    


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()