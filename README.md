# NTK-approximating MLP Fusion for Efficient Language Model Fine-tuning

  

  

This is the code repository for the paper "NTK-approximating MLP Fusion for Efficient Language Model Fine-tuning"(ICML 2023).

  
  
  
  

## Main Idea

  
  

We proposed to create a light-weighted pretrained language model by clustering sub-MLPs into centroids that could be restored as a compressed MLP, which well approximates the NTK(neural tangent kernel) of the original MLP. We validated our method on both natural language understanding (NLU) and generation (NLG) tasks.

 
<img width="600" alt="image (2)" src="https://github.com/weitianxin/MLP_Fusion/assets/67295087/78d42331-30df-4e4e-9c3f-fa9e35e94500">

  

## Run Code

  
### NLU Tasks
  

Navigate to "scripts/run_glue.sh", and edit the parameters, including but limited to:

- "**model_name_or_path**": select a model

- "**task_name**": NLU benchmark's name (e.g. "sst2", "mnli").

- "**distill**": whether to enable distillation.

- "**max_seq_length**": maximum sequence length(to be padded/truncated to).

- "**ffn_mode**": chosen reduction/sketching method (eg. "sketch", "cluster", "mmd").

- "**sketch_layers**": MLP layers that the sketching/reduction methods applied to.

- "**ffn_bn**": feedforward network's bottleneck layer's dimension.

- "**mid_dim**": intermediate dimension.

- "**re_init_layers**": chosen re-initialize layers.

- "**seed**": chosen random seed.

- "**num_train_epochs**": number of training epochs.

- "**learning_rate**": chosen learning rate.

- "**metric_for_best_model**": metric for the chosen NLU benchmark.

Then, type into command prompt:

```
sh run_glue.sh
```
It will run the "run_glue.py" with the set of modified configurations, which validates the chosen reduction/sketching method on the selected NLU benchmark.

  

  ### NLG Tasks
  
To choose a set of configurations for the task, navigate to the file "nlg/scripts/run_nlg.sh". Within this file, you can choose to use any of configurations available in "nlg/configs" by modifying the parameters. In case you want to create your own set of configurations, you can do so by creating a new .json file within the "nlg/configs" directory. This way, you can customize the configurations according to your specific requirements.

Some important parameters in configurations:

 - "**seed**":  random seed.
 
 - "**task**": task to be executed.
 
 - "**model_name**": model's name.
 
 - "**n_epochs**": number of training epochs.
 
 - "**train_batch_size/valid_batch_size**": batch size for training/validation.
 
 - "**lr**": learning rate.
 
 - "**ffn_mode**": chosen reduction/sketching method (e.g. "sketch", "cluster", "mmd").
 
 - "**sketch_layers**": MLP layers that the sketching/reduction methods applied to.
 
 - "**ffn_bn**": feedforward network's bottleneck layer's dimension.
 
 - "**mid_dim**": intermediate dimension.

After adding configurations, simply execute the task by typing into command prompt:
```
sh run_nlg.sh
```
It will run "train.py" and "evaluate.py" with whatever configurations assigned to trainings and validations.
  

  

## Notice

  

  

Please cite the relevant paper if you use the code for scholarly or commercial purposes.

  

  

```
@InProceedings{pmlr-v202-wei23b,
  title = 	 {NTK-approximating MLP Fusion for Efficient Language Model Fine-tuning},
  author =       {Wei, Tianxin and Guo, Zeming and Chen, Yifan and He, Jingrui},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {36821--36838},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR}
}
```
