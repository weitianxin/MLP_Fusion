# NTK-approximating MLP Fusion for Efficient Language Model Fine-tuning

  

This is the code repository for the paper "NTK-approximating MLP Fusion for Efficient Language Model Fine-tuning"(ICML 2023). 




## Main Idea


We proposed to create a light-weighted pretrained language model by clustering sub-MLPs into centroids that could be restored as a compressed MLP, which well approximates the NTK(neural tangent kernel) of the original MLP. We validated our method on both natural language understanding (NLU) and generation (NLG) tasks.

  

## Run Code

  

Navigate to the "scripts" folder, and type into command prompt:

```

sh run_glue.sh

```

  

It will run the "run_glue.py" with our preset configurations(task, model settings, etc.).

  

Execute the following command before applying the "prunning" method(ffn_mode='prune'):

```

python oneshot.py --weight pre --model glue --rate 0.75

```

  

## Notice

  

Please cite the relevant paper if you use the code for scholarly or commercial purposes.

  

```

@InProceedings{wei-etal-2023-ntk,

title = {NTK-approximating MLP Fusion for Efficient Language Model Fine-tuning},

author = {Wei, Tianxin and Guo, Zeming and Chen, Yifan and He, Jingrui},

booktitle = {Proceedings of the 40th International Conference on Machine Learning},

year = {2023},

publisher = {PMLR},

}