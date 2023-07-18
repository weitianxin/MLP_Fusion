#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=glue
#SBATCH --nodes=1
#SBATCH --gres=gpu:3090:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=4
#SBATCH --time=0
#SBATCH --array=0-4%5

# two re_init + 1 distill
export TRANSFORMERS_CACHE=checkpoints/hf_model
export HF_DATASETS_CACHE=checkpoints/hf_model
export HF_METRICS_CACHE=checkpoints/hf_model
 
cache_dir=${TRANSFORMERS_CACHE}
 

# Natural language understanding benchmarks
TASK_NAME=mnli
# TASK_NAME=sst2
metric="accuracy"


# wandb env variables
export WANDB_PROJECT=glue.${TASK_NAME}
export WANDB_WATCH="false"
 
DATE=`date +%Y%m%d`
 
# declare -a root_seed_list=(42 2 4 6 8)
# seed=${root_seed_list[$SLURM_ARRAY_TASK_ID]}
seed=42
 
 
# ----- TX Test -----
 
attn_mode="none"
attn_option="concat"
attn_composition="add"
attn_bn=16  # attn bottleneck dim
 

#Reduction/Sketching Methods:
# ffn_mode="none"
# Our method:
ffn_mode='ntk_cluster'
# Other methods:
# ffn_mode="sketch"
# ffn_mode='cluster'
# ffn_mode='mmd'
# ffn_mode='svd'

n_steps=100

# Feedforward Network's bottleneck layer's dimension < 3076
ffn_bn=768 
approx_check=False
# Apply distillation
distill=False
load_clustering=True

# MLP layers that the sketching/reduction methods applied to:
sketch_layers='4,5,6,7,8,9,10,11'

ffn_option="none"
ffn_adapter_layernorm_option="none"
ffn_adapter_init_option="bert"
ffn_adapter_scalar="1"
 
# The number of neurons to keep (the dim of W2a and W1a)
# W2 = [W2a, W2b], W1^T = [W1a^T, W1b^T]
# W2 W1 = (W2a W1a^T) + (W2b W1b^T) is now sketched as
# (W2a W1a^T) + (W2b SS^T W1b^T)
mid_dim=256

# Reinitialized layers:
re_init_layers=,
 
# lora params are not set
if [ -z ${lora_alpha+x} ];
then
    lora_alpha=0
    lora_init="lora"
    lora_dropout=0
fi
 
# set to "wandb" to use weights & bias
# report_to="none"

num_train_epochs=10
# Batch size is fixed at 32
bsz=32
gradient_steps=1
# Learning rate is searched in the range: 1e-5, 2e-5,4e-5, 6e-5, 8e-5
lr=1e-5
lr_scheduler_type="polynomial"
weight_decay=0.1
warmup_ratio=0.06
max_tokens_per_batch=4400
max_seq_length=512
 
unfreeze='ef_'
 
extra_cmd=""
debug_str=""
 
TASK_ARGS=" --gradient_accumulation_steps ${gradient_steps} \
            --evaluation_strategy epoch \
            --save_strategy epoch \
            --save_steps 5000 \
            --eval_steps 5000 \
            --logging_steps 50 \
            --report_to none"
 
 
 
# set to 1 for debug mode which only uses 1600 training examples
debug=0
 
if [ "${debug}" = 1 ];
then
lr=1e-1
bsz=16
gradient_steps=1
weight_decay=0
num_train_epochs=2
debug_str="debug"
TASK_ARGS=" --mid_dim 256 \
            --max_grad_norm 1 \
            --max_train_samples 1000 \
            --max_eval_samples 150 \
            --max_steps -1 \
            --evaluation_strategy steps \
            --save_strategy steps \
            --save_steps 100 \
            --eval_steps 100 \
            --logging_steps 10 \
            --report_to none \
            --max_predict_samples 150"
fi
 
 
exp_name=glue.${TASK_NAME}.am_${attn_mode}.ao_${attn_option}.fm_${ffn_mode}
exp_name=${exp_name}".fo_${ffn_option}.abn${preseqlen}.fbn${ffn_bn_len}.ac_${attn_composition}"
exp_name=${exp_name}".fl_${ffn_adapter_layernorm_option}.finit_${ffn_adapter_init_option}"
exp_name=${exp_name}".fs_${ffn_adapter_scalar}.unfrz_${unfreeze}.ne${num_train_epochs}"
exp_name=${exp_name}".bsz${bsz}.grad_steps${gradient_steps}.lr${lr}"
exp_name=${exp_name}".warm${warmup_ratio}.wd${weight_decay}.seed${seed}.${debug_str}"
SAVE=checkpoints/glue/${TASK_NAME}/${DATE}/${exp_name}
echo "${SAVE}"
 
 
cuda=2

CUDA_VISIBLE_DEVICES=$cuda python -u run_glue.py \
    ${TASK_ARGS} \
    --model_name_or_path roberta-base \
    --task_name $TASK_NAME \
    --model_type roberta_custom \
    --do_train \
    --do_eval \
    --do_predict \
    --approx_check ${approx_check}\
    --distill ${distill}\
    --max_seq_length ${max_seq_length} \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size ${bsz} \
    --max_tokens_per_batch ${max_tokens_per_batch} \
    --weight_decay ${weight_decay} \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --attn_mode ${attn_mode} \
    --attn_option ${attn_option} \
    --attn_composition ${attn_composition} \
    --ffn_mode ${ffn_mode} \
    --ffn_option ${ffn_option} \
    --sketch_layers ${sketch_layers} \
    --n_steps ${n_steps} \
    --ffn_adapter_layernorm_option ${ffn_adapter_layernorm_option} \
    --ffn_adapter_scalar ${ffn_adapter_scalar} \
    --ffn_adapter_init_option ${ffn_adapter_init_option} \
    --attn_bn ${attn_bn} \
    --ffn_bn ${ffn_bn} \
    --load_clustering ${load_clustering} \
    --unfreeze_params ${unfreeze} \
    --mid_dim ${mid_dim} \
    --re_init_layers ${re_init_layers} \
    --seed ${seed} \
    --num_train_epochs ${num_train_epochs} \
    --learning_rate ${lr} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --warmup_ratio ${warmup_ratio} \
    --max_seq_length ${max_seq_length} \
    --fp16 \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --run_name ${TASK_NAME}.${DATE}.${exp_name} \
    --overwrite_output_dir \
    --disable_tqdm "True" \
    --metric_for_best_model ${metric} \
    --greater_is_better "True" \
    --ddp_find_unused_parameter "False" \
    --output_dir ${SAVE} ${extra_cmd} \
        2>&1 | tee ${SAVE}/log.txt
