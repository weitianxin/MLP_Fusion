# Our method:
ffn_mode='ntk_cluster'
# Other methods:
# ffn_mode="sketch"
# ffn_mode='cluster'
# ffn_mode='mmd'
# ffn_mode='svd'
# ffn_mode=''

# Select from [mrpc, mnli, sst2, cola]
dataset="mnli"

device=3
batch=64
lr=1e-4
epoch=10
seed=42
distill=1


python sft.py --device "$device" --batch "$batch" --lr "$lr" --epoch "$epoch" --seed "$seed" --mode "$ffn_mode" --dataset "$dataset" --distill "$distill"