# CUDA_VISIBLE_DEVICES=0 python3 train.py --config configs/inducer-mam.json
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch \
#     --nproc_per_node=4 train.py --config configs/inducer-mam.json

# CUDA_VISIBLE_DEVICES=0 python3 evaluate.py configs/inducer-mam.json

CUDA_VISIBLE_DEVICES=3 python3 train.py --config configs/sketchFT.json
CUDA_VISIBLE_DEVICES=3 python3 evaluate.py configs/sketchFT.json

# CUDA_VISIBLE_DEVICES=1 python3 train.py --config configs/mmdFT.json
# CUDA_VISIBLE_DEVICES=1 python3 evaluate.py configs/mmdFT.json


# CUDA_VISIBLE_DEVICES=0 python3 train.py --config configs/clusterFT.json
# CUDA_VISIBLE_DEVICES=0 python3 evaluate.py configs/clusterFT.json

# CUDA_VISIBLE_DEVICES=0 python3 train.py --config configs/ntkclusterFT.json
# CUDA_VISIBLE_DEVICES=0 python3 evaluate.py configs/ntkclusterFT.json

# CUDA_VISIBLE_DEVICES=3 python3 train.py --config configs/sketchFT.json --distillation
