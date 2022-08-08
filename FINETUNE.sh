# finetune MILAN on ImageNet-1K

# vit_base_patch16
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
    --accum_iter 1 \
    --batch_size 128 \
    --model vit_base_patch16 \
    --finetune ./milan_vit_base_pretrain_400epoch_useclip_changedecoder_attnmask/checkpoint-399.pth \
    --epochs 100 \
    --blr 1e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path /data/imagenet \
    --output_dir ./milan_vit_base_finetune_pretrain400epochuseclipchangedecoderattnmask --log_dir ./milan_vit_base_finetune_pretrain400epochuseclipchangedecoderattnmask \
    --global_pool

# vit_large_patch16
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
    --accum_iter 2 \
    --batch_size 64 \
    --model vit_large_patch16 \
    --finetune ./milan_vit_large_pretrain_400epoch_useclip_changedecoder_attnmask/checkpoint-399.pth \
    --epochs 100 \
    --blr 1e-4 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path /data/imagenet \
    --output_dir ./milan_vit_large_finetune_pretrain400epochuseclipchangedecoderattnmask --log_dir ./milan_vit_large_finetune_pretrain400epochuseclipchangedecoderattnmask \
    --global_pool

# linear probing
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_linprobe.py \
    --accum_iter 1 \
    --batch_size 2048 \
    --model vit_base_patch16 \
    --cls_token \
    --finetune ./milan_vit_base_pretrain_400epoch_useclip_changedecoder_attnmask/checkpoint-399.pth \
    --epochs 100 \
    --blr 0.05 \
    --weight_decay 0.0 \
    --dist_eval --data_path /data/imagenet \
    --output_dir ./milan_vit_base_linearprobe_pretrain400epochuseclipchangedecoderattnmask --log_dir ./milan_vit_base_linearprobe_pretrain400epochuseclipchangedecoderattnmask

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_linprobe.py \
    --accum_iter 1 \
    --batch_size 2048 \
    --model vit_large_patch16 \
    --cls_token \
    --finetune ./milan_vit_large_pretrain_400epoch_useclip_changedecoder_attnmask/checkpoint-399.pth \
    --epochs 100 \
    --blr 0.02 \
    --weight_decay 0.0 \
    --dist_eval --data_path /data/imagenet \
    --output_dir ./milan_vit_large_linearprobe_pretrain400epochuseclipchangedecoderattnmask --log_dir ./milan_vit_large_linearprobe_pretrain400epochuseclipchangedecoderattnmask


