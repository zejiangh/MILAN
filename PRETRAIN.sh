# pretrain MILAN on ImageNet-1K dataset

# vit_base_patch16
python -m torch.distributed.launch --nproc_per_node=8 --use_env main_pretrain.py \
    --model mae_vit_base_patch16 \
    --batch_size 256 \
    --accum_iter 2 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --data_path /mnt/zejiang.hou/dataset/imagenet \
    --output_dir ./milan_vit_base_pretrain_400epoch_useclip_changedecoder_attnmask --log_dir ./milan_vit_base_pretrain_400epoch_useclip_changedecoder_attnmask \
    --use_clip \
    --change_decoder \
    --attn_mask

# vit_large_patch16
python -m torch.distributed.launch --nproc_per_node=8 --use_env main_pretrain.py \
    --model mae_vit_large_patch16 \
    --batch_size 128 \
    --accum_iter 4 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --data_path /data/imagenet \
    --output_dir ./milan_vit_large_pretrain_400epoch_useclip_changedecoder_attnmask --log_dir ./milan_vit_large_pretrain_400epoch_useclip_changedecoder_attnmask \
    --use_clip \
    --change_decoder \
    --attn_mask





