CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master-port=11113 train.py --model DiT-XL/2
--data-path ./datasets/Ithaca365/Ithaca365-scenario --boxes-path ./datasets/box_info  \
--epochs 3000 --global-batch-size 16 --lr 1e-5 --log-every 50 --ckpt-every 100 \
--resume-checkpoint ./pretrained_models/DiT-XL-2-256x256.pt \
--vae-checkpoint ./pretrained_models/sd-vae-ft-ema \
--embed-checkpoint ./pretrained_models/clip_similarity_embed.pt \
--dataset_name ithaca365 --training_sample_steps 500 --scenario_num 5 --rank 2 --modulation \
--cond_mlp_modulation --rope --finetune_depth 28 --mask_rl 2 --noise_schedule progress