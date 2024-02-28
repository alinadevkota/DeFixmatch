#!/bin/bash -lT

#SBATCH -J PI
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -n 1
#SBATCH -p gpu_7day
#SBATCH --gpus 4
#SBATCH --mem-per-gpu 24G
#SBATCH -w dscog011
#SBATCH -t 48:00:00

conda activate retfound2

nvidia-smi

python train.py --world-size 1 --rank 0 --dataset APTOS2019 --num_classes 5 --overwrite --modified_fixmatch --ulb_loss_ratio 0.5 --data_dir /users/ad00139/datasets/APTOS2019_50percent --net RetFound --save_dir saved_models_50percent --retfound_dir /users/ad00139/RETFound_MAE/RETFound_cfp_weights.pth --debiased
