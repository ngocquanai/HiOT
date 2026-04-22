#!/bin/bash -e

#SBATCH --job-name=imagenet# create a short name for your job
#SBATCH --output=/lustre/scratch/client/movian/research/users/quanpn2/public/HiOT/hiot-imagenet/sbatch/pretrained_ot0.1.out # create a output file
#SBATCH --error=/lustre/scratch/client/movian/research/users/quanpn2/public/HiOT/hiot-imagenet/sbatch/pretrained_ot0.1.err # create a error file
#SBATCH --partition=movianr # choose partition
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=128GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10-00:00          # total run time limit (DD-HH:MM)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail          # send email when job fails
#SBATCH --mail-user=v.quanpn2@vinai.io
#SBATCH --exclude=sdc2-hpc-dgx-a100-004

module purge
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"



conda activate /lustre/scratch/client/movian/research/users/quanpn2/virtual/hcast
cd /lustre/scratch/client/movian/research/users/quanpn2/public/HiOT/hiot-imagenet




DATA_PATH="/lustre/scratch/client/movian/research/users/quanpn2/public/dataset/imagenet-1k/imagenet-1k"
OUTPUT_PATH="./local/pretrain_ot0.1"


## OT LOSS uniform weight
CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.launch --master_port 25505 --nproc_per_node=2 --use_env main.py \
--model deit_tiny_patch16_224 --batch-size 1024 --lr 3e-5 --data-path $DATA_PATH --output_dir $OUTPUT_PATH \
--ot-loss --tree_folder_path ./tree --base-weight 1 --ot-weight 0.1 --finetune ./local/base_runtime/18_01_2026_01:41:35_deit_tiny_patch16_224_checkpoint.pth



# ## OT LOSS learnable weight
# CUDA_VISIBLE_DEVICES='2,3' python -m torch.distributed.launch --master_port 29505 --nproc_per_node=2 --use_env main.py \
# --model deit_tiny_patch16_224 --batch-size 1024 --data-path $DATA_PATH --output_dir $OUTPUT_PATH \
# --ot-loss --tree_folder_path ./tree --ot-learnable-w --base-weight 1 --ot-weight 1

chmod -R 777 /lustre/scratch/client/movian/research/users/quanpn2/public/HiOT/hiot-imagenet/sbatch
chmod -R 777 ./local




