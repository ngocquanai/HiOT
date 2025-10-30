#!/bin/bash -e

#SBATCH --job-name=hiot# create a short name for your job
#SBATCH --output=/lustre/scratch/client/movian/research/users/quanpn2/public/HiOT/sbatch_results/hiot.out # create a output file
#SBATCH --error=/lustre/scratch/client/movian/research/users/quanpn2/public/HiOT/sbatch_results/hiot.err # create a error file
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


module purge
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"



conda activate hcast
cd /lustre/scratch/client/movian/research/users/quanpn2/public/HiOT

export PYTHONPATH=/lustre/scratch/client/movian/research/users/quanpn2/public/HiOT

torchrun --nproc_per_node=2 deit/main_suppix_hier.py \
  --model cast_small \
  --batch-size 256 \
  --epochs 100 \
  --num-superpixels 196 --num_workers 12 \
  --data-set INAT21-MINI-HIER-SUPERPIXEL \
  --data-path ../dataset/ \
  --output_dir ./output/inat21_mini_hcast \
  --ot_loss --ot_weight 0.5 \
  --finetune best_checkpoint.pth --distributed

  # --globalkl --gk_weight 0.5 \

