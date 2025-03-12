#!/bin/bash

#SBATCH --job-name=bird_hcast_seed0
#SBATCH --mail-user=seulki@umich.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24g
#SBATCH --time=240:00:00
#SBATCH --account=stellayu
#SBATCH --partition=stellayu
#SBATCH --output=./logs/%x-%j.log

export PYTHONPATH=deit/:$PYTHONPATH
export PYTHONPATH=deit/dataset/:$PYTHONPATH


module purge
module load cuda/12.1.1 gcc/11.2.0
source activate py310

python deit/main_suppix_hier.py \
  --model cast_small \
  --batch-size 256 \
  --epochs 100 \
  --num-superpixels 196 --num_workers 8 \
  --globalkl --gk_weight 0.5 \
  --data-set BIRD-HIER-SUPERPIXEL \
  --data-path /nfs/turbo/coe-stellayu/shared_data/CUB_200_2011/CUB_200_2011/images_split \
  --output_dir ./output/bird_hcast \
  --finetune /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/best_checkpoint.pth
  

