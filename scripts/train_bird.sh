#!/bin/bash

#SBATCH --job-name=bird_hcast
#SBATCH --mail-user=seulki@umich.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --ntasks=1 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24g
#SBATCH --time=12:00:00
#SBATCH --account=145466848039
#SBATCH --output=./logs/%x-%j.log

export PYTHONPATH=deit/:$PYTHONPATH


module purge
module load CUDA/11.6.0

source activate cast

Birds
python deit/main_suppix_hier.py \
  --model cast_small \
  --batch-size 256 \
  --epochs 100 \
  --num-superpixels 196 --num_workers 8 \
  --globalkl --gk_weight 0.5 \
  --data-set BIRD-HIER-SUPERPIXEL \
  --data-path /scratch/user/u.sp270400/data/CUB_200_2011/images \
  --output_dir ./output/bird_hcast \
  --finetune best_checkpoint.pth
  

